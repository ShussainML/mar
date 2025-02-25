from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from timm.models.vision_transformer import Block
from models.diffloss import DiffLoss  # Ensure this import is correct

# Debugging utility function
def debug_print(tensor, name, shape_only=False):
    """Utility function to print tensor details for debugging."""
    print(f"[Debug] {name}:")
    print(f"  - Shape: {tensor.shape}")
    if not shape_only:
        print(f"  - Dtype: {tensor.dtype}")
        print(f"  - Device: {tensor.device}")
        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
            print(f"  - Values (min/max/mean): {tensor.min().item():.4f}, {tensor.max().item():.4f}, {tensor.mean().item():.4f}")
        else:
            print(f"  - Values (min/max): {tensor.min().item()}, {tensor.max().item()}")

def mask_by_order(mask_len, order, bsz, seq_len):
    """Create a mask based on the generation order."""
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    def __init__(self, img_size=64, vae_stride=16, patch_size=2,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=8,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='50',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 device=None):
        super().__init__()

        # Set device
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

        # Compute seq_len and token_embed_dim before they are used
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w  # Compute seq_len here
        self.vae_embed_dim = vae_embed_dim
        self.token_embed_dim = vae_embed_dim * patch_size**2  # Compute token_embed_dim here

        # Patch projection layer
        self.patch_proj = nn.Conv2d(
            in_channels=3,  # Assuming input images have 3 channels
            out_channels=vae_embed_dim,
            kernel_size=1,  # 1x1 convolution to project channels
            stride=vae_stride  # Match the VAE stride to maintain spatial dimensions
        )

        # Rest of the initialization...
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # Masking ratio generator
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # Encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # Decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        # Initialize weights
        self.initialize_weights()

        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        """Initialize weights for all layers."""
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for specific layer types."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, bsz):
        """Generate random generation orders."""
        orders = [np.random.permutation(self.seq_len) for _ in range(bsz)]
        return torch.tensor(orders, dtype=torch.long, device=self.device)

    def forward(self, imgs, labels):
        """Forward pass for the MAR model."""
        debug_print(imgs, "Input images")
        debug_print(labels, "Input labels")
        class_embedding = self.class_emb(labels)
        debug_print(class_embedding, "Class embedding")
        x = self.patchify(imgs)
        debug_print(x, "Patchified input")
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)
        debug_print(mask, "Random mask")
        x = self.forward_mae_encoder(x, mask, class_embedding)
        z = self.forward_mae_decoder(x, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        """Sample tokens using the MAR model."""
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        for step in indices:
            cur_tokens = tokens.clone()
            class_embedding = self.class_emb(labels) if labels is not None else self.fake_latent.repeat(bsz, 1)
            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            x = self.forward_mae_encoder(tokens, mask, class_embedding)
            z = self.forward_mae_decoder(x, mask)
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool()) if step < num_iter - 1 else mask[:bsz].bool()
            mask = mask_next
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        tokens = self.unpatchify(tokens)
        return tokens


# Model variants
def mar_base(**kwargs):
    return MAR(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
               decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def mar_large(**kwargs):
    return MAR(encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
               decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def mar_huge(**kwargs):
    return MAR(encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
               decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
