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
# Import DiffLoss
from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,  # Number of classes (can be adjusted)
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 device=None):
        super().__init__()
        # Add a projection layer to map 3 channels -> vae_embed_dim
        self.patch_proj = nn.Conv2d(
            in_channels=3,  # Assuming input images have 3 channels
            out_channels=vae_embed_dim,
            kernel_size=1,  # 1x1 convolution to project channels
            stride=vae_stride  # Match the VAE stride to maintain spatial dimensions
        )
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        # Rest of the initialization...
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps  # Removed grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        Convert images into patches and project to vae_embed_dim.
        Returns:
            Tensor of shape (batch_size, num_patches, token_embed_dim)
        """
        # Debug 1: Initial input shape
        print(f"\n[Debug] Initial input shape: {x.shape} (bsz, channels, height, width)")
        
        # Step 1: Project channels using Conv2d
        print(f"[Debug] patch_proj layer parameters:")
        print(f"  - in_channels: {self.patch_proj.in_channels}")
        print(f"  - out_channels: {self.patch_proj.out_channels}")
        print(f"  - kernel_size: {self.patch_proj.kernel_size}")
        print(f"  - stride: {self.patch_proj.stride}")
        
        x = self.patch_proj(x)
        print(f"[Debug] Shape after patch_proj: {x.shape} (bsz, vae_embed_dim, h_proj, w_proj)")
    
        # Step 2: Unfold into patches
        bsz, c, h, w = x.shape
        p = self.patch_size
        s = self.vae_stride
    
        # Debug 2: Before unfolding
        print(f"\n[Debug] Before unfolding:")
        print(f"  - Patch size: {p}")
        print(f"  - Stride: {s}")
        print(f"  - Input height: {h}, width: {w}")
    
        # First unfold (height dimension)
        x = x.unfold(2, p, s)
        print(f"[Debug] After height unfolding: {x.shape} (bsz, c, h_patches, w, p)")
    
        # Second unfold (width dimension)
        x = x.unfold(3, p, s)
        print(f"[Debug] After width unfolding: {x.shape} (bsz, c, h_patches, w_patches, p, p)")
    
        # Permute and reshape
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        print(f"[Debug] After permute: {x.shape} (bsz, h_patches, w_patches, c, p, p)")
    
        # Final reshape
        original_elements = x.numel()
        x = x.view(bsz, -1, self.token_embed_dim)
        new_elements = x.numel()
        
        # Debug 3: After final reshape
        print(f"\n[Debug] After final reshape:")
        print(f"  - Expected token_embed_dim: {self.token_embed_dim} (vae_embed_dim * p^2 = {self.vae_embed_dim}*{p**2})")
        print(f"  - Actual last dimension: {x.shape[-1]}")
        print(f"  - Element count check: {original_elements} -> {new_elements} "
              f"({'OK' if original_elements == new_elements else 'MISMATCH'})")
        
        print(f"\n[Debug] Final patchified shape: {x.shape} (bsz, num_patches, token_embed_dim)")
        
        return x


    def unpatchify(self, x):
        """
        Convert patches back into images.
        Args:
            x: Input tensor of shape (batch_size, num_patches, patch_embed_dim).
        Returns:
            Tensor of shape (batch_size, channels, height, width).
        """
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_patches = int(np.sqrt(x.shape[1]))  # Compute number of patches dynamically
        w_patches = h_patches
    
        x = x.reshape(bsz, h_patches, w_patches, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_patches * p, w_patches * p)
        return x

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        print(f"Shape of x before z_proj: {x.shape}")  # Debug shape
        x = self.z_proj(x)
        print(f"Shape of x after z_proj: {x.shape}")  # Debug shape
        
        bsz, seq_len, embed_dim = x.shape
        print(f"Batch size: {bsz}, Sequence length: {seq_len}, Embedding dim: {embed_dim}")  # Debug shape
        
        # Concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        print(f"Shape of x after concat buffer: {x.shape}")  # Debug shape
        
        # Add class embedding
        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        print(f"Shape of x after adding class embedding: {x.shape}")  # Debug shape
        
        # Encoder position embedding
        print(f"Shape of encoder_pos_embed_learned: {self.encoder_pos_embed_learned.shape}")  # Debug shape
        if x.shape[1] != self.encoder_pos_embed_learned.shape[1]:
            raise ValueError(f"Shape mismatch: x has seq_len {x.shape[1]}, but encoder_pos_embed_learned has seq_len {self.encoder_pos_embed_learned.shape[1]}")
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)
    
        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
    
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)
    
        return x
        
    def forward_mae_decoder(self, x, mask):
        print(f"Shape of x before decoder_embed: {x.shape}")  # Debug shape
        x = self.decoder_embed(x)
        print(f"Shape of x after decoder_embed: {x.shape}")  # Debug shape
        
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
        print(f"Shape of mask_with_buffer: {mask_with_buffer.shape}")  # Debug shape
    
        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        print(f"Shape of mask_tokens: {mask_tokens.shape}")  # Debug shape
        
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        print(f"Shape of x_after_pad: {x_after_pad.shape}")  # Debug shape
    
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned
        print(f"Shape of x after decoder_pos_embed_learned: {x.shape}")  # Debug shape
    
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)
    
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        print(f"Shape of x after diffusion_pos_embed_learned: {x.shape}")  # Debug shape
        
        return x

    def forward_loss(self, z, target, mask):
        print(f"Shape of z: {z.shape}")  # Debug shape
        print(f"Shape of target: {target.shape}")  # Debug shape
        print(f"Shape of mask: {mask.shape}")  # Debug shape
        
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        
        print(f"Shape of target after reshape: {target.shape}")  # Debug shape
        print(f"Shape of z after reshape: {z.shape}")  # Debug shape
        print(f"Shape of mask after reshape: {mask.shape}")  # Debug shape
        
        loss = self.diffloss(z=z, target=target, mask=mask)
        print(f"Shape of loss: {loss.shape if isinstance(loss, torch.Tensor) else 'scalar'}")  # Debug shape
        
        return loss

    def forward(self, imgs, labels):
        # Debug input shapes
        print(f"Input imgs shape: {imgs.shape}")
        print(f"Input labels shape: {labels.shape}")
    
        # class embed
        class_embedding = self.class_emb(labels)
        print(f"Class embedding shape: {class_embedding.shape}")
    
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        print(f"Shape of x after patchify: {x.shape}")
        
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)
        print(f"Shape of mask: {mask.shape}")
    
        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)
    
        # mae decoder
        z = self.forward_mae_decoder(x, mask)
    
        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
    
        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
