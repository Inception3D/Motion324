import os
import math
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from .transformer import (QK_Norm_TransformerBlock, init_weights, 
                          QK_Norm_CrossAttentionBlock)
from .loss import MSELossComputer
from .image_encoder.dinov2 import DinoEncoder

class FrequencyPositionalEmbedding(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self._get_dims(input_dim)

    def _get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device)).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

def get_sinusoidal_time_embed(num_frames, embed_dim, device):
    """
    Generate time embedding with shape=[num_frames, embed_dim].
    """
    position = torch.arange(num_frames, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(num_frames, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, C]

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.
    """
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        """
        Args:
            d_model (int): Model dimension (embedding dimension).
            max_len (int): Maximum sequence length supported.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even to use paired sin/cos for sinusoidal positional encoding.")

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Shape: [d_model / 2]
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        Args:
            x: Input tensor with shape [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor: Output tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=768):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed
        
class PatchEmbed(nn.Module):#4 patch_length normal
    """Video patch embedding layer."""
    def __init__(self, video_size=224, video_length=16, patch_size=16, patch_length=4, 
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.video_length = video_length
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_h = video_size // patch_size
        self.num_patches_w = video_size // patch_size
        self.num_patches_t = video_length // patch_length
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        
        # Patch embedding layer
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=(patch_length, patch_size, patch_size),
                             stride=(patch_length, patch_size, patch_size))
    
    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape
        assert T == self.video_length and H == self.video_size and W == self.video_size, \
            f"Input size mismatch: got {T}x{H}x{W}, expected {self.video_length}x{self.video_size}x{self.video_size}"
        
        x = self.proj(x)  # B, embed_dim, T/pT, H/pH, W/pW
        return x

def resize_pos_embed(posemb, src_shape, target_shape):
    """Resize position embedding to match target shape."""
    posemb = posemb.reshape(1, src_shape[0], src_shape[1], src_shape[2], -1)
    posemb = posemb.permute(0, 4, 1, 2, 3)
    posemb = nn.functional.interpolate(posemb, size=target_shape, mode='trilinear', align_corners=False)
    posemb = posemb.permute(0, 2, 3, 4, 1)
    posemb = posemb.reshape(1, target_shape[0] * target_shape[1] * target_shape[2], -1)
    return posemb

def generate_pos_embed(T, H, W, embed_dim):
    """Generate 3D position embedding."""
    latent_t = torch.arange(T, dtype=torch.float32)
    latent_h = torch.arange(H, dtype=torch.float32)
    latent_w = torch.arange(W, dtype=torch.float32)
    
    if T > 1:
        latent_t = 2 * (latent_t / (T - 1)) - 1
    else:
        latent_t = torch.tensor([0.0], dtype=torch.float32)

    if H > 1:
        latent_h = 2 * (latent_h / (H - 1)) - 1
    else:
        latent_h = torch.tensor([0.0], dtype=torch.float32)

    if W > 1:
        latent_w = 2 * (latent_w / (W - 1)) - 1
    else:
        latent_w = torch.tensor([0.0], dtype=torch.float32)

    t, h, w = torch.meshgrid(latent_t, latent_h, latent_w, indexing='ij')
    
    pos = torch.stack([t, h, w], dim=-1)

    # Apply Fourier feature mapping
    freq_bands = 2.0 ** torch.linspace(0.0, 7.0, embed_dim // 6)
    freq_bands = freq_bands.view(1, 1, 1, 1, -1)
    
    # Project to higher dimensions using sin and cos
    pos = pos.unsqueeze(-1) * freq_bands
    pos = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)
    
    # Reshape to 1D sequence
    pos = pos.reshape(1, -1, embed_dim)
    
    return pos

class Motion_Latent_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat_dim = self.config.model.feat_dim
        
        self._init_video_patchify()
    
        transformer_config = self.config.model.video_encoder.transformer
        use_qk_norm = transformer_config.get("use_qk_norm", True)
        d_model = transformer_config.d
        d_head = transformer_config.d_head

        use_qk_norm = transformer_config.get("use_qk_norm", True)

        self.point_embed = PointEmbed(dim=transformer_config.d)
        self.point_normal_rgb_proj = nn.Linear(transformer_config.d + 3 + 3, transformer_config.d)
        self.point_normal_rgb_proj.apply(init_weights)

        self.num_learnable_tokens = config.model.tokens
        self.learnable_tokens = nn.Parameter(torch.randn(1, self.num_learnable_tokens, d_model))

        # Special tokens for t=0 and t>0. Each is (1, 4, d_model).
        self.special_token_0 = nn.Parameter(torch.randn(1, 4, d_model))
        self.special_token_rest = nn.Parameter(torch.randn(1, 4, d_model))

        self.encoder_cross_attn = QK_Norm_CrossAttentionBlock(
            dim=d_model,
            head_dim=d_head,
            kv_dim=d_model,
            use_qk_norm=use_qk_norm
        )

        self.points_transformer_blocks = nn.ModuleList([
            QK_Norm_TransformerBlock(
                transformer_config.d, transformer_config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.model.pcd_layers)
        ])
        self.points_transformer_blocks.apply(init_weights)

        self.image_encoder = DinoEncoder(patch_size=14)
        self.alternating_layers = transformer_config.get("n_layer", 12)
        assert self.alternating_layers % 2 == 0, "Alternating layers should be even."
        
        self.global_transformer_blocks = nn.ModuleList([
            QK_Norm_TransformerBlock(
                transformer_config.d, transformer_config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(self.alternating_layers // 2)
        ])
        self.global_transformer_blocks.apply(init_weights)

        self.local_transformer_blocks = nn.ModuleList([
            QK_Norm_TransformerBlock(
                transformer_config.d, transformer_config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(self.alternating_layers // 2)
        ])
        self.local_transformer_blocks.apply(init_weights)

        self.transformer_input_layernorm = nn.LayerNorm(transformer_config.d, bias=False)
        self.transformer_input_layernorm.apply(init_weights)

        self.decoder_cross_attn = QK_Norm_CrossAttentionBlock(
            dim=d_model,
            head_dim=d_head,
            kv_dim=d_model,
            use_qk_norm=use_qk_norm
        )

        self.shared_mlp_output = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, 3),
        )
        self.shared_mlp_output.apply(init_weights)

        self.loss_computer = MSELossComputer(self.config)

    def _init_video_patchify(self):
        video_config = self.config.model.video_encoder
        transformer_config = video_config.transformer
        token_config = video_config.image_tokenizer
        
        self.image_size = token_config.get("image_size", 224)
        self.video_length = self.config.training.frames
        self.patch_size = token_config.get("patch_size", 14)
        self.patch_length = token_config.get("patch_length", 1)
        self.embed_dim = transformer_config.d
        
        self.num_patches_h = self.image_size // self.patch_size
        self.num_patches_w = self.image_size // self.patch_size
        self.num_patches_t = self.video_length // self.patch_length
        
        self.latent_length = self.num_patches_t
        self.latent_size = self.num_patches_h  # Assuming square patches
        
        pos_embed = generate_pos_embed(
            self.latent_length, self.latent_size, self.latent_size, self.embed_dim
        )
        self.register_buffer('pos_embed', pos_embed)
        
        drop_rate = transformer_config.get("drop_rate", 0.1)
        self.pos_drop = nn.Dropout(p=drop_rate)

    def train(self, mode=True):
        super().train(mode)

    def pass_transformer_layers(self, blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        num_layers = len(blocks)
        if not gradient_checkpoint:
            for layer in blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        def _process_layer_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = blocks[idx](tokens)
            return tokens
            
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group, input_tokens, start_idx, end_idx, use_reentrant=False
            )
        return input_tokens
        
    def pass_alternating_attention(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        B, T, L, C = input_tokens.shape
        num_loops = len(self.global_transformer_blocks)

        if not gradient_checkpoint:
            processed_tokens = input_tokens
            for i in range(num_loops):
                processed_tokens = processed_tokens.view(B, T * L, C)
                processed_tokens = self.global_transformer_blocks[i](
                    processed_tokens
                )
                processed_tokens = processed_tokens.view(B, T, L, C)
                processed_tokens_reshaped = processed_tokens.view(B * T, L, C)
                processed_tokens_local = self.local_transformer_blocks[i](processed_tokens_reshaped)
                processed_tokens = processed_tokens_local.view(B, T, L, C)
            return processed_tokens

        def _process_alternating_group(tokens, start_idx, end_idx):
            for i in range(start_idx, end_idx):
                tokens = tokens.view(B, T * L, C)
                tokens = self.global_transformer_blocks[i](
                    tokens
                )
                tokens = tokens.view(B, T, L, C)
                tokens_reshaped = tokens.view(B * T, L, C)
                tokens_processed = self.local_transformer_blocks[i](tokens_reshaped)
                tokens = tokens_processed.view(B, T, L, C)
            return tokens
        
        processed_tokens = input_tokens
        for start_idx in range(0, num_loops, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_loops)
            processed_tokens = torch.utils.checkpoint.checkpoint(
                _process_alternating_group, processed_tokens, start_idx, end_idx, use_reentrant=False
            )
        return processed_tokens

    def pass_pcd_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        num_layers = len(self.points_transformer_blocks)
        if not gradient_checkpoint:
            for layer in self.points_transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        def _process_layer_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = self.points_transformer_blocks[idx](tokens)
            return tokens
            
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group, input_tokens, start_idx, end_idx, use_reentrant=False
            )
        return input_tokens

    def forward(self, sample):
        B = sample['ref_pcd'].shape[0]
        N_points = sample['ref_pcd'].shape[1]
        device = sample['ref_pcd'].device
        tokens = self.learnable_tokens.shape[1]
        
        point_embedding = self.point_embed(sample['ref_shape_pcd'])
        checkpoint_every = self.config.training.grad_checkpoint_every
        
        point_normal_rgb_embedding = self.point_normal_rgb_proj(torch.cat([point_embedding, sample['ref_shape_normals'], sample['ref_shape_rgbs']], dim=-1))
        
        query_tokens = self.learnable_tokens.expand(B, -1, -1).to(device)
        mesh_point_feat = self.encoder_cross_attn(query_tokens, point_normal_rgb_embedding, point_normal_rgb_embedding)
        use_checkpoint = self.config.training.get('use_checkpoint', True)
        mesh_feat = self.pass_pcd_layers(mesh_point_feat, gradient_checkpoint=use_checkpoint, checkpoint_every=checkpoint_every)

        rgb_video = sample['rgb_video']  # B, T, H, W, C
        B_vid, T_in, H_in, W_in, C_rgb = rgb_video.shape
        #assert self.config.training.frames == T_in, "Mismatch between configured frames and input video frames."

        tokenizer_input = rgb_video.permute(0, 1, 4, 2, 3)
        tokenizer_input = tokenizer_input.reshape(B_vid * T_in, tokenizer_input.shape[-3], tokenizer_input.shape[-2], tokenizer_input.shape[-1])
        tokenizer_input = F.interpolate(tokenizer_input, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            image_tokens_raw = self.image_encoder(tokenizer_input)
        
        x = image_tokens_raw.view(B_vid, T_in, self.num_patches_h, self.num_patches_w, image_tokens_raw.shape[-1])
        x = x.permute(0, 4, 1, 2, 3)
        latentT, latentH, latentW = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)
        if latentT != self.latent_length or latentH != self.latent_size or latentW != self.latent_size:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                src_shape=(self.latent_length, self.latent_size, self.latent_size),
                target_shape=(latentT, latentH, latentW),
            )
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        x = self.pos_drop(x)
        video_tokens_projected = x
        num_video_patches_per_frame = video_tokens_projected.shape[1] // T_in
        video_tokens_per_frame = video_tokens_projected.view(B, T_in, num_video_patches_per_frame, -1)  # [B,T,L_img,C]

        special_token_0 = self.special_token_0.expand(B, 4, -1)  # [B, 4, C]
        special_token_rest = self.special_token_rest.expand(B, 4, -1)  # [B, 4, C]

        all_special_tokens = torch.stack([special_token_0] + [special_token_rest]*(T_in-1), dim=1)  # [B, T_in, 4, C]

        # mesh_feat: [B, tokens, C] -> [B, T_in, tokens, C]
        pcd_tokens_repeated = mesh_feat.unsqueeze(1).expand(B, T_in, tokens, mesh_feat.shape[-1])  # [B, T_in, tokens, C]
        # [B, T_in, tokens+4+L_img, C]
        concatenated_tokens = torch.cat([
            all_special_tokens,
            pcd_tokens_repeated,                         # [B, T_in, tokens, C]
            video_tokens_per_frame                       # [B,T_in,L_img,C]
        ], dim=2)

        concatenated_tokens = self.transformer_input_layernorm(concatenated_tokens)
        B, T, L, C = concatenated_tokens.shape
        
        use_checkpoint = self.config.training.get('use_checkpoint', True)
        checkpoint_every = self.config.training.get('grad_checkpoint_every', 1)

        processed_tokens_BTLC = self.pass_alternating_attention(
            input_tokens=concatenated_tokens,
            gradient_checkpoint=use_checkpoint,
            checkpoint_every=checkpoint_every
        )
        pcd_tokens_final = processed_tokens_BTLC[:, :, 4:4+tokens, :]

        output_pcds = sample['ref_pcd']
        output_normals = sample['ref_normal']
        output_rgbs = sample['ref_rgb']

        B, N, _ = output_pcds.shape
        chunk_size = 4096

        def decode_chunk(pcd_chunk, normal_chunk, rgb_chunk, pcd_tokens_for_decode):
            B = pcd_chunk.shape[0]
            N = pcd_chunk.shape[1]
            T = T_in
            # Repeat to [B, T_in, N, C]
            pcd_chunk_proj = pcd_chunk[:, None, :, :].expand(-1, T, -1, -1)
            normal_chunk_proj = normal_chunk[:, None, :, :].expand(-1, T, -1, -1)
            rgb_chunk_proj = rgb_chunk[:, None, :, :].expand(-1, T, -1, -1)

            outputs = []
            for t in range(T):
                # For each time step, get corresponding token slice
                # pcd_tokens_for_decode: [B, T_in, tokens, C]
                pcd_tokens_t = pcd_tokens_for_decode[:, t, :, :]  # [B, tokens, C]

                # The inputs for this time step
                pcd_input = pcd_chunk_proj[:, t, :, :]  # [B, N, C]
                normal_input = normal_chunk_proj[:, t, :, :]
                rgb_input = rgb_chunk_proj[:, t, :, :]

                # Project and concat features
                out_ref_emb = self.point_embed(pcd_input)
                point_feat = self.point_normal_rgb_proj(torch.cat([
                    out_ref_emb, normal_input, rgb_input
                ], dim=-1))

                # query: [B, chunk_size, C], key/value: [B, num_pcd_tokens, C]
                decoded_out_t = self.decoder_cross_attn(
                    query=point_feat,
                    key=pcd_tokens_t,
                    value=pcd_tokens_t
                )
                output_t = self.shared_mlp_output(decoded_out_t) # [B, N, 3]
                outputs.append(output_t)
            decoded_chunk = torch.stack(outputs, dim=1).contiguous().view(-1, outputs[0].shape[1], outputs[0].shape[2])
            return decoded_chunk

        if not self.training and N > chunk_size:
            pcds_chunks = []
            for i in range(0, N, chunk_size):
                pcd_chunk_slice = output_pcds[:, i:i + chunk_size, :]
                normal_chunk_slice = output_normals[:, i:i + chunk_size, :]
                rgb_chunk_slice = output_rgbs[:, i:i + chunk_size, :]
                
                decoded_chunk = decode_chunk(pcd_chunk_slice, normal_chunk_slice, rgb_chunk_slice, pcd_tokens_final)
                pcds_chunks.append(decoded_chunk)
            pcds = torch.cat(pcds_chunks, dim=1) # [B*T, N, C]
        else:
            pcds = decode_chunk(output_pcds, output_normals, output_rgbs, pcd_tokens_final) # [B*T, N, C]

        output = pcds.view(B, T_in, N_points, 3)
        
        # Check if ground truth is available for loss computation
        if 'point_clouds' in sample:
            xyz_loss_metrics = self.loss_computer(output[..., :3], sample['point_clouds'])
            loss_metrics = edict()
            loss_metrics.loss = xyz_loss_metrics.loss
            loss_metrics.xyz_loss = xyz_loss_metrics.coord_mse_loss
            
            result = edict(
                input_data=sample,
                pcd_moved=output[..., :3],
                loss_metrics=loss_metrics
            )
        else:
            result = edict(
                input_data=sample,
                pcd_moved=output[..., :3]
            )
        return result
