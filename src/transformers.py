"""
Transformer-based models: Temporal Classifier, Token-level Prior, Frame-level Prior.
All use a frozen VQ-VAE as feature extractor.
"""
import math
import torch
import torch.nn as nn


# ============================================================================
# Positional Encoding (shared utility)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=9000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Model 3: Temporal Transformer Classifier (8-frame -> SAFE/DANGER)
# ============================================================================
class TemporalTransformerClassifier(nn.Module):
    """
    Classifies sequences of 8 frames using self-attention on VQ-VAE features.
    The VQ-VAE encoder + quantizer is frozen.
    """
    def __init__(self, vqvae_encoder, vq_module, seq_len=8,
                 d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.vqvae_encoder = vqvae_encoder
        self.vq_module = vq_module
        for param in self.vqvae_encoder.parameters():
            param.requires_grad = False
        for param in self.vq_module.parameters():
            param.requires_grad = False

        self.latent_flat_dim = 64 * 10 * 10
        self.input_projection = nn.Linear(self.latent_flat_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        with torch.no_grad():
            z = self.vqvae_encoder(x_flat)
            z_q, _ = self.vq_module(z)
        z_q = z_q.view(B * T, -1)
        tokens = self.input_projection(z_q).view(B, T, -1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_encoding(tokens)
        out = self.transformer(tokens)
        return self.classifier(out[:, 0, :])


# ============================================================================
# Model 4: Token-level Prior (autoregressive over 800 tokens)
# ============================================================================
class TransformerPrior(nn.Module):
    """
    Autoregressive Transformer that predicts the next token in a sequence of
    800 discrete tokens (8 frames x 100 tokens/frame).
    """
    def __init__(self, num_embeddings=128, d_model=256, nhead=4,
                 num_layers=4, dropout=0.15):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.generative_head = nn.Linear(d_model, num_embeddings)

    def forward(self, x_indices):
        tokens = self.token_embedding(x_indices)
        tokens = self.pos_encoding(tokens)
        seq_len = x_indices.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x_indices.device)
        out = self.transformer(tokens, mask=causal_mask)
        return self.generative_head(out)


# ============================================================================
# Model 5: Frame-level Prior (autoregressive over continuous latents)
# ============================================================================
class FrameLevelPrior(nn.Module):
    """
    Predicts the next latent z (continuous) given 8 previous latents.
    Loss is MSE on the latent space. 1 forward pass = 1 frame predicted.
    """
    def __init__(self, latent_channels=64, latent_h=10, latent_w=10,
                 d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_flat = latent_channels * latent_h * latent_w

        self.input_proj = nn.Linear(self.latent_flat, d_model)

        pe = torch.zeros(32, d_model)
        position = torch.arange(0, 32).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.latent_flat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_sequence):
        B, T, C, H, W = z_sequence.shape
        z_flat = z_sequence.view(B, T, -1)
        x = self.input_proj(z_flat)
        x = x + self.pe[:, :T, :]
        x = self.dropout(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.transformer(x, mask=causal_mask)
        last_hidden = out[:, -1, :]
        z_pred_flat = self.output_proj(last_hidden)
        return z_pred_flat.view(B, C, H, W)