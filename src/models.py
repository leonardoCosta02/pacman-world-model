"""
Core models: Baseline (PacmanWorldModel), VectorQuantizer with EMA, VQ-VAE multitask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model 1: Baseline (continuous latent, multitask)
# ============================================================================
class PacmanWorldModel(nn.Module):
    """
    Baseline multitask model with shared encoder and two heads:
    - Decoder for reconstruction (MSE)
    - Classifier for SAFE/DANGER prediction (BCE)
    """
    def __init__(self, latent_dim=64):
        super().__init__()

        # Shared encoder: [B, 1, 80, 80] -> [B, 6400]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.flattened_size = latent_dim * 10 * 10

        # Head A: Decoder
        self.decoder_reshape = nn.Linear(self.flattened_size, latent_dim * 10 * 10)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Head B: Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        dec_reshaped = self.decoder_reshape(encoded).view(-1, 64, 10, 10)
        reconstruction = self.decoder_convs(dec_reshaped)
        state_prediction = self.classifier(encoded)
        return reconstruction, state_prediction, encoded


# ============================================================================
# Vector Quantizer with EMA update (anti-collapse)
# ============================================================================
class VectorQuantizer(nn.Module):
    """
    VQ layer with Exponential Moving Average update of the codebook.
    Prevents codebook collapse by updating codebook entries as moving averages
    of the encoder vectors assigned to them.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embeddings', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z):
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # EMA update (only during training)
        if self.training:
            cluster_size = encodings.sum(dim=0)
            embed_sum = torch.matmul(encodings.t(), z_flat)

            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size_smooth = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            self.embeddings.data.copy_(self.embed_avg / cluster_size_smooth.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, loss

    @property
    def weight(self):
        return self.embeddings


# ============================================================================
# Model 2: VQ-VAE multitask (discrete latent + multitask heads)
# ============================================================================
class VQVAE(nn.Module):
    """
    VQ-VAE with multitask heads:
    - Encoder: [B, 1, 80, 80] -> [B, 64, 10, 10]
    - Vector Quantizer: 128 codes of 64 dims
    - Decoder: [B, 64, 10, 10] -> [B, 1, 80, 80]
    - Classifier: [B, 64, 10, 10] -> [B, 2] (SAFE/DANGER)
    """
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=1.0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=1)
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(quantized)
        logits = self.classifier(quantized)
        return x_recon, vq_loss, logits