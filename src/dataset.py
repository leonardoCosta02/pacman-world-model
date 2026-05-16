"""
Dataset utilities for Pac-Man World Model.

Includes:
- Frame collection from Gymnasium Atari environment with self-supervised labeling
- Frame-level dataset for Baseline/VQ-VAE classifiers
- Sequence-level dataset for Temporal Transformer and Token Prior
- Frame-level sequence dataset for Frame Prior
"""
import os
import numpy as np
import gymnasium as gym
import ale_py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms


# ============================================================================
# Frame collection from environment
# ============================================================================
def collect_pacman_data(num_frames=10000, danger_window=15, seed=42):
    """
    Collect frames from Ms. Pac-Man with self-supervised DANGER labeling.
    
    A frame is labeled DANGER (1) if Pac-Man will lose a life within
    `danger_window` frames in the future, SAFE (0) otherwise.
    
    Returns:
        frames: np.ndarray of shape [num_frames, 210, 160]
        labels: np.ndarray of shape [num_frames]
    """
    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", obs_type="grayscale")
    env.action_space.seed(seed)  # IMPORTANTE: seeda anche action_space

    frames, lives_history = [], []
    observation, info = env.reset(seed=seed)
    current_lives = info.get('lives', 3)

    print(f"Collecting {num_frames} frames (seed={seed})...")
    for step in range(num_frames):
        if step % 5000 == 0 and step > 0:
            print(f"  Collected {step}/{num_frames} frames...")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(observation)
        lives_history.append(info.get('lives', current_lives))
        if terminated or truncated:
            observation, info = env.reset(seed=seed)
            current_lives = info.get('lives', 3)
    env.close()

    # Look-ahead labeling
    labels = np.zeros(len(frames))
    for i in range(len(frames) - danger_window):
        labels[i] = 1 if lives_history[i + danger_window] < lives_history[i] else 0

    return np.array(frames), np.array(labels)


def load_or_collect(cache_path, num_frames, danger_window=15, seed=42):
    """Load frames from disk if cached, otherwise collect and save."""
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path)
        return data['frames'], data['labels']
    
    frames, labels = collect_pacman_data(num_frames, danger_window, seed)
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, frames=frames, labels=labels)
    return frames, labels


# ============================================================================
# Transform pipeline shared by all datasets
# ============================================================================
def get_transform_pipeline(image_size=80):
    """Standard preprocessing: crop scoreboard, resize, normalize to [0,1]."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ============================================================================
# Dataset for Baseline / VQ-VAE (single frame, label SAFE/DANGER)
# ============================================================================
class PacmanDataset(Dataset):
    """Dataset of single frames with SAFE/DANGER labels."""
    
    def __init__(self, frames, labels, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        image = self.frames[idx][0:170, :]  # Crop HUD
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


# ============================================================================
# Dataset for Temporal Transformer / Token Prior (8-frame sequence + label)
# ============================================================================
class SequencePacmanDataset(Dataset):
    """Dataset of 8-frame sequences with SAFE/DANGER label of last frame."""
    
    def __init__(self, frames, labels, seq_len=8, transform=None):
        self.frames = frames
        self.labels = labels
        self.seq_len = seq_len
        self.transform = transform
        self.num_sequences = len(frames) - seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        sequence = []
        for t in range(self.seq_len):
            frame = self.frames[idx + t][0:170, :]
            if self.transform:
                frame = self.transform(frame)
            sequence.append(frame)
        sequence = torch.stack(sequence, dim=0)
        label = torch.tensor(self.labels[idx + self.seq_len - 1], dtype=torch.long)
        return sequence, label


# ============================================================================
# Dataset for Frame Prior (8-frame context + 1-frame target, no labels)
# ============================================================================
class FrameSequenceDataset(Dataset):
    """Dataset of (context_frames, target_frames) pairs for next-frame prediction."""
    
    def __init__(self, frames, seq_len=8, target_len=1, transform=None):
        self.frames = frames
        self.seq_len = seq_len
        self.target_len = target_len
        self.transform = transform
        self.num_sequences = len(frames) - seq_len - target_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        context = []
        for t in range(self.seq_len):
            frame = self.frames[idx + t][0:170, :]
            if self.transform:
                frame = self.transform(frame)
            context.append(frame)
        target = []
        for t in range(self.target_len):
            frame = self.frames[idx + self.seq_len + t][0:170, :]
            if self.transform:
                frame = self.transform(frame)
            target.append(frame)
        return torch.stack(context, dim=0), torch.stack(target, dim=0)


# ============================================================================
# Helper to build train/test splits + weighted samplers
# ============================================================================
def build_loaders_single(frames, labels, batch_size=32, split_seed=42, sampler_seed=42):
    """Build train/test DataLoaders for single-frame dataset (Baseline/VQ-VAE)."""
    transform = get_transform_pipeline()
    full_ds = PacmanDataset(frames, labels, transform=transform)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    
    split_gen = torch.Generator().manual_seed(split_seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=split_gen)
    
    # Balanced sampler for training
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[l] for l in train_labels]
    
    sampler_gen = torch.Generator().manual_seed(sampler_seed)
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights),
        replacement=True, generator=sampler_gen
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, class_counts


def build_loaders_sequence(frames, labels, seq_len=8, batch_size=16, split_seed=42, sampler_seed=42):
    """Build train/test DataLoaders for sequence dataset (Temporal Transformer/Token Prior)."""
    transform = get_transform_pipeline()
    full_ds = SequencePacmanDataset(frames, labels, seq_len=seq_len, transform=transform)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    
    split_gen = torch.Generator().manual_seed(split_seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=split_gen)
    
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[l] for l in train_labels]
    
    sampler_gen = torch.Generator().manual_seed(sampler_seed)
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights),
        replacement=True, generator=sampler_gen
    )
    
    train_loader_cls = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=2, pin_memory=True)
    train_loader_prior = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    return train_loader_cls, train_loader_prior, test_loader, class_counts


def build_loaders_frame_prior(frames, seq_len=8, target_len=1, batch_size=16, split_seed=42):
    """Build train/test DataLoaders for Frame Prior (no labels needed)."""
    transform = get_transform_pipeline()
    full_ds = FrameSequenceDataset(frames, seq_len=seq_len, target_len=target_len, transform=transform)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    
    split_gen = torch.Generator().manual_seed(split_seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=split_gen)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    return train_loader, test_loader