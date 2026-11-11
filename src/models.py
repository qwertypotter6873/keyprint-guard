"""
Biometric Keystroke Verification Models for KeyPrint Guard:
1. ScaledManhattanDetector: Statistical anomaly baseline (Killourhy & Maxion, DSN 2009)
2. Vanilla1DCNN: Scratch 1D-CNN trained with standard Softmax Cross-Entropy (GAP only)
3. KeyPrintNet (Proposed): Positional Multi-Scale Dilated Residual SE-1DCNN +
   Dual Temporal Pooling (Attentive Statistics Pooling + 6-Mode 1D-DCT Spectral Cadence Pooling),
   trained with Smoothly Ramped Sub-Center CosFace + Supervised Contrastive Loss
   and evaluated with Adaptive Symmetric Cohort Normalization (AS-Norm).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledManhattanDetector:
    """
    Statistical anomaly detection baseline from CMU Keystroke Benchmark
    (Killourhy & Maxion, 2009), with empirical Bayes MAD shrinkage for K=5 enrollment.
    """

    def __init__(self, shrinkage: float = 0.35) -> None:
        self.shrinkage = shrinkage
        self.prior_mad: np.ndarray | None = None

    def fit_prior(self, train_flat: np.ndarray, train_labels: np.ndarray) -> "ScaledManhattanDetector":
        mads = []
        for uid in np.unique(train_labels):
            sub_x = train_flat[train_labels == uid]
            mu = np.mean(sub_x, axis=0, keepdims=True)
            mad = np.mean(np.abs(sub_x - mu), axis=0)
            mads.append(mad)
        self.prior_mad = np.mean(np.stack(mads, axis=0), axis=0) + 1e-4
        return self

    def enroll(self, enroll_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.prior_mad is not None, "Call fit_prior() on Train split first."
        mu = np.mean(enroll_flat, axis=0)
        sample_mad = np.mean(np.abs(enroll_flat - mu[None, :]), axis=0)
        reg_mad = (1.0 - self.shrinkage) * sample_mad + self.shrinkage * self.prior_mad
        return mu, np.maximum(reg_mad, 1e-4)

    @staticmethod
    def score(query_flat: np.ndarray, template: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        mu, mad = template
        dist = np.mean(np.abs(query_flat - mu[None, :]) / mad[None, :], axis=1)
        return np.exp(-0.5 * dist)


class Vanilla1DCNN(nn.Module):
    """
    Baseline Neural Model (Scratch):
    Position-agnostic 3-layer 1D Convolutional Network + Global Average Pooling -> 64-D embedding.
    """

    def __init__(self, in_channels: int = 9, embed_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = h.mean(dim=-1)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


class MultiScaleResSEBlock1D(nn.Module):
    """
    Multi-scale 1D residual block capturing both adjacent digraph mechanics (dilation=1)
    and multi-character trigram/quadgram phrasing (dilation=2) with Squeeze-and-Excitation.
    """

    def __init__(self, channels: int, dropout: float = 0.18) -> None:
        super().__init__()
        half = channels // 2
        self.local_conv = nn.Conv1d(channels, half, kernel_size=3, padding=1, dilation=1, bias=False)
        self.wide_conv = nn.Conv1d(channels, channels - half, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        se_hidden = max(16, channels // 4)
        self.se = nn.Sequential(
            nn.Conv1d(channels, se_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(se_hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(torch.cat([self.local_conv(x), self.wide_conv(x)], dim=1)))
        h = self.drop(self.bn2(self.pw_conv(h)))
        scale = self.se(h.mean(dim=-1, keepdim=True))
        return F.gelu(x + h * scale)


class KeyPrintNet(nn.Module):
    """
    Proposed Biometric Encoder (KeyPrintNet):
    - Dynamic-length ONNX-safe continuous relative positional harmonics [tau, cos(pi*tau), sin(pi*tau)]
    - 3 Multi-Scale Dilated Residual SE-Conv1D blocks
    - Dual Pooling:
      1) Attentive Statistics Pooling (channel-wise attentive mean + std)
      2) 6-Mode 1D-DCT Spectral Temporal Pooling (captures start-to-end phrase cadence for any length L)
    - Compact regularized projection to a 64-D L2-normalized unit hypersphere.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 80,
        embed_dim: int = 64,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels + 3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.block1 = MultiScaleResSEBlock1D(hidden_channels, dropout=dropout)
        self.block2 = MultiScaleResSEBlock1D(hidden_channels, dropout=dropout)
        self.block3 = MultiScaleResSEBlock1D(hidden_channels, dropout=dropout)

        self.attn = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(32, hidden_channels, kernel_size=1),
        )

        ks = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1) * math.pi
        self.register_buffer("freq_pi", ks)
        self.spec_reduce = nn.Sequential(
            nn.Conv1d(hidden_channels, 24, kernel_size=1, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )

        pooled_dim = hidden_channels * 2 + 24 * 6
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic-length ONNX-safe relative position tau in (0, 1) without constant-folding L
        ones = torch.ones_like(x[:, :1, :])
        steps = torch.cumsum(ones, dim=-1) - 0.5
        length = torch.sum(ones, dim=-1, keepdim=True)
        tau = steps / length
        x_pos = torch.cat([x, tau, torch.cos(math.pi * tau), torch.sin(math.pi * tau)], dim=1)

        h = self.stem(x_pos)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        # 1) Attentive Statistics Pooling (global rhythm mean & dispersion)
        alpha = torch.softmax(self.attn(h), dim=-1)
        mu = torch.sum(alpha * h, dim=-1)
        diff = h - mu.unsqueeze(-1)
        std = torch.sqrt(torch.clamp(torch.sum(alpha * (diff * diff), dim=-1), min=1e-6))

        # 2) 6-Mode 1D-DCT Spectral Temporal Pooling (start-to-end phrase cadence, arbitrary L)
        z_spec = self.spec_reduce(h)  # (B, 24, L)
        basis = torch.cos(self.freq_pi * tau)  # (B, 6, L)
        spec = (torch.bmm(z_spec, basis.transpose(1, 2)) / length).flatten(start_dim=1)  # (B, 144)

        emb = self.head(torch.cat([mu, std, spec], dim=1))
        return F.normalize(emb, p=2, dim=-1)


class SubCenterCosFaceHead(nn.Module):
    """
    Sub-Center Large Margin Cosine Loss (Sub-Center CosFace) with smooth margin ramp
    and mild label smoothing to capture intra-subject multi-modal rhythm modes
    without overfitting the 35 training subjects.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        sub_centers: int = 2,
        scale: float = 20.0,
        max_margin: float = 0.22,
        label_smoothing: float = 0.04,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.scale = scale
        self.max_margin = max_margin
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.empty(num_classes * sub_centers, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, ramp_ratio: float = 1.0) -> torch.Tensor:
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        cos_all = F.linear(emb, norm_w).view(-1, self.num_classes, self.sub_centers)
        cos, _ = torch.max(cos_all, dim=-1)

        margin = self.max_margin * float(ramp_ratio)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = self.scale * (cos - one_hot * margin)
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


class SupervisedContrastiveLoss(nn.Module):
    """
    Batch Supervised Contrastive Loss pulling cross-session keystrokes of the same
    identity together while pushing impostor identities apart on the unit hypersphere.
    """

    def __init__(self, temperature: float = 0.12) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = emb.size(0)
        if batch_size <= 2:
            return emb.sum() * 0.0

        sim = torch.matmul(emb, emb.transpose(0, 1)) / self.temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True).values.detach()

        eye = torch.eye(batch_size, dtype=torch.bool, device=emb.device)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)

        exp_sim = torch.exp(sim) * (~eye).float()
        log_prob = sim - torch.log(torch.sum(exp_sim, dim=1, keepdim=True) + 1e-8)

        pos_counts = pos_mask.float().sum(dim=1)
        valid = pos_counts > 0
        if not torch.any(valid):
            return emb.sum() * 0.0

        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid]
        return -mean_log_prob_pos.mean()
