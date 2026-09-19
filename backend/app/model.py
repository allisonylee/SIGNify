"""
The classifier: one shared encoder + one small head per sign language.

SHIPS -- app/ loads trained weights from this definition at inference.

Design (plan section 7): train the encoder on GISLR ASL (94k sequences), then
FREEZE it and train only a fresh head per additional language. A 30-sign
self-recorded language is viable only because the encoder already learned hand
shapes and motion primitives from ASL. At inference we hold one encoder plus N
tiny heads, so switching language is a head swap -- no reload, no latency hit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .landmarks import N_CHANNELS, N_LANDMARKS

IN_FEATURES = N_LANDMARKS * N_CHANNELS          # 159


class Encoder(nn.Module):
    """
    Conv1d stem over time, then a BiGRU.

    The conv stem captures local motion (how a hand moves over a few frames);
    the recurrent layer captures the overall trajectory. ~1M params, which is
    why this trains on a laptop -- the hard vision work was already done by
    MediaPipe before the model sees anything.
    """

    def __init__(self, d_model=192, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(IN_FEATURES, d_model, 5, padding=2, groups=1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(d_model, hidden, layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, x):                        # x: (B, T, 53, 3)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b, t, IN_FEATURES).transpose(1, 2)    # (B, 159, T)
        x = self.stem(x).transpose(1, 2)                     # (B, T, d_model)
        y, _ = self.rnn(x)                                   # (B, T, 2*hidden)
        # Mean + max pooling over time: mean carries the overall shape of the
        # sign, max catches the single most distinctive instant.
        return torch.cat([y.mean(1), y.max(1).values], dim=-1)


class SignClassifier(nn.Module):
    """Shared encoder + a named head per sign language."""

    def __init__(self, heads: dict[str, int], **enc_kw):
        super().__init__()
        self.encoder = Encoder(**enc_kw)
        self.heads = nn.ModuleDict({
            lang: nn.Sequential(
                nn.LayerNorm(self.encoder.out_dim * 2),
                nn.Dropout(0.3),
                nn.Linear(self.encoder.out_dim * 2, n_classes),
            )
            for lang, n_classes in heads.items()
        })

    def forward(self, x, lang: str):
        return self.heads[lang](self.encoder(x))

    def add_head(self, lang: str, n_classes: int):
        """Attach a head for a new sign language (plan Stage 5)."""
        dev = next(self.parameters()).device
        self.heads[lang] = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(self.encoder.out_dim * 2, n_classes),
        ).to(dev)

    def freeze_encoder(self, frozen=True):
        """Stage 5: freeze, then train only the new head. 2-3k samples would
        overfit a 2M-param model instantly if trained end to end."""
        for p in self.encoder.parameters():
            p.requires_grad = not frozen

    def n_params(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        hd = {k: sum(p.numel() for p in v.parameters()) for k, v in self.heads.items()}
        return {"encoder": enc, "heads": hd, "total": enc + sum(hd.values())}
