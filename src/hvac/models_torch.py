import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """2-layer LSTM over a lookback window -> next-hour load."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
