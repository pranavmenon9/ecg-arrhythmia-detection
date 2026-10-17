import torch.nn as nn

CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
SIGNAL_LENGTH = 187


class CNN(nn.Module):
    """1D CNN for ECG heartbeat classification. Same architecture as notebooks/04_cnn.ipynb."""

    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),    # (B, 1, 187)   -> (B, 32, 187)
            nn.ReLU(),
            nn.MaxPool1d(2),                               # (B, 32, 187)  -> (B, 32, 93)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),   # (B, 32, 93)   -> (B, 64, 93)
            nn.ReLU(),
            nn.MaxPool1d(2),                               # (B, 64, 93)   -> (B, 64, 46)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # (B, 64, 46)   -> (B, 128, 46)
            nn.ReLU(),
            nn.MaxPool1d(2),                               # (B, 128, 46)  -> (B, 128, 23)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # (B, 128, 23)  -> (B, 2944)
            nn.Dropout(dropout),
            nn.Linear(128 * 23, 64),                       # (B, 2944)     -> (B, 64)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),                      # (B, 64)       -> (B, 5) raw logits
        )

    def forward(self, x):                                  # x: (B, 187)
        x = x.unsqueeze(1)                                 # (B, 187) -> (B, 1, 187)
        return self.classifier(self.features(x))
