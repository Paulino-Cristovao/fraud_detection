import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, num_features: int):
        """
        A simple 1D CNN for tabular data.
        Input shape: (batch_size, num_features)
        """
        super(CNNClassifier, self).__init__()
        # We treat the features as a 1D signal with one channel.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # After two poolings, length reduces to num_features//4 (assuming num_features divisible by 4)
        self.fc1 = nn.Linear(32 * (num_features // 4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_features)
        x = x.unsqueeze(1)  # Convert to (batch, 1, num_features)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)
