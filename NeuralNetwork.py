import torch
import torch.nn as nn
import torch.nn.functional as F


class DroneDetectorCNN(nn.Module):
    """
    Treats the mel spectrogram like a grayscale image.
    Convolutional layers learn to recognize drone-specific
    frequency patterns: motor harmonics, blade pass frequencies.

    Input shape:  (batch, 1, n_mels, time_frames)
    Output shape: (batch, num_classes)
    """

    def __init__(self, num_classes=2):
        super(DroneDetectorCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        # Squeeze-and-Excite channel attention: focuses on rotor harmonic bands
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        attn = self.channel_attention(x).view(x.size(0), x.size(1), 1, 1)
        x = x * attn
        x = self.block4(x)
        return self.classifier(x)

    def predict_proba(self, x):
        """Returns probability of drone presence (0.0 to 1.0)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)[:, 1]
