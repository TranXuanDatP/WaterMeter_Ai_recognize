"""
CRNN Model for Meter Reading (M4)

Architecture: Custom CNN + BiLSTM + CTC
- Custom CNN backbone for feature extraction (matches trained checkpoint)
- Bidirectional LSTM for sequential modeling
- CTC Loss for end-to-end training

This architecture matches the M4_OCR.pth checkpoint trained in Colab.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN Model for end-to-end meter reading

    Architecture:
        Input (B, 1, H, W) - H=64, W=224
          ↓
        Custom CNN Backbone
          → Output: (B, 512, H/16, W/8) = (B, 512, 4, 28)
          ↓
        Permute + Reshape
          → Output: (W', B, C*H) = (28, B, 2048)
          ↓
        BiLSTM (2 layers, 256 hidden, batch_first=True)
          → Output: (W', B, 512)
          ↓
        Linear projection
          → Output: (W', B, num_chars)
          ↓
        Permute for CTC
          → Output: (T, B, C) where T=W'
    """

    def __init__(self, num_chars=11, num_channels=1, img_height=64, hidden_size=256):
        """
        Initialize CRNN model

        Args:
            num_chars: Number of output classes (default: 11 for 0-9 + blank)
            num_channels: Input channels (default: 1 for grayscale)
            img_height: Input image height (default: 64)
            hidden_size: LSTM hidden size (default: 256)
        """
        super(CRNN, self).__init__()

        self.num_chars = num_chars
        self.hidden_size = hidden_size

        # Custom CNN Feature Extractor (matches Colab training)
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 64
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/8, W/8

            # Block 4: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16, W/8

            # Block 5: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # No pooling here
        )

        # Calculate RNN input size
        # After CNN: height = img_height // 16, width = img_width // 8
        # Features: 512 channels * (img_height // 16)
        h_out = img_height // 16
        self.rnn_input_size = 512 * h_out  # 512 * 4 = 2048 for img_height=64

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # Output projection (bidirectional → 2*hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W)
               - B: batch size
               - C: channels (1 for grayscale)
               - H: height (default 64)
               - W: width (default 224)

        Returns:
            logits: Output tensor (T, N, C)
                    - T: time steps (sequence length)
                    - N: batch size
                    - C: number of classes
        """
        # CNN feature extraction
        conv_out = self.cnn(x)  # (B, 512, H/16, W/8)

        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = conv_out.size()
        assert h == 4, f"Expected height=4, got {h}"

        # Permute and reshape: flatten features along height
        features = conv_out.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.contiguous().view(b, w, c * h)  # (B, W, 512*4=2048)

        # RNN processing
        rnn_out, _ = self.rnn(features)  # (B, W, 512)

        # Output projection
        logits = self.fc(rnn_out)  # (B, W, num_chars)

        # For compatibility with CTC decoder, transpose: (B, T, C) -> (T, B, C)
        logits = logits.permute(1, 0, 2)  # (T, B, C)

        return logits
        conv_out = self.cnn(x)  # (B, 512, H/32, W/32)

        # Pool to reduce height to 1
        pooled = self.adaptive_pool(conv_out)  # (B, 512, 1, W')

        # Reshape to sequence: (B, C, 1, W') → (W', B, C)
        b, c, h, w = pooled.size()
        assert h == 1, f"Height should be 1, got {h}"


class CTCDecoder(nn.Module):
    """
    CTC Decoder for converting logits to text

    Uses greedy decoding to collapse CTC output
    """

    def __init__(self, num_chars=11, blank_idx=10):
        """
        Initialize CTC decoder

        Args:
            num_chars: Number of classes
            blank_idx: Index of blank token (default: 10)
        """
        super(CTCDecoder, self).__init__()
        self.num_chars = num_chars
        self.blank_idx = blank_idx

        # Character mapping
        self.chars = [str(i) for i in range(10)]  # '0'-'9'

    def decode(self, logits):
        """
        Decode CTC logits to text using greedy decoding

        Args:
            logits: CTC output (T, N, C)
                    - T: time steps
                    - N: batch size
                    - C: num classes

        Returns:
            text: Decoded string (for single batch)
        """
        # Get argmax along class dimension
        pred_indices = logits.argmax(dim=2)  # (T, N)

        # Collapse CTC output (remove blanks and consecutive duplicates)
        decoded = []
        prev_idx = None

        for t in range(pred_indices.size(0)):
            idx = pred_indices[t, 0].item()  # First batch item

            # Skip blank tokens
            if idx == self.blank_idx:
                continue

            # Skip consecutive duplicates
            if idx == prev_idx:
                continue

            # Add character
            decoded.append(self.chars[idx])
            prev_idx = idx

        return ''.join(decoded)


class CTCLoss(nn.Module):
    """
    CTC Loss for sequence-to-sequence learning
    """

    def __init__(self, blank_idx=10):
        """
        Initialize CTC loss

        Args:
            blank_idx: Index of blank token (default: 10)
        """
        super(CTCLoss, self).__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean')

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Calculate CTC loss

        Args:
            logits: Model output (T, N, C)
                    - T: time steps
                    - N: batch size
                    - C: num classes
            targets: Concatenated target labels (1D tensor)
            input_lengths: Length of each sequence in batch (1D tensor)
            target_lengths: Length of each target in batch (1D tensor)

        Returns:
            loss: CTC loss value
        """
        # Log softmax for CTC
        log_probs = logits.log_softmax(2)

        # Calculate loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        return loss


if __name__ == "__main__":
    # Test model
    print("Testing CRNN Model...")

    # Create model
    model = CRNN(num_chars=11, hidden_size=256, num_layers=2)
    print(f"✓ Model created")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 160)
    output = model(dummy_input)
    print(f"✓ Forward pass: {dummy_input.shape} → {output.shape}")

    # Test decoder
    decoder = CTCDecoder(num_chars=11)
    text = decoder.decode(output)
    print(f"✓ Decoded: '{text}'")

    # Test loss
    ctc_loss = CTCLoss(blank_idx=10)
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)
    input_lengths = torch.tensor([output.size(0), output.size(0)], dtype=torch.long)
    target_lengths = torch.tensor([5, 5], dtype=torch.long)
    loss = ctc_loss(output, targets, input_lengths, target_lengths)
    print(f"✓ CTC Loss: {loss.item():.4f}")

    print("\n✅ All tests passed!")
