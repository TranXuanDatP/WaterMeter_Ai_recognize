"""
Tests for CRNN Model (M4)

Test suite following red-green-refactor cycle:
1. RED: Write failing tests
2. GREEN: Implement minimal code to pass tests
3. REFACTOR: Improve code while keeping tests green
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCRNNModel:
    """Test CRNN model architecture"""

    def test_model_import(self):
        """Test that CRNN model can be imported"""
        try:
            from src.m4_crnn_reading.model import CRNN
            assert CRNN is not None
        except ImportError as e:
            pytest.fail(f"Failed to import CRNN: {e}")

    def test_model_initialization(self):
        """Test that CRNN model initializes correctly"""
        from src.m4_crnn_reading.model import CRNN

        # Model parameters
        num_chars = 11  # 0-9 + blank
        hidden_size = 256
        num_layers = 2

        # Create model
        model = CRNN(num_chars=num_chars, hidden_size=hidden_size, num_layers=num_layers)

        # Check model is not None
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_forward_pass(self):
        """Test that CRNN forward pass works"""
        from src.m4_crnn_reading.model import CRNN

        # Create model
        model = CRNN(num_chars=11, hidden_size=256, num_layers=2)
        model.eval()

        # Create dummy input: (batch, channels, height, width)
        # Typical input: (B, 1, 64, 160)
        dummy_input = torch.randn(2, 1, 64, 160)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape: (seq_len, batch, num_classes)
        # Expected: (T, B, 11) where T depends on width
        assert output is not None
        assert len(output.shape) == 3
        assert output.shape[1] == 2  # batch size
        assert output.shape[2] == 11  # num_classes

    def test_ctc_decoder_initialization(self):
        """Test that CTC decoder initializes correctly"""
        from src.m4_crnn_reading.model import CTCDecoder

        decoder = CTCDecoder(num_chars=11)
        assert decoder is not None
        assert isinstance(decoder, nn.Module)

    def test_ctc_decoder_decode(self):
        """Test that CTC decoder converts logits to text"""
        from src.m4_crnn_reading.model import CTCDecoder

        decoder = CTCDecoder(num_chars=11)

        # Create dummy logits: (seq_len, batch, num_classes)
        # Simulate prediction for "12345"
        # Initialize with negative infinity (blank tokens)
        dummy_logits = torch.full((20, 1, 11), float('-inf'))
        dummy_logits[2, 0, 1] = 10.0  # '1'
        dummy_logits[5, 0, 2] = 10.0  # '2'
        dummy_logits[8, 0, 3] = 10.0  # '3'
        dummy_logits[11, 0, 4] = 10.0  # '4'
        dummy_logits[14, 0, 5] = 10.0  # '5'
        # Set blank token (10) for all other time steps
        dummy_logits[:, 0, 10] = 0.0

        # Decode
        with torch.no_grad():
            result = decoder.decode(dummy_logits)

        # Check result
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 5  # Should decode to "12345"


class TestCRNNTraining:
    """Test CRNN training components"""

    def test_ctc_loss_function(self):
        """Test that CTC loss function works"""
        from src.m4_crnn_reading.model import CTCLoss

        # Create loss
        ctc_loss = CTCLoss(blank_idx=10)

        # Create dummy data
        logits = torch.randn(20, 2, 11).log_softmax(2)  # (T, N, C)
        targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=torch.long)  # Concatenated targets
        input_lengths = torch.tensor([20, 20], dtype=torch.long)
        target_lengths = torch.tensor([5, 5], dtype=torch.long)

        # Calculate loss
        loss = ctc_loss(logits, targets, input_lengths, target_lengths)

        # Check loss is valid
        assert loss is not None
        assert loss.item() > 0  # Loss should be positive


class TestCRNNInference:
    """Test CRNN inference pipeline"""

    def test_inference_class_initialization(self):
        """Test that M4Inference class initializes"""
        from src.m4_crnn_reading.inference import M4Inference

        # Create dummy model path
        dummy_model_path = "/tmp/dummy_model.pth"

        # Should raise error if model doesn't exist
        with pytest.raises(FileNotFoundError):
            M4Inference(model_path=dummy_model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
