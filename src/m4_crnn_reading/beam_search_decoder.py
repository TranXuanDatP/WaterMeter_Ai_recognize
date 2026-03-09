"""
Beam Search Decoder for CTC Output

Improves over greedy decoding by exploring multiple paths and selecting
the most probable sequence. This helps with repeated digits which CTC
often collapses incorrectly.
"""

import torch
import numpy as np
from typing import List, Tuple


class BeamSearchCTCDecoder:
    """
    Beam Search Decoder for CTC output

    Args:
        chars: Character set (e.g., "0123456789")
        blank_idx: Index of blank token
        beam_width: Number of paths to keep (default: 10)
        prune: Minimum probability to keep a path (default: 0.001)

    Example:
        >>> decoder = BeamSearchCTCDecoder("0123456789", blank_idx=10, beam_width=10)
        >>> text = decoder.decode(logits)  # logits: (T, C) or (T, N, C)
    """

    def __init__(
        self,
        chars: str = "0123456789",
        blank_idx: int = 10,
        beam_width: int = 10,
        prune: float = 0.001
    ):
        self.chars = list(chars)
        self.blank_idx = blank_idx
        self.beam_width = beam_width
        self.prune = prune

    def decode(self, logits: torch.Tensor) -> str:
        """
        Decode CTC logits using beam search

        Args:
            logits: CTC output
                   - Shape (T, C) if single sample
                   - Shape (T, N, C) if batch, will use first sample

        Returns:
            Decoded string
        """
        # Handle batch dimension
        if logits.dim() == 3:
            logits = logits[:, 0, :]  # (T, N, C) -> (T, C)

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # (T, C)

        # Initialize beam with empty sequence
        beams = [
            {
                'text': '',
                'log_prob': 0.0,
                'last_blank': True
            }
        ]

        # Iterate through time steps
        for t in range(log_probs.size(0)):
            new_beams = []

            # Get top k tokens at this time step (pruning)
            top_k_log_probs, top_k_indices = torch.topk(
                log_probs[t],
                min(self.beam_width, log_probs.size(0))
            )

            for beam in beams:
                for log_prob, idx in zip(top_k_log_probs, top_k_indices):
                    idx = idx.item()
                    log_prob = log_prob.item()

                    # Prune low probability paths
                    if log_prob < np.log(self.prune):
                        continue

                    new_beam = beam.copy()
                    new_beam['log_prob'] += log_prob

                    # Add character if not blank and different from last char
                    if idx != self.blank_idx:
                        char = self.chars[idx]

                        # Only add if different from last character (CTC collapse)
                        if not new_beam['last_blank'] and \
                           len(new_beam['text']) > 0 and \
                           new_beam['text'][-1] == char:
                            # Skip duplicate (CTC contraction)
                            pass
                        else:
                            new_beam['text'] += char
                            new_beam['last_blank'] = False
                    else:
                        # Blank character
                        new_beam['last_blank'] = True

                    new_beams.append(new_beam)

            # Keep only top-k beams
            new_beams.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = new_beams[:self.beam_width]

        # Return best beam
        if len(beams) > 0:
            return beams[0]['text']
        else:
            return ""

    def decode_batch(self, logits: torch.Tensor) -> List[str]:
        """
        Decode batch of logits

        Args:
            logits: (T, N, C) - T time steps, N batch, C classes

        Returns:
            List of decoded strings
        """
        results = []
        for i in range(logits.size(1)):
            result = self.decode(logits[:, i, :])
            results.append(result)
        return results


class PrefixBeamSearchCTCDecoder:
    """
    Prefix Beam Search Decoder (more accurate than simple beam search)

    Maintains unique prefixes and merges paths with same prefix.
    More computationally expensive but better for repeated digits.
    """

    def __init__(
        self,
        chars: str = "0123456789",
        blank_idx: int = 10,
        beam_width: int = 10,
        prune: float = 0.001
    ):
        self.chars = list(chars)
        self.blank_idx = blank_idx
        self.beam_width = beam_width
        self.prune = prune

    def decode(self, logits: torch.Tensor) -> str:
        """
        Decode using prefix beam search

        Args:
            logits: (T, C) or (T, N, C)

        Returns:
            Decoded string
        """
        # Handle batch dimension
        if logits.dim() == 3:
            logits = logits[:, 0, :]

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Initialize with empty prefix
        beams = {
            ('', True): 0.0  # (prefix, last_blank): log_prob
        }

        # Iterate through time steps
        for t in range(log_probs.size(0)):
            new_beams = {}

            # Get top k tokens (pruning)
            top_k_log_probs, top_k_indices = torch.topk(
                log_probs[t],
                min(len(self.chars) + 1, log_probs.size(0))
            )

            for (prefix, last_blank), prefix_prob in beams.items():
                for log_prob, idx in zip(top_k_log_probs, top_k_indices):
                    idx = idx.item()
                    log_prob = log_prob.item()

                    # Prune
                    if log_prob < np.log(self.prune):
                        continue

                    new_prob = prefix_prob + log_prob
                    new_last_blank = False

                    if idx == self.blank_idx:
                        # Blank: don't extend prefix
                        new_prefix = prefix
                        new_last_blank = True
                    else:
                        char = self.chars[idx]

                        # CTC collapsing rule
                        if not last_blank and len(prefix) > 0 and prefix[-1] == char:
                            # Skip duplicate
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + char

                    key = (new_prefix, new_last_blank)

                    # Merge paths with same prefix (keep max prob)
                    if key not in new_beams or new_prob > new_beams[key]:
                        new_beams[key] = new_prob

            # Keep top-k beams
            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = dict(sorted_beams[:self.beam_width])

        # Return best prefix (ignore blank status)
        if len(beams) > 0:
            best_prefix = max(beams.items(), key=lambda x: x[1])[0][0]
            return best_prefix
        else:
            return ""


# Convenience function for easy usage
def create_decoder(
    method: str = "beam",
    chars: str = "0123456789",
    blank_idx: int = 10,
    beam_width: int = 10
):
    """
    Create CTC decoder

    Args:
        method: "greedy", "beam", or "prefix_beam"
        chars: Character set
        blank_idx: Blank token index
        beam_width: Beam width for beam search methods

    Returns:
        Decoder instance
    """
    if method == "greedy":
        # Simple greedy decoder (original)
        class GreedyDecoder:
            def __init__(self, chars, blank_idx):
                self.chars = list(chars)
                self.blank_idx = blank_idx

            def decode(self, logits):
                if logits.dim() == 3:
                    logits = logits[:, 0, :]

                pred_indices = logits.argmax(dim=1)
                decoded = []
                prev_idx = None

                for idx in pred_indices:
                    idx = idx.item()

                    if idx == self.blank_idx:
                        continue
                    if idx == prev_idx:
                        continue

                    decoded.append(self.chars[idx])
                    prev_idx = idx

                return ''.join(decoded)

        return GreedyDecoder(chars, blank_idx)

    elif method == "beam":
        return BeamSearchCTCDecoder(chars, blank_idx, beam_width)

    elif method == "prefix_beam":
        return PrefixBeamSearchCTCDecoder(chars, blank_idx, beam_width)

    else:
        raise ValueError(f"Unknown method: {method}")


# Test
if __name__ == "__main__":
    import sys

    print("Beam Search CTC Decoder Test")
    print("=" * 60)

    # Create sample logits (simulating "441")
    # High probability for '4' at t=0, '4' at t=1, '1' at t=2
    T, C = 10, 11  # 10 time steps, 11 classes (0-9 + blank)
    logits = torch.randn(T, C)

    # Set high probabilities for "441"
    logits[0, 4] = 5.0   # '4'
    logits[1, 4] = 5.0   # '4'
    logits[2, 4] = 5.0   # '4' (duplicate to test collapse)
    logits[3, 1] = 5.0   # '1'
    logits[4, 10] = 5.0  # blank

    # Test decoders
    print("\n[1] Testing Greedy Decoder:")
    greedy = create_decoder("greedy", beam_width=1)
    result_greedy = greedy.decode(logits)
    print(f"Result: '{result_greedy}'")

    print("\n[2] Testing Beam Search Decoder:")
    beam = create_decoder("beam", beam_width=10)
    result_beam = beam.decode(logits)
    print(f"Result: '{result_beam}'")

    print("\n[3] Testing Prefix Beam Search Decoder:")
    prefix_beam = create_decoder("prefix_beam", beam_width=10)
    result_prefix = prefix_beam.decode(logits)
    print(f"Result: '{result_prefix}'")

    print("\n" + "=" * 60)
    print("[OK] All decoders ready!")
    print("\nUsage:")
    print("  from src.m4_crnn_reading.beam_search_decoder import create_decoder")
    print("  decoder = create_decoder('beam', beam_width=10)")
    print("  text = decoder.decode(logits)")
