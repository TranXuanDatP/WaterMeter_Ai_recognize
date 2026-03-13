# =============================================================================
# 🛠️ FIXED VALIDATION FUNCTION - Copy into Colab notebook
# =============================================================================

def decode_ctc(preds, blank_idx=10):
    """
    Decode CTC predictions: remove blanks and consecutive duplicates

    Args:
        preds: (T, N) or (N, T) - predicted indices
        blank_idx: blank token index

    Returns:
        List of decoded strings
    """
    if preds.dim() == 2:
        # (T, N) -> (N, T)
        if preds.size(0) < preds.size(1):
            preds = preds.permute(1, 0)

    decoded_texts = []
    for i in range(preds.size(0)):
        pred_seq = preds[i]

        # Remove consecutive duplicates
        prev_token = None
        collapsed = []
        for token in pred_seq:
            token_int = token.item()
            if token_int != prev_token:
                collapsed.append(token_int)
                prev_token = token_int

        # Remove blanks
        text = ''.join([str(t) if t < blank_idx else '' for t in collapsed])
        decoded_texts.append(text)

    return decoded_texts


def validate_fixed(model, dataloader, criterion, device):
    """
    ✅ FIXED: Use CTC decoding for accuracy calculation
    """
    model.eval()
    total_loss = 0
    total_ctc_loss = 0
    total_sar_loss = 0
    total_correct = 0
    total_samples = 0

    # Digit-wise accuracy
    digit_correct = 0
    digit_total = 0

    # Track 6→0 errors
    six_to_zero_errors = 0
    total_six = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['label_lengths'].to(device)
            sar_targets = batch['sar_targets'].to(device)

            # Forward pass
            ctc_logits, sar_logits = model(images, sar_targets)

            # Calculate loss
            losses = criterion(ctc_logits, sar_logits, labels, input_lengths, label_lengths)

            total_loss += losses['total'].item()
            total_ctc_loss += losses['ctc'].item()
            total_sar_loss += losses['sar'].item()

            # ✅ FIX: Use CTC for accuracy (CTC học tốt hơn)
            preds = ctc_logits.argmax(dim=2)  # (T, N)
            decoded_texts = decode_ctc(preds, blank_idx=10)

            for i, pred_text in enumerate(decoded_texts):
                true_text = batch['texts'][i]

                # Debug: print some predictions
                if total_samples < 5:
                    print(f"  Sample {i}: Pred='{pred_text}' | True='{true_text}' | Match={pred_text == true_text}")

                if pred_text == true_text:
                    total_correct += 1
                total_samples += 1

                # Digit-wise accuracy & 6→0 tracking
                # Align predicted and true texts
                max_len = max(len(true_text), len(pred_text))
                true_padded = true_text.ljust(max_len, ' ')
                pred_padded = pred_text.ljust(max_len, ' ')

                for t, p in zip(true_padded, pred_padded):
                    if t == '6':
                        total_six += 1

                    if t == p and t != ' ':
                        digit_correct += 1
                    digit_total += 1

                    # Track 6→0 errors
                    if t == '6' and p == '0':
                        six_to_zero_errors += 1

    return {
        'loss': total_loss / len(dataloader),
        'ctc_loss': total_ctc_loss / len(dataloader),
        'sar_loss': total_sar_loss / len(dataloader),
        'accuracy': total_correct / total_samples,
        'digit_accuracy': digit_correct / digit_total if digit_total > 0 else 0,
        'six_to_zero_error_rate': six_to_zero_errors / total_six if total_six > 0 else 0
    }


# =============================================================================
# 📌 HOW TO USE IN COLAB:
# =============================================================================
#
# 1. Copy toàn bộ code trên vào một cell mới trong notebook
# 2. Trong training loop, thay thế validate() bằng validate_fixed()
#
#   # Thay dòng này:
#   val_metrics = validate(model, val_loader, criterion, device)
#
#   # Bằng dòng này:
#   val_metrics = validate_fixed(model, val_loader, criterion, device)
#
# 3. Run lại từ epoch hiện tại
# =============================================================================
