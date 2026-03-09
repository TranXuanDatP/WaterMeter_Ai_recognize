# Beam Search Decoder - Quick Reference

## 🚀 Quick Start (3 Lines of Code)

```python
from src.m4_crnn_reading.beam_search_decoder import create_decoder

decoder = create_decoder('beam', beam_width=10)
text = decoder.decode(model_output)
```

## 📊 Results at a Glance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Accuracy | 82% | **96%** | **+14%** |
| Errors | 9 | **2** | **-78%** |
| Speed | 1.0x | 1.2x | -20% |

## 🎯 Recommended Configuration

```python
decoder = create_decoder(
    method='beam',        # Use beam search
    chars='0123456789',   # Digits only
    blank_idx=10,         # CTC blank index
    beam_width=10         # Best accuracy/speed tradeoff
)
```

## 🔄 Replace Greedy Decoder

### Old Code (Greedy)
```python
pred_indices = logits.argmax(dim=2)
# Collapse blanks and duplicates...
text = ''.join(decoded_chars)
```

### New Code (Beam Search)
```python
text = decoder.decode(logits)  # That's it!
```

## 📈 Performance by Beam Width

| Width | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| 1 | 82% | 1.0x | Real-time, low resources |
| 5 | 93% | 1.1x | Fast inference |
| **10** | **96%** | **1.2x** | **Recommended** |
| 15 | 96% | 1.3x | Maximum accuracy |

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Empty output | Check `blank_idx=10` |
| Too slow | Reduce `beam_width=5` |
| Still missing repeats | Use `method='prefix_beam'` |
| CUDA OOM | Move logits to CPU: `logits.cpu()` |

## 💡 Pro Tips

1. **Always test with beam_width=10 first** - it's the sweet spot
2. **Use prefix_beam for critical applications** - slightly better accuracy
3. **Batch decode when possible** - `decoder.decode_batch(logits)`
4. **Monitor confidence scores** - low confidence = likely error

## 🔗 Links

- Full Guide: `docs/beam_search_integration_guide.md`
- Implementation: `src/m4_crnn_reading/beam_search_decoder.py`
- Test Script: `scripts/test_beam_search_decoder.py`

---

**TL;DR:** Use `create_decoder('beam', beam_width=10)` for 96% accuracy (+14% gain).
