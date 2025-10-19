# 🗣️ English-to-Hindi Translation using Seq2Seq

This project implements a simple yet effective **Sequence-to-Sequence (Seq2Seq)** neural machine translation (NMT) model using **LSTM encoder-decoder architecture** to translate English sentences into Hindi. It is built in PyTorch and is designed for educational purposes and experimentation.

---

## ✨ Features

- 🔁 Sequence-to-sequence learning with LSTM-based encoder and decoder
- 🧠 Attention-less baseline (optional: extend to include attention)
- 📝 Tokenization with vocabulary generation and padding
- 🔤 Word-level training for both English and Hindi
- 📊 Training progress tracking with loss visualization
- 💬 Inference mode with greedy decoding

---

## 🏗️ Model Architecture

```text
English Input Sentence --> [Encoder LSTM] --> [Hidden States]
                                              ↓
                                    [Decoder LSTM + Linear]
                                              ↓
                                   Hindi Translated Sentence

```

- Encoder: Embedding + LSTM

- Decoder: Embedding + LSTM + Linear (softmax output)

- Optional: <sos> and <eos> tokens for better sentence control

## 🧑‍💻 Installing Dependencies
```bash
pip install torch nltk tqdm
```
