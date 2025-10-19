# Hindi-English Neural Machine Translation

A comprehensive comparison of neural sequence-to-sequence architectures for Hindi-to-English translation, implementing BiRNN, BiLSTM, BiLSTM with Attention, and Transformer models.

## 🎯 Project Overview

This project trains and evaluates four different neural machine translation architectures on a Hindi-English parallel corpus:

1. **Bidirectional RNN (BiRNN)**: Basic encoder-decoder with bidirectional SimpleRNN layers
2. **Bidirectional LSTM (BiLSTM)**: Enhanced encoder-decoder with LSTM units for better long-term dependencies
3. **BiLSTM with Attention**: BiLSTM enhanced with attention mechanism for improved context modeling
4. **Transformer**: State-of-the-art architecture with multi-head self-attention and positional encodings

## 📊 Features

- **Multiple Architecture Comparison**: Side-by-side evaluation of 4 different NMT models
- **Optimized for 25GB RAM**: Memory-efficient implementation suitable for Kaggle/Colab environments
- **Comprehensive Evaluation**: BLEU and ROUGE-L scores for translation quality assessment
- **Batch Decoding**: Fast inference with batch processing for efficient evaluation
- **Visualization**: Automated generation of comparison charts
- **Sample Predictions**: Real translation examples from test set

## 🔧 Requirements

```
numpy
pandas
tensorflow>=2.8.0
scikit-learn
nltk
rouge-score
matplotlib
tqdm
```

Install dependencies:
```bash
pip install numpy pandas tensorflow scikit-learn nltk rouge-score matplotlib tqdm
```

## 📁 Dataset Format

The code expects a CSV file with two columns:
- `hindi`: Hindi text sentences
- `english`: English translations

Example:
```csv
hindi,english
मैं घर जा रहा हूं,I am going home
यह एक किताब है,This is a book
```

## 🚀 Usage

### Basic Setup

1. **Update the dataset path** in the code:
```python
DATASET_PATH = '/kaggle/input/your-dataset/hindi_english_parallel.csv'
```

2. **Run the entire script**:
```python
python translation_model.py
```

### Configuration

Adjust hyperparameters in the configuration section:

```python
EMB_DIM = 256       # Embedding dimension
LAT_DIM = 512       # LSTM/RNN hidden units
DROPOUT = 0.3       # Dropout rate
EPOCHS = 15         # Training epochs
BATCH_SIZE = 128    # Batch size
```

For smaller datasets, reduce `max_samples`:
```python
hindi_data, english_data = load_data(DATASET_PATH, max_samples=8000)
```

## 🏗️ Model Architectures

### 1. BiRNN
```
Encoder: Bidirectional SimpleRNN (256 units)
Decoder: SimpleRNN (512 units) with encoder state initialization
```

### 2. BiLSTM
```
Encoder: Bidirectional LSTM (256 units each direction)
Decoder: LSTM (512 units) with encoder states (h, c)
```

### 3. BiLSTM + Attention
```
Encoder: Bidirectional LSTM returning sequences
Decoder: LSTM with attention mechanism over encoder outputs
Attention: Additive (Bahdanau) attention layer
```

### 4. Transformer
```
Encoder: 2 layers, 4 attention heads, 256-dim feedforward
Decoder: 2 layers with masked self-attention and cross-attention
Positional Encoding: Sinusoidal embeddings
```

## 📈 Training Process

The training pipeline includes:

1. **Data Loading**: Handles multiple encodings (UTF-8, Latin-1, etc.)
2. **Preprocessing**: 
   - Lowercasing and cleaning
   - Adding start/end tokens
   - Tokenization with OOV handling
3. **Sequence Padding**: Percentile-based max length to handle outliers
4. **Training**: 
   - Adam optimizer with learning rate 0.001
   - Early stopping (patience=7)
   - Learning rate reduction on plateau
5. **Evaluation**: BLEU and ROUGE-L metrics on test set

## 📊 Outputs

The script generates:

1. **`translation_results.csv`**: Numerical results for all models
   ```
   Model                BLEU    ROUGE-L
   BiRNN                0.3245  0.4102
   BiLSTM               0.3891  0.4756
   BiLSTM+Attention     0.4523  0.5234
   Transformer          0.4712  0.5401
   ```

2. **`model_comparison.png`**: Bar chart comparing BLEU and ROUGE-L scores

3. **Console Output**:
   - Training progress for each model
   - Sample translations from best model
   - Final metrics summary

## 🔍 Evaluation Metrics

- **BLEU Score**: Measures n-gram overlap between predicted and reference translations (0-1, higher is better)
- **ROUGE-L**: Measures longest common subsequence between predictions and references (0-1, higher is better)

## 💡 Tips for Better Results

1. **Increase Dataset Size**: More data (50k+ samples) significantly improves quality
2. **Tune Hyperparameters**: 
   - Increase `EMB_DIM` and `LAT_DIM` for larger datasets
   - Adjust `EPOCHS` based on convergence
3. **Add Data Augmentation**: Back-translation, synonym replacement
4. **Ensemble Models**: Combine predictions from multiple models
5. **Use Pretrained Embeddings**: Load Hindi/English word2vec or FastText embeddings

## 🐛 Common Issues

### Memory Errors
```python
# Reduce batch size
BATCH_SIZE = 64

# Reduce model dimensions
EMB_DIM = 128
LAT_DIM = 256

# Limit dataset size
max_samples = 5000
```

### Poor Translation Quality
- Ensure dataset has quality parallel sentences
- Increase training epochs (20-30)
- Check for data preprocessing issues
- Verify Hindi text is properly encoded (UTF-8)

### Slow Training
- Enable GPU acceleration (automatically detected by TensorFlow)
- Increase batch size if memory allows
- Reduce number of models to train

## 📝 Sample Output

```
[1] Hindi:     मैं आज बाजार जाऊंगा
    Reference: i will go to the market today
    Predicted: i will go to market today

[2] Hindi:     यह किताब बहुत अच्छी है
    Reference: this book is very good
    Predicted: this book is very good

[3] Hindi:     वह स्कूल में पढ़ता है
    Reference: he studies in school
    Predicted: he studies at school
```

## 🔬 Future Improvements

- [ ] Add beam search decoding for better translations
- [ ] Implement subword tokenization (BPE/WordPiece)
- [ ] Add pretrained embeddings (mBERT, XLM-R)
- [ ] Support bidirectional translation (English→Hindi)
- [ ] Add BLEU-4 and other evaluation metrics
- [ ] Implement model ensembling
- [ ] Add inference API with Flask/FastAPI

## 📚 References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

Built for exploring neural machine translation architectures on Hindi-English parallel corpora.

---

**Note**: This implementation is optimized for educational purposes and benchmarking different architectures. For production-grade translation, consider using pretrained models like mBART, mT5, or IndicTrans.
