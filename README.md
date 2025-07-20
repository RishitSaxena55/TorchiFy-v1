# MyTorch-v1 🚀

[![GitHub stars](https://img.shields.io/github/stars/RishitSaxena55/MyTorch-v1?style=social)](https://github.com/RishitSaxena55/MyTorch-v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **PyTorch-compatible** deep learning library featuring:
- Custom Transformer architectures for **Automatic Speech Recognition (ASR)** and **Language Modeling**
- From-scratch implementations of core neural network components
- End-to-end training pipelines with advanced techniques like beam search and gradient accumulation

## 📦 Key Features

### 🎙️ Automatic Speech Recognition (ASR)
- **Encoder-Decoder Transformer** architecture with:
  - Multi-head self/cross-attention
  - Positional encoding
  - CTC head for alignment
- Support for:
  - Greedy/beam search decoding
  - SpecAugment data augmentation
  - Character/subword tokenization

### 📖 Language Modeling
- **Decoder-only Transformer** (GPT-style) with:
  - Causal masking
  - Autoregressive generation
- Tokenization strategies:
  - Character-level
  - Byte Pair Encoding (BPE)
- Sampling methods:
  - Greedy, top-k, nucleus (top-p)

### ⚙️ Core Components
| Module | Implementations |
|--------|----------------|
| **Attention** | Scaled dot-product, multi-head |
| **Layers** | Linear, LayerNorm, FeedForward |
| **Optimization** | AdamW, LR scheduling |
| **Regularization** | Dropout, label smoothing |

## 📊 Architecture Diagrams

### Encoder-Decoder Transformer (ASR)
```mermaid
graph TD
    A[80-dim FBANK Features] --> B[SpeechEmbedding]
    B --> C[+PositionalEncoding]
    C --> D[Encoder Layers]
    D --> E[Decoder Layers]
    E --> F[Linear + Softmax]
    F --> G[Character/Subword Predictions]

graph LR
    A[Token Embeddings] --> B[+PositionalEncoding]
    B --> C[Decoder Layers]
    C --> D[Linear + Softmax]
    D --> E[Next-Token Predictions]

git clone https://github.com/RishitSaxena55/MyTorch-v1.git
cd MyTorch-v1
pip install -r requirements.txt

MyTorch-v1/
├── hw4lib/                 # Main library
│   ├── data/               # Dataset handling
│   │   ├── asr_dataset.py  # Speech dataset loader
│   │   ├── lm_dataset.py   # Text dataset loader
│   │   └── tokenizer.py    # Tokenization strategies
│   ├── model/              # Transformer components
│   │   ├── sublayers.py    # Attention/FFN layers
│   │   ├── transformers.py # Full architectures
│   │   └── positional_encoding.py
│   ├── decoding/           # Generation algorithms
│   └── trainers/           # Training pipelines
├── mytorch/                # Custom NN components
│   └── nn/
│       ├── linear.py       # Fully-connected layer
│       ├── activation.py   # Softmax/GELU
│       └── attention/      # Attention mechanisms
└── tests/                  # Unit tests
