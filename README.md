# ⚡️TorchiFy-v1

![License](https://img.shields.io/github/license/RishitSaxena55/TorchiFy-v1)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Language](https://img.shields.io/badge/python-3.10+-blue)
![Visitors](https://komarev.com/ghpvc/?username=RishitSaxena55&color=blue)
![Stars](https://img.shields.io/github/stars/RishitSaxena55/TorchiFy-v1?style=social)

> **"TorchiFy: Engineering Intelligence from First Principles — Build LLMs, Whisper-style ASR, and Transformer Architectures from Scratch."**

---

## 🚀 Introduction

**TorchiFy-v1** is a visionary, fully modular deep learning research framework written in pure PyTorch. It is designed to **build decoder-only, encoder-decoder, and speech-transformer architectures** entirely from scratch—without relying on abstracted libraries. Whether you're building GPT-style chatbots, Whisper-like ASR systems, or next-gen AI interfaces, **TorchiFy** puts the architecture in your hands.

GitHub: [RishitSaxena55/TorchiFy-v1](https://github.com/RishitSaxena55/TorchiFy-v1)

---

## 🌌 Philosophy

At the core of **TorchiFy** lies a mission:  
> **Empower the curious to understand and build the next frontier of deep learning models—line-by-line, token-by-token.**

TorchiFy is **educational**, **research-ready**, and **production-focused**, all at once.

---

## 🧠 What Makes TorchiFy Unique

| Feature                        | TorchiFy              | HuggingFace     | Fairseq        |
| ----------------------------- | --------------------- | --------------- | -------------- |
| Transformer from Scratch      | ✅ Yes (modular)       | ❌ No            | ❌ No           |
| PyTorch Only (No Frameworks)  | ✅ Zero abstraction    | ❌ Heavy deps    | ❌              |
| ASR + Text Integration        | ✅ Native (Whisper)    | ⚠️ Text-focused  | ✅              |
| Fully Unit Tested             | ✅ High Coverage       | ⚠️ Incomplete    | ✅              |
| Educational Purpose           | ✅ Clear & Commented   | ❌ Obscure APIs  | ❌              |
| Inference Optimization Tools  | ✅ Beam + Caching      | ⚠️ Partial       | ⚠️ Partial      |
| Training Engine               | ✅ Built-in            | ⚠️ Requires Trainer | ⚠️              |

---

## 🧭 Project Overview

TorchiFy breaks the complexity of transformers into understandable, testable components:

- ✅ **Custom Multi-Head Attention**
- ✅ **Causal & Bidirectional Masks**
- ✅ **Sinusoidal, Learned & Rotary Embeddings**
- ✅ **Flexible Transformer Blocks (Decoder, Encoder, Hybrid)**
- ✅ **Audio + Text Frontends**
- ✅ **Modular Beam Search & Greedy Decoding**
- ✅ **End-to-End Whisper-like ASR**
- ✅ **LLM Pretraining Routines**
- ✅ **FP16 Support & Caching for Deployment**

---

## 💥 Use TorchiFy For

- 💬 **LLM Prototyping:** GPT-style transformer decoder with full training pipeline.
- 🧠 **ASR Systems:** Whisper-style encoder-decoder for speech-to-text.
- 🎓 **Teaching & Research:** Transformer mechanics, custom experiments.
- ⚗️ **Inference Research:** Modify attention sparsity, decoding speed, etc.
- 📦 **Production Deployment:** Modular hooks for optimization, quantization.

---

## 🧬 Core Modules

| Folder             | Purpose                                                                 |
| ------------------| ------------------------------------------------------------------------ |
| `torchiFy/core/`   | Core transformer blocks (attention, embeddings, feedforward, etc.)     |
| `torchiFy/models/` | Model definitions: GPT, Whisper, Encoder-Decoder Transformers          |
| `torchiFy/nn/`     | Custom attention, masking, rotary/positional embeddings                |
| `torchiFy/audio/`  | Speech preprocessing: MFCC, Log-Mel, audio tokenizers                  |
| `torchiFy/train/`  | Training loop, loss functions, optimizers, schedulers                  |
| `torchiFy/decode/` | Greedy decoding, beam search, LM scoring                               |
| `torchiFy/utils/`  | Logging, timing, caching, weight loading/saving                        |
| `torchiFy/tests/`  | Full unit and integration test suite                                   |
| `configs/`         | YAML config files for models and training                              |
| `scripts/`         | Pretraining, inference, dataset setup scripts                          |

---

## 🧱 File Structure Explained

```
TorchiFy-v1/
├── torchiFy/
│   ├── core/
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── embeddings.py
│   │   ├── transformer_block.py
│   ├── nn/
│   │   ├── utils.py
│   │   ├── mask.py
│   ├── models/
│   │   ├── gpt.py
│   │   ├── whisper.py
│   │   ├── encoder_decoder.py
│   ├── decode/
│   │   ├── beam.py
│   │   ├── greedy.py
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── tokenizer.py
│   ├── train/
│   │   ├── engine.py
│   │   ├── losses.py
│   │   ├── optim.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── timer.py
│   │   ├── weights.py
│   ├── tests/
│   │   ├── test_attention.py
│   │   ├── test_gpt.py
├── configs/
│   ├── gpt_small.yaml
│   ├── whisper_base.yaml
├── scripts/
│   ├── train_gpt.py
│   ├── infer_speech.py
│   ├── convert_audio.py
├── README.md
└── requirements.txt
```

---

## 🔧 Installation

```bash
git clone https://github.com/RishitSaxena55/TorchiFy-v1.git
cd TorchiFy-v1
pip install -r requirements.txt
```

---

## 🚦 Getting Started

### 🧪 Run a Unit Test
```bash
pytest torchiFy/tests/
```

### 🔥 Train a GPT-like Model
```bash
python scripts/train_gpt.py --config configs/gpt_small.yaml
```

### 🎙️ Infer from Speech Input
```bash
python scripts/infer_speech.py --audio sample.wav --config configs/whisper_base.yaml
```

---

## 🧠 Key Innovations

- **No Hidden Blackboxes**
- **Custom Masking Engines**
- **Rotary Positional Embeddings**
- **Audio-Aware Transformer**
- **Beam Search from Scratch**
- **Doc-rich Modules**

---

## 🧪 Testing & Validation

```bash
pytest torchiFy/tests/
```

---

## 🛠️ Inference Strategies

- ✅ Greedy decoding
- ✅ Beam decoding with penalties
- ✅ CTC + LM Rescoring
- ✅ Caching of key-value pairs
- ✅ TorchScript-compatible exports

---

## 🌍 Future Roadmap

- [ ] FlashAttention Integration
- [ ] Quantized Attention Blocks
- [ ] Mixture of Experts
- [ ] Streaming ASR
- [ ] RLHF + PPO Support
- [ ] Vision Transformers

---

## 🧑‍💻 Contributing

We welcome contributors:

- 🔁 Add new models or ASR frontends
- 📊 Improve test coverage
- 🧠 Propose architectural variants
- 📚 Help write tutorials

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ✨ Final Note

> **"TorchiFy is not just code—it's a canvas for building the next frontier of machine intelligence. Let's build from scratch, understand deeply, and create openly."**

**GitHub**: [RishitSaxena55/TorchiFy-v1](https://github.com/RishitSaxena55/TorchiFy-v1)  
**Author**: [Rishit Saxena](https://www.linkedin.com/in/rishit-saxena-12922531b/)
