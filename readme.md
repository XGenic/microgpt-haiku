# MicroGPT Haiku

A small educational project exploring the mechanics of modern transformer-based language models by training a GPT from scratch to generate haikus.

This project began with Andrej Karpathy's MicroGPT/Makemore concepts and evolved into an exploration of tokenizer design, transformer architecture, training dynamics, and GPU acceleration. The goal was never to produce state-of-the-art poetry, but to understand *why* modern LLMs work.

## Goals

* Understand the core components of a transformer language model.
* Explore how different tokenization strategies affect generation.
* Compare a pure Python implementation with a PyTorch implementation.
* Learn the distinction between training, inference, checkpoints, and sampling.
* Produce coherent (or at least entertaining) AI-generated haikus.

## Dataset

The model was trained on a dataset of approximately 11,000 haikus.

Each haiku was converted into a structured format before training:

```text
<HAIKU>
<L1><5> winter morning light
<L2><7> coffee steam climbs through silence
<L3><5> cars hiss on wet roads
<END>
```

Special tokens explicitly describe the poem structure while the remaining text is modeled autoregressively.

## Architecture

The original implementation includes a minimal GPT-style transformer built entirely from scratch.

Features include:

* Character-level tokenization with structural tokens
* Learned token embeddings
* Learned positional embeddings
* Multi-head causal self-attention
* Feed-forward network
* Residual connections
* RMSNorm
* Cross-entropy loss
* Adam optimizer
* Temperature-based sampling

The initial implementation uses a custom scalar autograd engine inspired by Micrograd to expose every mathematical operation.

## Evolution

The project naturally progressed through several stages.

### Pure Python

The original implementation was intentionally minimal and educational.

Advantages:

* Every operation is visible.
* Easy to understand backpropagation.
* Excellent for learning.

Disadvantages:

* Extremely slow.
* Every scalar operation creates part of a computation graph.
* Long context lengths become impractical.

Training example:

* ~300 training steps
* ~40 minutes

### PyTorch

The same model was later rewritten using PyTorch tensors.

Advantages:

* GPU acceleration
* Vectorized operations
* Efficient automatic differentiation
* Orders-of-magnitude faster training

Training example:

* ~3000 training steps
* Under one minute on an NVIDIA RTX 3080 Ti Laptop GPU

The PyTorch implementation dramatically reduced training time while producing noticeably more coherent generations.

## Sample Output

Example generated haikus after training:

```text
<HAIKU>
<L1><5> what is a haiku
<L2><7> five seven five
<L3><5> rhyming not needed
<END>
```

```text
<HAIKU>
<L1><5> the moon shines brightly
<L2><7> the sun hides behind a rock
<L3><5> the moon is in a sky
<END>
```

While imperfect, these outputs demonstrate the gradual emergence of language structure and thematic understanding from statistical learning alone.

## What I Learned

This project provided practical exposure to:

* Tokenization
* Vocabulary construction
* Embedding layers
* Positional encoding
* Self-attention
* Multi-head attention
* Residual networks
* Layer normalization
* Teacher forcing
* Cross-entropy loss
* Gradient descent
* Adam optimization
* Context windows
* Temperature sampling
* Training vs inference
* Model checkpointing
* CUDA acceleration with PyTorch

Perhaps the biggest takeaway was observing how language understanding emerges gradually during training—from random characters, to fragments of words, to coherent phrases, and eventually to recognizable poetic themes.

## Future Ideas

* Byte Pair Encoding (BPE)
* Word or subword tokenization
* Validation split and checkpoint selection
* Larger transformer architectures
* LoRA fine-tuning experiments
* Coin recognition vision model
* Retrieval-Augmented Generation (RAG)
* Attention visualization
* Fine-tuning an existing small language model

## Disclaimer

This project is intended as a learning exercise rather than a production language model. Its primary purpose is to build intuition for how transformer language models are constructed and trained from first principles.
