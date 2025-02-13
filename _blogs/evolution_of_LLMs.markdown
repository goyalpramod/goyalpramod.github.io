<!-- ---
layout: blog
title: "Evolution of LLMs"
date: 2025-02-12 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

The landscape of language models has evolved dramatically since the introduction of the Transformer architecture in 2017. Here we explore the

- mathematical foundations
- architectural innovations
- training breakthroughs

From attention mechanisms to constitutional AI, we'll dive deep into the code, math, and ideas that revolutionized NLP.

## 2017: The Foundation Year

### Early Activation Functions

- GELU (Gaussian Error Linear Unit)
- Swish/SiLU
- Comparison with ReLU and ELU

## 2018: BERT and Early Innovations

### BERT's Architecture

- Bidirectional encoder
- WordPiece tokenization
- [CLS] and [SEP] tokens
- NSP and MLM objectives

### GPT-1

- Unidirectional decoder
- BPE tokenization
- Zero-shot capabilities
- Language modeling objective

### Training Innovations

- Warm-up learning rate schedules
- Adam optimizer variants
- Gradient clipping strategies

## 2019: Scaling and Efficiency

### RoBERTa

- Dynamic masking
- Removed NSP
- Larger batch sizes
- Extended training

### DistilBERT and Model Compression

- Knowledge distillation techniques
- Parameter sharing
- Pruning strategies
- Quantization methods

### ALBERT

- Cross-layer parameter sharing
- Factorized embedding parameterization
- Sentence ordering prediction

## 2020: The Scale Revolution

### GPT-3

- In-context learning
- Few-shot capabilities
- Scaling laws discovery
- Batch size scaling

### T5

- Encoder-decoder architecture
- Unified text-to-text framework
- Span corruption
- Multi-task pre-training

### Architecture Innovations

- Sparse Transformers
- Reformer
- Longformer
- Linear attention mechanisms

## 2021: Instruction Tuning and Alignment

### InstructGPT

- RLHF pipeline
- PPO implementation
- Human feedback collection
- Alignment techniques

### PaLM

- Pathways system
- Scaled dot product attention
- Multi-query attention
- Parallel training techniques

### Training Innovations

- Chain-of-thought prompting
- Constitutional AI principles
- SFT (Supervised Fine-Tuning)
- Mixture of Experts (MoE)

## 2022: Democratization

### BLOOM

- Multilingual pre-training
- Carbon footprint considerations
- Distributed training
- Community governance

### OPT

- Reproducible training
- Open source weights
- Training dynamics study
- Cost analysis

### Architectural Improvements

- Flash Attention
- Rotary embeddings
- Grouped-query attention
- ALiBi position encoding

## 2023: Multi-Modal and Reasoning

### GPT-4

- Multi-modal encoders
- System prompting
- Advanced reasoning capabilities
- Tool use

### LLaMA & LLaMA 2

- Efficient scaling
- Flash Attention-2
- Chat templates
- RLHF improvements

### Training Advances

- Direct Preference Optimization (DPO)
- Constitutional AI implementation
- Medprompt fine-tuning
- Continued pre-training

## 2024: Efficiency and Performance

### Gemma

- Efficient attention mechanisms
- Advanced position embeddings
- Improved tokenization
- Memory efficient training

### Claude 3

- Multi-modal understanding
- Tool use capabilities
- Advanced reasoning
- Constitutional AI improvements

## Technical Deep Dives

### Architecture Components

#### Attention Mechanisms

- Vanilla attention
- Multi-head attention
- Cross-attention
- Flash attention variants
- Sparse attention patterns

#### Position Embeddings

- Absolute
- Relative
- Rotary (RoPE)
- ALiBi
- T5 relative bias

#### Normalization and Residuals

- Layer normalization
- RMSNorm
- Pre-LN vs Post-LN
- Residual connections
- Skip connections

### Training Methods

#### Pre-training Objectives

- MLM (Masked Language Modeling)
- CLM (Causal Language Modeling)
- Span corruption
- RTD (Replaced Token Detection)
- Prefix Language Modeling

#### Fine-tuning Strategies

- Instruction tuning
- RLHF pipeline
- DPO (Direct Preference Optimization)
- LoRA and QLoRA
- Parameter efficient fine-tuning

#### Optimization Techniques

- Adam variants
- Learning rate schedules
- Gradient accumulation
- Mixed precision training
- ZeRO optimizer stages

### Efficiency Innovations

#### Model Compression

- Quantization (INT4/8)
- Pruning techniques
- Knowledge distillation
- Low-rank adaptation
- Sparse inference

#### Memory Optimization

- Gradient checkpointing
- Activation recomputation
- Memory efficient attention
- Selective activation storage
- CPU offloading

### Evaluation Framework

#### Language Understanding

- GLUE and SuperGLUE
- MMLU
- BIG-bench
- TruthfulQA
- GSM8K

#### Safety and Alignment

- TruthfulQA
- Anthropic's Constitutional AI eval
- Bias and toxicity metrics
- HONEST framework
- Safety benchmarks

#### Reasoning and Capabilities

- HumanEval
- MATH
- BBH (Big Bench Hard)
- HELM framework
- Chain-of-thought evaluation -->
