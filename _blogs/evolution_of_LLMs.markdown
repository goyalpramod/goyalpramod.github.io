<!-- ---
layout: blog
title: "Evolution of LLMs"
date: 2025-02-12 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

[TOP MEME, MOnkey to man evolution yearwise and leave the man with no model (to signify no AGI yet)]
[Add tree from this paper to clarify that it was not a linear evolution but a tree wise, much like real theory of evolution https://arxiv.org/pdf/2304.13712]

The landscape of language models has evolved dramatically since the introduction of the Transformer architecture in 2017. Here we explore the

- mathematical foundations
- architectural innovations
- training breakthroughs

From attention mechanisms to constitutional AI, we'll dive deep into the code, math, and ideas that revolutionized NLP.

The blog ["Transformer models: an introduction and catalog — 2023 Edition"](https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/) helped me immensely while making the timeline. Additionally you can treat this blog as a sort of part 2, to my original blog on transformers which you can checkout [here]().

## How this blog is structured

Year wise innovation, for each specific model. Link to special implementation, as many share such ideas. Only model specific innovation in it's section. 

below each model, link to the next model if you interested. 

"""
We will categorize each model according to the following properties: Family, Pretraining Architecture, Pretraining or Fine-tuning Task, Extension, Application, Date (of first known publication), Number of Parameters, Corpus, License, and Lab.

Family represents what original foundation model the specific model is extending, extension describes what the model is adding to the one it is deriving from, Date is when the model was firts published, Number of parameters of the pretrained model, Corpus is what data sources the model was pre-trained or fine-tuned on, License describes how the model can be legally used, and Lab lists the institution that published the model.
""" {TAKEN FROM THE BLOG MENTIONED ABOVE}

Only included os models, so no claude3.5, gpt 4 etc even though we love em

## A short introduction to how LLMs are trained & inferenced

This part is highly influenced by this [video](https://www.youtube.com/watch?v=7xTGNNLPyMI) by andrej karpathy

A paper on pretraining [paper](https://arxiv.org/pdf/2003.08271)

[Transformer catalog](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit?gid=0#gid=0)

### Encoder only models 

### Decoder only models 

### Pretraining  

[CHECK IF COHERE RELEASED ANYTHING SIGNIFICANT, AND IF ANTHROPIC HAD ANY PAPERS]

USE THIS [PAPER](https://arxiv.org/pdf/2106.04554) 

## 2017: The Foundation Year

### Early Activation Functions

- GELU (Gaussian Error Linear Unit)
- Swish/SiLU
- Comparison with ReLU and ELU

Glove, word2vec?

## 2018: BERT and Early Innovations

### BERT's Architecture

- Bidirectional encoder
- WordPiece tokenization
- [CLS] and [SEP] tokens
- NSP and MLM objectives

"""
Link: https://huggingface.co/docs/transformers/model_doc/bert
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM/NSP
Extension:It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder
Application:General Language Understanding and Question Answering. Many other language applications followed
Date (of first known publication): 10/2018
Num. Params:Base = 110M, Large = 340MT
Corpus:Toronto Book Corpus and Wikipedia (3.3B Tokens)
License: Open, Apache-2.0
Lab:Google
"""


### GPT-1

- Unidirectional decoder
- BPE tokenization
- Zero-shot capabilities
- Language modeling objective

"""
Link: https://huggingface.co/docs/transformers/model_doc/openai-gpt
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension:
Application: Text generation, but adaptable to many other NLP tasks when fine tuned.
Date (of first known publication): 06/2018
Num. Params:117M
Corpus: Unsupervised Pretraining on BookCorpus dataset. Supervised Finetuning on several task-specific datasets including SNLI, RACE, Quora…
License: N/A
Lab: OpenAI
"""

### Training Innovations

- Warm-up learning rate schedules
- Adam optimizer variants
- Gradient clipping strategies

## 2019: Scaling and Efficiency

### GPT-2

"""
Link: https://huggingface.co/docs/transformers/model_doc/gpt2
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Minor extensions to the GPT architecture (e.g. layer normalization moved to the input of each sub-layer, or increased context size from 512 to 1024)
Application: Text generation, but adaptable to many other NLP tasks when fine tuned.
Date (of first known publication): 02/2019
Num. Params: 124M, 355M, 774M, 1.5B
Corpus: 8 million web pages (40 GB). 10X GPT . WebText dataset is created by crawling all links at Reddit with at least 3 Karma points.
License: Open, Modified MIT license
Lab: OpenAI
"""

### RoBERTa

- Dynamic masking
- Removed NSP
- Larger batch sizes
- Extended training

"""
Link: https://huggingface.co/docs/transformers/model_doc/roberta
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM (Dynamic)
Extension: Extension of BERT with optimized training procedure and more data
Application: Same as BERT
Date (of first known publication): 07/2019
Num. Params: 356M
Corpus: Same as BERT + CC News + OpenWebText + Stories ( 33B Tokens)
License: N/A
Lab: UW/Google
"""

### DistilBERT and Model Compression

- Knowledge distillation techniques
- Parameter sharing
- Pruning strategies
- Quantization methods

"""
Link: https://huggingface.co/docs/transformers/model_doc/distilbert
Family: BERT
Pretraining Architecture: Encoder
Pretraining Task: MLM/NSP
Extension: Compressed version of BERT using distillation, which is much more efficient given the same number of parameters
Application: Same as BERT
Date (of first known publication): 10/2019
Num. Params:66M
Corpus: Same as BERT
License: Open, Apache-2.0
Lab: Huggingface
"""

### ALBERT

- Cross-layer parameter sharing
- Factorized embedding parameterization
- Sentence ordering prediction

### AlphaFold

"""
Link: https://github.com/deepmind/alphafold
Family: SE(3) Transformer
Pretraining Architecture: Encoder
Pretraining Task: Protein folding prediction of BERT using parameter sharing, which is much more efficient given the same number of parameters
Extension: The original Alphafold used a BERT-style Transformer. The details of Alphafold’s Transformer are not known, but it is believed it is an extension of the SE(3)-Tranformer, a 3-D equivariant Transformer
Application: Protein folding
Date (of first known publication): 09/2019
Num. Params:b12M, Large = 18M, XLarge = 60M*
Corpus: Same as BERT
License: the code is open sourced, with Apache-2.0
Lab: Deepmind
"""

### BART

"""
Link: https://huggingface.co/docs/transformers/model_doc/bart
Family: BERT for encoder, GPT for Decoder
Pretraining Architecture: Encoder/Decoder
Pretraining Task: DAE
Extension: It can be seen as a generalization of BERT and GPT in that it combines ideas from both in the encoder and decoder
Application: Mostly text generation but also some text understanding tasks
Date (of first known publication): 10/2019
Num. Params: Base = 140M, Large = 400M. In general, roughly 10% larger than BART for equivalent architectures.
Corpus:Same as RoBERTa (160Gb of news, books, stories)
License: Open, Apache-2.0
Lab:Facebook
"""

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

"""
Link: https://huggingface.co/docs/transformers/model_doc/t5
Family: Transformer
Pretraining Architecture: Encoder/Decoder
Pretraining Task: DAE
Extension: Same as original Transformer with some additions such as relative positional embeddings like Transformer XL
Application: General language tasks including machine translation, question answering, abstractive summarization, and text classification
Date (of first known publication): 10/2019
Num. Params: 11 B (up to)
Corpus: Colossal Clean Crawled Corpus (C4) — Cleaned up version of the Common Crawl dataset — 750 GB
License: Open, Apache-2.0
Lab: Google
"""

### Architecture Innovations

- Sparse Transformers
- Reformer
- Longformer
- Linear attention mechanisms

## 2021: Instruction Tuning and Alignment

### InstructGPT

- RLHF pipeline [blog on the topic](https://huggingface.co/blog/rlhf) & [blog 2](https://wandb.ai/ayush-thakur/RLHF/reports/Understanding-Reinforcement-Learning-from-Human-Feedback-RLHF-Part-1--VmlldzoyODk5MTIx)
- PPO implementation
- Human feedback collection
- Alignment techniques

"""
Link: https://github.com/openai/following-instructions-human-feedback
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: GPTInstruct starts off with a pretrained GPT3 model and adds reward modeling through reinforcement learning after a supervised finetuning
Application: Knowledge-intensive dialog or language tasks
Date (of first known publication): 01/2022
Num. Params: Same as GPT3
Corpus: Same as GPT3 for pretraining, but finetuned and optimized using labeler data and prompts
License: Closed source, Accessible through API
Lab: OpenAI
"""

### PaLM

- Pathways system
- Scaled dot product attention
- Multi-query attention
- Parallel training techniques

"""
Link: https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html
Family: Transformer
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Palm uses a typical decoder-only Transformer architecture, but adds quite a few extensions: SwiGLU activations, parallel layers, multi-query attention, RoPE embeddings, Shared Input-Output Embeddings, no biases, and a 256k SentencePiece vocabulary generated from the training data.
Application: PalM is designed as a general purpose language model with applicability to hundreds of different language tasks
Date (of first known publication): 04/2022
Num. Params: 540B
Corpus: 780B tokens from filtered webpages, books, Wikipedia, news articles, source code, and social media conversations. Code includes 24 programming languages.
License: Closed source, Accessible through API
Lab: Google
"""

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

"""
Link: https://huggingface.co/docs/transformers/model_doc/bloom
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Main difference to GPT-3 is that it uses full attention instead of sparse attention
Application: Same as GPT-3
Date (of first known publication): 07/2022
Num. Params:176B
Corpus: 366B tokens (1.5 TB of text data) multilingual dataset
Lab: Big Science/Huggingface
License: Open, but need to follow restrictions in Attachment A, BigScience RAIL License v1.0
"""

### OPT

- Reproducible training
- Open source weights
- Training dynamics study
- Cost analysis

### Chinchilla

"""
Link: https://arxiv.org/abs/2203.15556
Family: GPT
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: Same as Gopher but with optimizations to reduce model size and therefore training/inference time with equal or superior performance
Application: Same as Gopher/GPT3
Date (of first known publication): 03/2022
Num. Params:70B
Corpus: Massive Text
License: Closed source.
Lab: Deepmind
"""

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

### LLaMA

- Efficient scaling
- Flash Attention-2
- Chat templates
- RLHF improvements

"""
Link: https://huggingface.co/docs/transformers/main/model_doc/llama
Family: Transformer
Pretraining Architecture: Decoder
Pretraining Task: LM
Extension: LLaMA uses a Transformer architecture, and with extensions: Pre-normalization, SwiGLU activations, RoPE embeddings, reduced memory usage and runtime through efficient implementation of the causal multi-head attention, checkpointing to reduce the amount of activations that are recomputed during the backward pass, model and sequence parallelism to reduce memory usage of the model, and uses 1.4T BPE tokens after tokenization.
Application: Zero and few shot Commonsense reasoning, Question answering, Code generation and Reading comprehension.
Date (of first known publication): 02/2023
Num. Params: 7B, 13B, 33B and 65B
Corpus: English CommonCrawl + C4 + Github + Wikipedia + Gutenberg and Books3 + ArXiv + Stack Exchange
License: Limited, Non-commercial bespoke license
Lab: Meta
"""

### LLaMA 2

### MamBa



### Alpaca
    """
    Link: https://github.com/tatsu-lab/stanford_alpaca
    Family: LLaMA
    Pretraining Architecture: Decoder
    Fine-tuning Task: human instructions
    Extension: Alpaca is fine-tuned from a 7B LLaMA model.
    Application: Evaluated on a variety of text generation and classification tasks.
    Date (of first known publication): 03/2023
    Num. Params: 7B
    Corpus: 52K instruction-following data generated using self-instruct mechanism, from 175 human-written instruction-output pairs.
    License: Limited, Non-commercial bespoke license
    Lab: Stanford
    """

### Training Advances

- Direct Preference Optimization (DPO)
- Constitutional AI implementation
- Medprompt fine-tuning
- Continued pre-training

{IG qwen and deepseek come here}


{Add mistral as well}

{interLM? https://huggingface.co/internlm}

{LLaVA? https://llava-vl.github.io/}

{Do I include VLMs? Where?}
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

### LLaMA 3

{add quen and deepseek}

## Technical Deep Dives

### Architecture Components

#### MOE

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
- Chain-of-thought evaluation

 -->
