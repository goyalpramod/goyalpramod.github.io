<!-- ---
layout: blog
title: "Evolution of LLMs"
date: 2025-05-05 12:00:00 +0530
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

 Additionally you can treat this blog as a sort of part 2, to my original blog on transformers which you can checkout [here](https://goyalpramod.github.io/blogs/Transformers_laid_out/).

## How this blog is structured

First we would begin with a short to the transformers (Most LLM structure begin this way), then go sequentially how they are trained and infererenced. To lay a groundwork, on top of which we can build.

After which we will have a look at a short AI timeline over the years, going year by year seeing the most influential work that shaped that and future years of LLMs.

Where we will see what architectural innovation, computational breakthrough, training optimization that were invented over the years and how it affected LLMs and their benchmarks.

Finally and most importantly we will dive deep into the technical understanding and implementation of these different techniques. Some of them being Flash Attention, KV-Caching, GRPO etc.  


Additionally there have been a lot of innovations in vision modeling, TTS, Image gen, Video gen etc each of which deserves it's own blog(And there will be!! I promise you that). Over here I will just give quick intro and links to some ground breaking innovations.

> NOTE: Do not take for granted all the hardware, data and benchmark innovations, Though I will briefly mention them in the timeline. I implore you to explore them further if they interest you. This blog is strictly restricted to breakthroughs in Large Language Models, and mostly open source one's. Even though current models by OpenAI are amazing, not much is known about them to the public. So we will briefly talk about what we know about them, then move on to talk about mostly open source models.


## A short introduction to LLMs

This part is highly influenced by this [video](https://www.youtube.com/watch?v=7xTGNNLPyMI) by andrej karpathy

A paper on pretraining [paper](https://arxiv.org/pdf/2003.08271)


### Architecture

We will be skipping over the internal details about the transformers model (you can read more about it in my previous blog), I will proceed with the assumption that you have a very deep level understanding of atleast the transformer model. Having that that let us proceed. 

As we can see the original transformer has two parts, an Encoder and a Decoder. And as is known, it was initially made for the sole purpose of Machine Translation. 

But over the years they have been used for a plethora of tasks from 
- Question Answering 
- Summarization 
- Tagging 
- Classification 

And many more. 

LLMs consist of architectures which are solely based on the Encoder like Bert 

[ADD_IMAGE_OF_BERT]

LLMs consist of architectures which are solely based on the Decoder like gpt-1 

[ADD_IMAGE_OF_GPT1]

There are LLMs which use the good ol Encoder Decoder layer together too like T5

[WITH_IMAGE]

One thing to be mindful of is, that the development of LLMs have been done on the transformers. There was no radical shift, only gradual and slow incremental changes. As you read more about them you will understand them better.

In crux you only need to understand the basic Transformer architecture to understand most if not all LLMs.

### Training

As I mentioned there are 3 kinds of architectures when it comes to LLM, hence there is a different way of training each. Even though they share few similarities. Each has a different objective. Let us begin by first understanding the most popular LLM architecture, that being the decoder only architecture. 

#### Decoder Only 

#### Encoder Only 

#### Encoder Decoder

### Inference


## The AI timeline

This is a very short timeline of the most influential work, to read about more architectures that were huge at the time but died down eventually, consider going through the [Transformer catalog](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit?gid=0#gid=0).

The blog ["Transformer models: an introduction and catalog — 2023 Edition"](https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/) helped me immensely while making the timeline.

### 2017: The Foundation Year

#### Early Activation Functions

- Swish/SiLU


#### The Transformer Architecture
- Multi-head attention mechanism
- Positional encodings
- Layer normalization
- Feed-forward networks
- Encoder-decoder structure

#### Swish Activation Function
- Self-gated activation (x·sigmoid(βx))
- Smooth alternative to ReLU
- Discovered through neural architecture search

#### Attention Mechanisms
- Scaled dot-product attention
- Multi-head attention
- Self-attention
- Masked attention for autoregressive models

#### Key Papers and Implementations
- "Attention Is All You Need" (Vaswani et al.)
- "Searching for Activation Functions" (Ramachandran et al.)
- Tensor2Tensor library

#### Hardware and Training Innovations
- TPU v2 acceleration
- Gradient accumulation techniques
- Large-batch training methods

### 2018: BERT and Early Innovations

#### BERT's Architecture

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


#### GPT-1

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

#### Hardware Innovations
- **NVIDIA Tesla V100** (NVIDIA)
  - Specialized for AI training workloads with tensor cores
  - Enabled training of larger language models

#### Training Innovations

- Warm-up learning rate schedules
- Adam optimizer variants
- Gradient clipping strategies

### 2019: Scaling and Efficiency

#### GPT-2

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

#### RoBERTa

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

#### DistilBERT and Model Compression

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

#### ALBERT

- Cross-layer parameter sharing
- Factorized embedding parameterization
- Sentence ordering prediction

#### AlphaFold

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

#### BART

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

#### Notable Model Releases
- **XLNet** (Google/CMU)
  - Permutation-based training approach
  - Surpassed BERT on multiple benchmarks

- **Megatron** (NVIDIA)
  - Model parallelism for efficient large model training

#### Training Innovations
- **Sparse Attention Patterns** (OpenAI)
  - Reduced computational complexity for long sequences


"""
Clarification Needed

AlphaFold: The original AlphaFold was indeed presented at CASP13 in December 2018 as noted in Wikipedia: "In December 2018, DeepMind's AlphaFold placed first in the overall rankings of the 13th Critical Assessment of Techniques for Protein Structure Prediction (CASP)." Wikipedia While it used deep learning techniques, it wasn't specifically based on the SE(3)-Transformer architecture. The more advanced transformer-based version (AlphaFold 2) was released later in 2020.

The "SE(3)-Transformer" architecture was incorporated into later versions of AlphaFold, particularly AlphaFold 2 which was released in 2020. The original 2019 AlphaFold used convolutional neural networks rather than transformers as mentioned in this article: "one of the major differences between AlphaFold 1 and AlphaFold 2 is that the former used concurrent neural networks (CNNs) and the new version uses Transformers." 
"""

### 2020: The Scale Revolution

#### GPT-3

- In-context learning
- Few-shot capabilities
- Scaling laws discovery
- Batch size scaling

#### T5

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

#### Architecture Innovations

- Sparse Transformers
- Reformer
- Longformer
- Linear attention mechanisms


#### Notable Model Releases
- **Meena** (Google)
  - Specialized conversational model
  - 2.6B parameters

- **Turing-NLG** (Microsoft)
  - 17B parameters
  - Advanced natural language generation

- **Pangu-α** (Huawei)
  - 200B parameters
  - Chinese language model

#### Hardware Advancements
- **TPU v3** (Google)
  - Enhanced matrix multiplication acceleration

#### Training Methodologies
- **ZeRO (Zero Redundancy Optimizer)** (Microsoft)
  - Memory optimization for distributed training

"""
Your 2020 section is quite accurate, but I can provide a few important additions and clarifications:

### GPT-3 Release Date and Impact

GPT-3 was officially announced on May 28, 2020 as noted in Wikipedia: "On May 28, 2020, an arXiv preprint by a group of 31 engineers and researchers at OpenAI described the achievement and development of GPT-3, a third-generation 'state-of-the-art language model'."

You correctly highlight its key innovations:
- In-context learning
- Few-shot capabilities
- Scaling laws discovery
- Batch size scaling

### T5 Publication Timeline

While T5 was first presented in a preprint in October 2019, it gained significant attention and influence in 2020. One clarification - T5 was published on arXiv in October 2019, but was officially presented at a major conference in 2020, which is why it's often associated with both years.

### Additional Notable Models from 2020

You might want to add:

1. **ELECTRA** - Google's model that used a discriminative approach instead of masked language modeling, providing more efficient training As noted, "Electra deploys a 'Masked Language Modeling' approach that masks certain words and trains the model to predict them. Additionally, Electra incorporates a 'Discriminator' network that aids in comprehending language without the need to memorize the training data."

2. **BART (Facebook/Meta)** - While introduced in 2019, BART became more prominent in 2020 as it was integrated into various applications

3. **Switch Transformer** - Google's early mixture-of-experts approach that demonstrated trillion-parameter scale was possible

### Additional Training Innovation

**Scaling Laws** - OpenAI's publication on the mathematical relationships between model size, dataset size, and computational budget demonstrated predictable patterns for improving performance This was part of the GPT-3 research which showed "that scaling up language models greatly improves task-agnostic, few-shot performance."

Overall, your 2020 section captures most of the major developments, with these additions providing a more complete picture of this transformative year in LLM development.
"""

### 2021: Instruction Tuning and Alignment

#### InstructGPT

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

#### PaLM

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

#### Training Innovations

- Chain-of-thought prompting
- Constitutional AI principles
- SFT (Supervised Fine-Tuning)
- Mixture of Experts (MoE)


#### Notable Model Releases
- **Jurassic-1** (AI21 Labs)
  - 178B parameters
  - Language understanding with specialized abilities

- **CPM-2** (Baidu)
  - Chinese pre-trained model
  - Multilingual capabilities

- **HyperCLOVA** (Naver)
  - 204B parameters
  - Korean language model

- **T0** (BigScience)
  - Zero-shot capabilities through multi-task prompted training

#### Hardware Advancements
- **SambaNova DataScale** (SambaNova)
  - Specialized AI accelerator architecture
  - Alternative to traditional GPU-based training

#### Architectural Innovations
- **Switch Transformer** (Google)
  - Mixture of experts approach
  - Trillion parameter models

"""
Let me check the accuracy of your 2021 section and look for any important additions.

Your 2021 section is generally accurate, but there are a few important additions that would enhance the completeness of your timeline. 

### Models You're Missing

**Gopher (DeepMind)**
- 280B parameter model released in December 2021 DeepMind introduced this model as a "280 billion parameter model" that was "evaluated on 152 diverse tasks, achieving state-of-the-art performance across the majority."
- Demonstrated significant scaling benefits in reading comprehension and fact-checking
- Represented a major advancement in model scale from DeepMind

**Megatron-Turing NLG (Microsoft & NVIDIA)**
- 530B parameter model announced in October 2021
- Combined Microsoft's Turing and NVIDIA's Megatron technologies
- Demonstrated advanced distributed training techniques
- Applied significant hardware optimization for large-scale training

**GLaM (Google)**
- Mixture of Experts approach with 1.2 trillion parameters (sparsely activated)
- Data-efficient alternative to dense models
- Demonstrated competitive performance with significantly less computational cost

### Additional Technical Innovations

**Chinchilla Scaling Laws**
While the Chinchilla model itself wasn't released until 2022, the research behind it began in 2021, establishing important scaling principles that:
- Showed optimal token-to-parameter ratios should be approximately 20:1 This research found that "we need around 20 text tokens per parameter" for optimal training.
- Demonstrated many existing models were significantly undertrained
- Influenced the training methodology of subsequent models

**Training Data Quality**
- Improved data cleaning and filtering techniques
- Development of specialized corpora like The Pile
- Increased focus on dataset curation rather than just scale

### Other Significant Developments

- **Constitutional AI research** began to take shape, though formal paper publications would come later
- **Increased focus on safety and alignment** through careful supervision and filtering
- **Multi-modal foundations** were being laid, though primarily text-focused models dominated 2021

Your section does a good job covering the key developments around instruction tuning and alignment techniques (particularly RLHF and PPO), but these additions would provide a more comprehensive view of the LLM landscape in 2021.
"""

### 2022: Democratization

#### BLOOM

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

#### OPT

- Reproducible training
- Open source weights
- Training dynamics study
- Cost analysis

#### Chinchilla

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

#### Architectural Improvements

- Flash Attention
- Rotary embeddings
- Grouped-query attention
- ALiBi position encoding


#### Notable Model Releases
- **Galactica** (Meta)
  - Scientific knowledge model
  - 120B parameters

- **Anthropic Claude 1** (Anthropic)
  - Initial release focusing on helpfulness and harmlessness

- **GLaM** (Google)
  - Mixture of experts model
  - 1.2 trillion parameters (sparsely activated)

- **ERNIE 3.0** (Baidu)
  - Enhanced knowledge integration
  - Multilingual capabilities

#### Hardware Advancements
- **Cerebras CS-2** (Cerebras)
  - Wafer-scale engine for AI computation
  - Alternative architecture for AI training

#### Training Methodologies
- **FLAN (Fine-tuned LAnguage Net)** (Google)
  - Instruction tuning across multiple tasks
  - Improved zero-shot performance

#### Benchmark Developments
- **HELM (Holistic Evaluation of Language Models)** (Stanford)
  - Comprehensive benchmark suite for LLMs
  - Standardized evaluation metrics

"""
I'll verify your 2022 section and check for any missing developments.

Your 2022 section is largely accurate, but there are some important additions to include, particularly regarding multimodal models and text-to-image systems that emerged as significant developments that year.

### Multimodal Models to Add

**Flamingo (DeepMind)**
- Released in April 2022 as a "family of Visual Language Models (VLM)" designed for few-shot learning with visual inputs
- Pioneered visual-language integration capabilities
- Demonstrated strong few-shot learning in multimodal space
- Set benchmarks for vision-language tasks

**DALL-E 2 (OpenAI)**
- Released in April 2022
- Significant improvement over original DALL-E
- Demonstrated remarkably detailed text-to-image generation
- Maintained controlled access with gradual rollout

**Stable Diffusion (Stability AI)**
- Released in August 2022 as "a deep learning, text-to-image model" that became "the premier product of Stability AI"
- Open-source alternative to DALL-E 2
- Democratized access to high-quality image generation
- Trained on LAION-5B dataset

### Additional LLM Developments

**Sparrow (DeepMind)**
- Dialogue-optimized model built from Chinchilla
- Emphasized safety and helpful responses
- Incorporated reinforcement learning from human feedback
- Used rule-based constraints to guide model behavior

**Unified-IO (Allen Institute)**
- Multi-task, multi-modal model
- Demonstrated shared representations across vision and language tasks
- Single architecture for diverse AI tasks

### Technical Innovations Worth Adding

**Efficient Attention Mechanisms**
- Improved computational efficiency for long context processing
- Reduced memory requirements for large model inference
- Enhanced throughput for deployed models

**Quantization Advances**
- Post-training quantization techniques for model compression
- Reduced inference costs while maintaining performance
- Enabled deployment on consumer hardware

### Community and Ecosystem Developments

**HuggingFace Hub Growth**
- Expanded repository of pre-trained models
- Democratized access to fine-tuning and deployment tools
- Created standard interfaces for model sharing

**Ethical AI Guidelines**
- Expanded frameworks for responsible AI deployment
- Increased focus on documentation and transparency
- Development of tools for bias detection and mitigation

These additions would provide a more comprehensive view of the 2022 AI landscape, particularly highlighting the emergence of multimodal capabilities and open-source developments that significantly shaped the field.
"""

### 2023: Multi-Modal and Reasoning

#### GPT-4

- Multi-modal encoders
- System prompting
- Advanced reasoning capabilities
- Tool use

#### LLaMA

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

#### LLaMA 2

#### MamBa



#### Alpaca
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

#### Training Advances

- Direct Preference Optimization (DPO)
- Constitutional AI implementation
- Medprompt fine-tuning
- Continued pre-training


#### Notable Model Releases
- **PaLM 2** (Google)
  - Improved multilingual capabilities
  - Enhanced reasoning

- **LAION-5B** (LAION)
  - Large-scale image-text dataset
  - Enabled better multimodal training

- **Vicuna** (LMSYS)
  - Fine-tuned LLaMA
  - Open-source conversational agent

- **Alpaca** (Stanford)
  - Instruction-tuned LLaMA
  - Efficient fine-tuning approach

- **Yi** (01.AI)
  - Bilingual Chinese-English model
  - 34B parameters

- **MPT** (MosaicML)
  - Open-source model with commercial usage rights
  - Efficient training techniques

#### Hardware Advancements
- **Graphcore IPU** (Graphcore)
  - Intelligent Processing Unit
  - Alternative architecture for AI computation

#### Training Methodologies
- **LIMA (Less Is More for Alignment)** (Meta)
  - Demonstrated efficiency of small high-quality datasets
  - 1,000 examples for alignment

- **UL2 (Unified Language Learner)** (Google)
  - Unified approach to pre-training
  - Combined multiple objectives

#### Architectural Innovations
- **Mamba** (Albert Gu & Tri Dao)
  - State space model for sequence modeling
  - Linear scaling with sequence length

"""
I'll research and verify your 2023 section with the latest information.

Your 2023 section is quite good, but there are a few important models and developments to add for a more comprehensive picture:

### Multimodal Models to Add

**LLaVA (Visual Instruction Tuning)**
- Released in April 2023 LLaVA was among the first vision-language models created using visual instruction tuning
- Combined vision encoders with language models
- Pioneered efficient visual instruction tuning
- Set foundation for open-source multimodal models

**Claude 1/Claude 2 (Anthropic)**
- Released in March 2023 (Claude 1) and July 2023 (Claude 2)
- Focused on constitutional AI approach
- Enhanced safety and alignment
- Specialized in long-form content generation

**Gemini (Google)**
- Announced initially in May 2023, fully released in December Described as "a family of multimodal large language models developed by Google DeepMind, and the successor to LaMDA and PaLM 2"
- Designed from the ground up as a multimodal model
- Positioned as Google's answer to GPT-4

### Additional LLM Developments

**Mistral 7B (Mistral AI)**
- Released in September 2023
- Demonstrated exceptional performance for model size
- Open weights with permissive license
- Achieved near-Llama 13B performance with only 7B parameters

**Phi-1 (Microsoft)**
- Small (1.3B parameter) but remarkably capable model
- Demonstrated efficiency of specialized training data
- Focused on code and mathematical reasoning

### Technical Innovations to Highlight

**Long Context Windows**
- Significant extensions beyond previous 2-4K token limits
- Novel approaches to efficient attention for long sequences
- Enabled processing of much longer documents

**Mixture of Experts Improvements**
- Enhanced routing algorithms
- More efficient training and inference
- Reduced communication overhead

### Ecosystem Developments

**Open Source Ecosystem Growth**
- HuggingFace growing as central repository
- Emergence of specialized fine-tuning frameworks
- Improved quantization and deployment tools

**Hardware Optimizations**
- Specialized kernels for transformer operations
- Improved memory bandwidth utilization
- Novel architectures optimized for inference

The key theme for 2023 was the mainstreaming of multimodal capabilities and the explosive growth of the open-source LLM ecosystem through models like Mistral and the proliferation of Llama derivatives. The year saw significant progress in making models more efficient in both training and inference, with particular emphasis on long-context handling and multimodal integration.
"""

{IG qwen and deepseek come here}


{Add mistral as well}

{interLM? https://huggingface.co/internlm}

{LLaVA? https://llava-vl.github.io/}

{Do I include VLMs? Where?}

### 2024: Efficiency and Performance

#### MordernBert

#### Gemma

- Efficient attention mechanisms
- Advanced position embeddings
- Improved tokenization
- Memory efficient training

#### Claude 3

- Multi-modal understanding
- Tool use capabilities
- Advanced reasoning
- Constitutional AI improvements

#### LLaMA 3

{add quen and deepseek}


#### Notable Model Releases
- **Claude 3** models (Anthropic)
  - Opus, Sonnet, and Haiku variants
  - Improved reasoning and multimodal capabilities

- **phi-1/phi-2/phi-3** (Microsoft)
  - Small but powerful models
  - High performance with limited training data

- **Command** (Cohere)
  - Enterprise-focused model
  - Multilingual capabilities

- **Falcon 2** (TII)
  - Improved performance over original Falcon
  - Open licensing

- **Jamba** (AI Alliance)
  - Open mixture of experts model
  - 32B parameters

- **OpenAI o1** (OpenAI)
  - First specialized reasoning model
  - Advanced mathematical problem-solving

#### Hardware Advancements
- **Groq LPU** (Groq)
  - Language Processing Unit
  - Record-breaking inference speeds

#### Training Methodologies
- **RSO (Reinforced Self-training with Online feedback)** (DeepMind)
  - Self-improvement through AI evaluation
  - Reduced human annotation needs

- **SPIN (Self-Played Improvement Narration)** (Anthropic)
  - Self-correction capabilities
  - Improved factual accuracy

#### Benchmark Developments
- **ORCA Bench** (Microsoft)
  - Advanced reasoning evaluation
  - Complex problem-solving assessment

"""
I'll search for the latest information to verify and enhance your 2024 section.

Your 2024 section has good coverage, but I'll help enhance it with additional models and developments based on my research.

### Models to Add or Update

**Qwen 2.5 (Alibaba)**
- Released in September 2024 as "the latest addition to the Qwen family," which the developers called "the largest opensource release in history"
- Specialized variants for coding and mathematics
- Sizes ranging from 1.5B to 72B parameters
- Strong multilingual capabilities

**DeepSeek 2.5 (DeepSeek)**
- Released in September 2024 combining "DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct" as an "upgraded version"
- Competitive code generation capabilities 
- Cost-effective alternative to larger models
- 128K token context window

**Claude 3.5 Sonnet (Anthropic)**
- Released in October 2024 featuring improved performance "in undergraduate knowledge, graduate-level reasoning, general reasoning, and code generation"
- Advanced reasoning and coding capabilities
- Introduces Artifacts for interactive content creation
- Significant improvements over Claude 3 Opus

**DeepSeek-R1 (DeepSeek)**
- Specialized reasoning model released in December 2024
- Focus on mathematical and logical reasoning
- Designed to compete with OpenAI's o1
- Significantly faster inference than o1

### Architectural Advances

**Transformer Hybrids**
- Mixed attention mechanisms for efficiency and quality
- Integration of traditional transformers with newer architectures
- Specialized routing for different types of reasoning

**Attention Mechanism Innovations**
- Further optimizations of Flash Attention
- New formulations of efficient attention for long sequences
- Retrieval-based augmentation for grounded responses

### Efficiency Improvements

**Quantization Breakthroughs**
- Advances in INT4/INT8 quantization with minimal quality loss
- Hardware-aware optimizations for consumer devices
- Specialized kernels for mobile deployment

**Token Efficiency**
- New tokenization strategies for multilingual support
- Context compression techniques
- Token-pruning methodologies for inference speedup

### Tooling and Ecosystem

**Inference Optimization Frameworks**
- vLLM and similar tools for high-throughput inference
- Parallel decoding techniques
- Specialized tools for multimodal deployment

**Advanced API Capabilities**
- Tool use standardization
- Vision-language improvements
- Function calling enhancements

2024 has seen a remarkable convergence in model capabilities with many companies releasing models that rival or surpass previous leaders, as "LLMs around 10B params converge to GPT-3.5 performance, and LLMs around 100B and larger converge to GPT-4 scores" while focusing on specialization, efficiency, and multimodal capabilities as the main differentiators.
"""

### 2025


#### Llama 4

#### Qwen

#### DeepSeek

(there were some amazing developments on tts, video gen, image gen etc but all of those for a different video)




### 2025: Reasoning and Compression

#### Notable Model Releases
- **DeepSeek-MoE** (DeepSeek)
  - Mixture of experts architecture
  - Efficient scaling

- **Grok** (xAI)
  - Open-source model
  - 314B parameters

- **Pixtral** (Mistral AI)
  - Multimodal capabilities
  - 12B parameters

- **Qwen2** (Alibaba)
  - Multilingual capabilities
  - 72B parameters

#### phi

#### Hardware Advancements
- **Gaudi3** (Intel)
  - AI accelerator for deep learning
  - Alternative to NVIDIA for training

- **Tensor Streaming Processors** (Cerebras)
  - Memory-centric architecture
  - Optimized for LLM workloads

#### Training Methodologies
- **SSL-RL (Self-Supervised Learning with Reinforcement)** (Google)
  - Combined approach for more efficient training
  - Reduced need for human labels

- **iPOPE (Iterative Pairwise Online Preference Elicitation)** (Apple)
  - Advanced alignment technique
  - Efficient preference learning

#### Benchmark Developments
- **ARC-AGI** (DeepMind)
  - Advanced Reasoning Challenge
  - Complex problem-solving assessment

- **Frontier Math Benchmark** (Various)
  - Advanced mathematical reasoning evaluation
  - Complex mathematical problem-solving


## Technical Deep Dives

### Training data over the years

### Optimization Breakthroughs


### Architectural Breakthroughs


### Training Breakthroughs


Consider adding these categories:

Tokenization Evolution: BPE → SentencePiece → Tokenizer efficiency
Training Data Evolution: From BookCorpus to web-scale datasets
Inference Optimization: KV caching, speculative decoding, etc.

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

### Pre-training Objectives

- MLM (Masked Language Modeling)
- CLM (Causal Language Modeling)
- Span corruption
- RTD (Replaced Token Detection)
- Prefix Language Modeling

### Fine-tuning Strategies

- Instruction tuning
- RLHF pipeline
- DPO (Direct Preference Optimization)
- LoRA and QLoRA
- Parameter efficient fine-tuning

### Optimization Techniques

- Adam variants
- Learning rate schedules
- Gradient accumulation
- Mixed precision training
- ZeRO optimizer stages

### Efficiency Innovations

### Model Compression

- Quantization (INT4/8)
- Pruning techniques
- Knowledge distillation
- Low-rank adaptation
- Sparse inference

### Memory Optimization

- Gradient checkpointing
- Activation recomputation
- Memory efficient attention
- Selective activation storage
- CPU offloading

### Evaluation Framework

### Language Understanding

- GLUE and SuperGLUE
- MMLU
- BIG-bench
- TruthfulQA
- GSM8K

### Safety and Alignment

- TruthfulQA
- Anthropic's Constitutional AI eval
- Bias and toxicity metrics
- HONEST framework
- Safety benchmarks

### Reasoning and Capabilities

- HumanEval
- MATH
- BBH (Big Bench Hard)
- HELM framework
- Chain-of-thought evaluation


Visual Elements

Add performance charts showing scaling laws
Include architecture diagrams for key innovations
Create a "family tree" showing model lineage

NOTES TO SELF

- Add a note for hardware, not in the scope of this blog but should not be ignored [DONE]
- Quick note about benchmark, Not hear to explain these but these are the major ones that are used mostly.   -->
