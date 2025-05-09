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

##### Pretraining

**Step 1: Data Collection**

We know Decoder based LLMs are able to answer a variety of answer to questions about science, food, mathematics, facts etc. That is because they have been trained on a huge plethora of the data. 

So the first step is to collect the said data. One important thing to keep in mind is that these models are able to answer based on what they have seen. They are not performing any kind of reasoning. It is given a distribution, what is the most likely token that is to appear. 

So if you want a coding LLM you will collect a lot of code related data from publically available places like github. 

If you want a cooking LLM you will collected a lot of recipes and so on. 

Most general purpose LLMs are trained on data collected from various sources, hence they are able to answer a lot of question. 

A lot of filtering steps also goes behind it

"
Garbage in, garbage out
"

Most of the internet when crawled has data which looks something like this 

```html
<html>
{add shit}
</html>
```

Hence it needs to be processed into a more human readable form, You can imagine how humongous of a task it must be to clean huge datasets with GBs of data. 

Now there are other filterning that also needs to be done, How do you take care of profanity? What about fake news and so on

**Step 2: Tokenization**

{maybe talk about vocabulary and token length}

When we talked about transformers, we skipped talking about tokenization, but as it is a vital piece of LLM training. We shall spend some time talking about it here. 

Sentence level tokenization 

My first question was, why do we need to break down words. Why not give the entire sentence. Heck why not give the whole paragraph as an input. 

While in practice it can be done, it is not wise. Because if we go back to our basic principle about LLMs. 

"They are next token predictors"

So in given any large enough paragraph or even an essay. The likelihood of a sentence repeating is very low. So if we think it in terms of machines, if we transform each sentence into a token, we will have a vocabular with a lot of numbers, which do not relate to each other at all. So we can never predict what sentence will come after any given sentence. 

Word level tokenization 

A simple work around this seems to be, well why not just tokenize the words, instead of sentences. Because in any large enough paragraph or essay. Words repeat and they follow a logical sequence of what is to appear next. For example 

"Water is ___", if you gave me this sentence word by word, I will assume the next word is wet. Whereas if you gave me a sentence 

"Water is wet, This is a debatable topic." I will have no clue what can be said after this sentence, Maybe someone raises a point, maybe someone says something else. 

So word level helps us retain the logical sequence, and words have meanings to them too. But there is still one big issue. There can be millions of words, some have way higher representation in usual text and some are highly unlikely to occur in common place. 

There are words which are commonplace in one industry and rare in another. 

So we will have a huge vocabulary. 

If you think for a moment we may come to the conclusion that, why not use character level tokenization to solve this problem, this will reduce the vocabulary drastically. 

Here the problem would lie in the fact that characters by themselves do not hold much meaning (atleast in the english lexicon)

**Step 3: Training the network**

##### Fine-Tuning

#### Encoder Only

Now let's understand how a usual Encoder is trained, We will talk about BERT here 

#### Encoder Decoder

Now let's do the same for T5

### Inference



## The AI timeline

This is a very short timeline of the most influential work, to read about more architectures that were huge at the time but died down eventually, consider going through the [Transformer catalog](https://docs.google.com/spreadsheets/d/1ltyrAB6BL29cOv2fSpNQnnq2vbX8UrHl47d7FkIf6t4/edit?gid=0#gid=0).

The blog ["Transformer models: an introduction and catalog — 2023 Edition"](https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/) helped me immensely while making the timeline.

[Write the name of the creators and labs]

### 2017: The Foundation Year

#### Attention is all you need

[paper](https://arxiv.org/abs/1706.03762)

The foundational paper on transformers is released, some of the key ideas introduced include

- Scaled dot-product attention
- Multi-head attention mechanism
- Positional encodings
- Layer normalization
- Masked attention for autoregressive models

We have talked deeply about each of these topics previously and I implore you to check that part out [here]()

#### Deep reinforcement learning from human preferences

[paper](https://arxiv.org/abs/1706.03741)

The RLHF paper 

#### Proximal Policy Optimization Algorithms

[paper](https://arxiv.org/abs/1707.06347)


### 2018: BERT and Early Innovations

#### GPT-1

[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
[blog](https://openai.com/index/language-unsupervised/)

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

#### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[paper](https://arxiv.org/abs/1810.04805)

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


### 2019: Scaling and Efficiency

#### GPT-2

[blog](https://openai.com/index/better-language-models/)
[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

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


### 2020: The Scale Revolution

#### GPT-3

[paper](https://arxiv.org/abs/2005.14165)

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


#### Sparse Transformers



#### Training Methodologies
- **ZeRO (Zero Redundancy Optimizer)** (Microsoft)
  - Memory optimization for distributed training





#### ELECTRA
 
Google's model that used a discriminative approach instead of masked language modeling, providing more efficient training As noted, "Electra deploys a 'Masked Language Modeling' approach that masks certain words and trains the model to predict them. Additionally, Electra incorporates a 'Discriminator' network that aids in comprehending language without the need to memorize the training data."


#### Switch Transformer

Google's early mixture-of-experts approach that demonstrated trillion-parameter scale was possible


#### Scaling Laws

OpenAI's publication on the mathematical relationships between model size, dataset size, and computational budget demonstrated predictable patterns for improving performance This was part of the GPT-3 research which showed "that scaling up language models greatly improves task-agnostic, few-shot performance."


### 2021: Instruction Tuning and Alignment

#### Dall-e


#### LoRA

[paper](https://arxiv.org/abs/2106.09685)


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


**Gopher (DeepMind)**
- 280B parameter model released in December 2021 DeepMind introduced this model as a "280 billion parameter model" that was "evaluated on 152 diverse tasks, achieving state-of-the-art performance across the majority."
- Demonstrated significant scaling benefits in reading comprehension and fact-checking
- Represented a major advancement in model scale from DeepMind

**Megatron-Turing NLG (Microsoft & NVIDIA)**
- 530B parameter model announced in October 2021
- Combined Microsoft's Turing and NVIDIA's Megatron technologies
- Demonstrated advanced distributed training techniques
- Applied significant hardware optimization for large-scale training



### 2022: Democratization


#### Chinchilla

[paper](https://arxiv.org/abs/2203.15556)
While the Chinchilla model itself wasn't released until 2022, the research behind it began in 2021, establishing important scaling principles that:
- Showed optimal token-to-parameter ratios should be approximately 20:1 This research found that "we need around 20 text tokens per parameter" for optimal training.
- Demonstrated many existing models were significantly undertrained
- Influenced the training methodology of subsequent models

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


#### Chain-of-thought prompting

[paper](https://arxiv.org/abs/2201.11903)



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


#### Flash Attention
#### Rotary embeddings



#### Grouped-query attention
#### ALiBi position encoding


- **Anthropic Claude 1** (Anthropic)
  - Initial release focusing on helpfulness and harmlessness

#### FLAN (Fine-tuned LAnguage Net) (Google)
- Instruction tuning across multiple tasks
- Improved zero-shot performance

#### HELM (Holistic Evaluation of Language Models)** (Stanford)
  Comprehensive benchmark suite for LLMs
  Standardized evaluation metrics

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

#### GPTQ

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
