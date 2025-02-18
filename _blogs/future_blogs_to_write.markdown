<!-- ---
layout: blog
title: Future blogs to write
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: [add image]
---

- Evolution of LLMs

  - [Flash Attention Blog](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
  - [Flash Attention 3](https://tridao.me/blog/2024/flash3/)
  - [Mamba](https://tridao.me/blog/)
  - [Paper on fine-tuning](https://arxiv.org/abs/2408.13296v1)
  - [Scaling laws](https://arxiv.org/pdf/2001.08361)
  - [Fine-tune lora on CPU](https://rentry.org/cpu-lora)
  - [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
  - [LLMs in 5 formulas](https://www.youtube.com/watch?v=KCXDr-UOb9A)
  - [LLM from scratch](https://bclarkson-code.com/posts/llm-from-scratch-scalar-autograd/post.html)
  - [The transformer family v2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
  - [Llama from scratch](https://blog.briankitano.com/llama-from-scratch/)
  - [Understanding large language models](https://magazine.sebastianraschka.com/p/understanding-large-language-models)
  - [GPT in 60 lines](https://jaykmody.com/blog/gpt-from-scratch)
  - [LLM Playbook](https://cyrilzakka.github.io/llm-playbook/index.html)
  - [Harvard slides on transformers to LLM](https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/transformers)
  - [Timeline and family tree](https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/) [DONE]
  - [A Brief History of Large Language Models](https://medium.com/@bradneysmith/98a1320e7650) [DONE]

{I believe the below can be clubbed together in one blog called. A very technical deep dive into LLMs, Section being, CUDA, training, inference}

- CUDA & optimising GPUs

  - [Understanding triton](https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c)
  - [Triton Documentation](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)
  - [Reddit post on triton](https://www.reddit.com/r/OpenAI/comments/18nf310/openai_triton_coursetutorial_recommendations/)
  - [Daniels video on GPU mode](https://www.youtube.com/watch?v=hfb_AIhDYnA&ab_channel=GPUMODE)
  - [Blog series on CUDA by maharish](https://maharshi.bearblog.dev/blog/)
  - [Beating cuBLASS blog](https://salykova.github.io/)
  - [Lectures on CUDA and GPU stuff](https://www.youtube.com/@GPUMODE/videos)
  - [Cuda tutorial videos](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb)
  - [GPU puzzles](https://github.com/srush/GPU-Puzzles)

- Superfast inference with vLLMs, triton etc/ quantization
  
  - [Fast lora implementation](https://github.dev/unslothai/unsloth)
  - [Code for different optimization from unsloth](https://github.com/unslothai/unsloth/tree/main/unsloth/kernels)
  - [4 bit flux](https://github.com/HighCWu/flux-4bit/blob/main/model.py)
  - [Quantized flux inference](https://gist.github.com/sayakpaul/05afd428bc089b47af7c016e42004527)
  - [unsloth wiki](https://github.com/unslothai/unsloth/wiki)
  - [Blog on qlora, bitsandbyte etc](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
  - [tips ons fine tuning qlora by sebastian](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
  - [medium article on fine tuning](https://medium.com/@levxn/lora-and-qlora-effective-methods-to-fine-tune-your-llms-in-detail-6e56a2a13f3c)
  - [Medium article on fine tuning](https://medium.com/@dillipprasad60/qlora-explained-a-deep-dive-into-parametric-efficient-fine-tuning-in-large-language-models-llms-c1a4794b1766)
  - [Floating point values](https://huggingface.co/blog/hf-bitsandbytes-integration)
  - [Quanto flux inference](https://huggingface.co/blog/quanto-diffusers)
  - [TorchAO](https://huggingface.co/docs/transformers/main/en/quantization/torchao)
  - [QALora](https://arxiv.org/abs/2309.14717)
  - [Quantization](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)
  - [Lora](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
  - [Pytorch guide on performance tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  - [Efficient training guide by HF](https://huggingface.co/docs/transformers/en/perf_train_gpu_one)
  - [Reduce memory usage hf guide](https://huggingface.co/docs/diffusers/main/en/optimization/memory)
  - [Performance and scalability by hf](https://huggingface.co/docs/transformers/v4.19.4/en/performance)
  - [Optimization for llama](https://atscaleconference.com/videos/faster-than-fast-networking-and-communication-optimizations-for-llama-3/)
  - [TorchTitan](https://github.com/pytorch/torchtitan)
  - [Technical conferences on sharded training ](https://www.youtube.com/@scaleconference/videos)
  - [HF blog on quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- Guide for distributed training and training multiple GPUs

  - [Distributed inference](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#model-sharding)
  - [Meta blog on sharded training](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
  - [Making deep learning go brrrr](https://horace.io/brrr_intro.html)
  - [A guide on good usage of non_blocking and pin_memory() in PyTorch](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
  - [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)

- A guide to hacking LLMs

  - [Pliny the liberator](https://x.com/elder_plinius/highlights) -> I think I can only reverse engineer his tweets, no other option.
  - [Blog on hacking LLMS](https://yourgpt.ai/blog/general/how-to-hack-large-language-models-llm)
  - [SOme blog](https://www.siam.org/publications/siam-news/articles/how-to-exploit-large-language-models-for-good-or-bad/)
  - [Another blog](https://www.comet.com/site/blog/prompt-hacking-of-large-language-models/)
  - [A youtube video on the topic](https://www.youtube.com/watch?v=6bYGhY9HB8k)

- Building a 2B model from scratch

  - pytorch docs, hf and umar jamil

- Building a vision model

  - [HF blog on building vlm from scratch](https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model)
  - [Meta's intro to VLM modelling](https://ai.meta.com/research/publications/an-introduction-to-vision-language-modeling/)
  - [Medium article on building a VLM](https://medium.com/@uthmant14/building-a-custom-vision-language-model-with-hugging-face-7d51099bfbcb)
  - [Another one](https://medium.com/@achrafabbaoui/implementation-of-vision-language-models-vlm-from-scratch-a-comprehensive-technical-deep-dive-d348322f9b3c)
  - [Paper on building a VLM](https://arxiv.org/abs/2408.12637)
  - [Encode blog on VLMs](https://encord.com/blog/vision-language-models-guide/#:~:text=Building%20VLMs%20involves%20pre%2Dtraining,for%20more%20specific%20downstream%20tasks.)

- How do video gen models work

  - [lil log blog on the topic](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
  -

- unsupervised learning and RL, the dark horse of ML

  - [RL training of gaming agents](https://diamond-wm.github.io/)

- An MLE's guide to WEB DEVELOPMENT

IDK read to figure out

- [The paradox of diffusion distillation](https://sander.ai/2024/02/28/paradox.html) -> more blogs on diffusion by this guy
- [Good blogs, some may be useful](https://mccormickml.com/)
- [Timeline for AI](https://nhlocal.github.io/AiTimeline/#2024)
- [All of sebastians blogs](https://magazine.sebastianraschka.com/archive?sort=top)
- [idk](https://jakevdp.github.io/blog/2013/04/15/code-golf-in-python-sudoku/)
- [Build your own x](https://github.com/codecrafters-io/build-your-own-x)
- [Neetcode](https://neetcode.io/)
- [Umar's videos on various topics](https://www.youtube.com/@umarjamilai/videos)
- [Implementation of the most popular papers](https://nn.labml.ai/)
- [karpathy's blogs on various topics](https://karpathy.github.io/2019/04/25/recipe/)

Books reading list and order 

```
This is excellent information! Given your ambitious projects and significant time commitment, let me create a comprehensive reading plan.

Phase 1: Deep Learning & PyTorch Foundations (8-10 weeks)
1. "Understanding Deep Learning" by Prince (4-5 weeks)
   - Focus: Modern implementation + theory
   - Priority chapters: Neural networks, optimization, and advanced architectures
   - Concurrent with PyTorch documentation for implementation

2. "Deep Learning with PyTorch" (2-3 weeks)
   - Use as reference while implementing concepts from Prince's book
   - Focus on custom extensions and advanced features
   - Skip basic tutorials you're already familiar with

3. "Mathematics for Machine Learning" (2 weeks)
   - Focus on optimization chapters
   - Reference while implementing gradient descent variants

Phase 2: LLM Focus (6-8 weeks)
1. "Build a Large Language Model" (3-4 weeks)
   - Core focus for your 2B LLM project
   - Implement while reading

2. "Dive into Deep Learning" (3-4 weeks)
   - Focus on transformer chapters
   - Attention mechanisms
   - Modern architectures

Phase 3: Systems & Optimization (8-10 weeks)
1. Professional CUDA C Programming (4-5 weeks)
   - Essential for kernel optimization
   - Focus on practical exercises
   - Concurrent with small PyTorch kernel implementations

2. "Programming Massively Parallel Processors" (2-3 weeks)
   - Focus on memory hierarchy
   - Optimization techniques
   - Skip redundant CUDA basics

3. "Parallel Programming Concepts" (2 weeks)
   - Reference for distributed training
   - Focus on scaling strategies

Optional/Reference Materials:
- "Alice's Adventures in a differentiable wonderland"
   - Use as supplementary reading for intuition
- Goodfellow's "Deep Learning"
   - Reference for theoretical depth when needed

Project Integration Timeline:
1. Weeks 1-10: Build basic LLM while studying foundations
2. Weeks 11-18: Scale to 2B parameters, integrate VLM components
3. Weeks 19-28: Optimize kernels, implement custom CUDA operations
4. Weeks 29-32: Work on quantization and efficiency (llama.cpp style)

Additional Resources to Consider:
1. HuggingFace documentation
2. PyTorch Internals documentation
3. NVIDIA's CUDA documentation
4. llama.cpp GitHub repository

Would you like me to break down any specific phase in more detail or suggest specific projects to implement while reading these books?
```
 
1. MML
2. DL with PyTorch + Dive into DL (in parallel)
3. Understanding DL
4. Build a LLM
5. Professional C CUDA programming 
6. PMPP
7. PPC
- -->
