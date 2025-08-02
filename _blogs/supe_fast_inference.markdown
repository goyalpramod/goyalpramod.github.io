<!-- ---
layout: blog
title: "Super Fast Inference"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

[Start with a GEMM, solve it using for loops and make it as efficient as possible]

[medium](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)

https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

https://zeux.io/2024/03/15/llm-inference-sol/?s=08

https://github.com/mlvu/worksheets/blob/master/Worksheet%205%2C%20Pytorch.ipynb

https://pytorch.org/blog/pytorch-vllm-%E2%99%A5%EF%B8%8F/

A few month's ago, I saw this [tweet](https://x.com/danielhanchen/status/1891194528931209644) by Daniel Han and it absolutely blew my mind. Not because of how much they were willing to offer, but because I couldn't solve any of the problems. That is when I decided, I will spend days and nights till 1 day I can confidently say I can solve each of those problems, and with ease. This Blog was originally me trying to solve them. But it evolved into more of a general guide into how to make your ML models more efficient. So we will role with that. 


I didn't want to limit this blog to any particular "library" or "framework". But we all must have one medium of communication that we understand. Hence I am going forward with PyTorch as the library of choice, for the following reasons: 

- reason 1 
- reason 2 
- reason 3 

## Pytorch


### Tensors 

Understanding Tensor Memory Management in PyTorch

In PyTorch, tensors might look multi-dimensional, but they're actually stored sequentially in memory. Each element occupies a fixed space (4 bytes for integers), and PyTorch uses a clever indexing system with strides to access specific elements.

To access a particular index it uses a formula like the following:
index_1*stride_1 + index_2*stride_2 + ... index_n*stride_n = {location of the data in storage}

Here's what's fascinating:

When you use .view(), you're creating a new way to look at the same data - without actually moving anything in memory.
The .stride() method reveals how PyTorch jumps through memory to access elements.
Operations like .transpose() physically reorganize data in memory, potentially making it non-contiguous.

Understanding the difference between .view() and .reshape() is crucial:

* .view() only works with contiguous tensors
* .reshape() works with both, but creates a copy for non-contiguous data
* .transpose()
* unsqueeze

https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch -> 2ndanswer 

https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch

Changes in a .view() tensor reflect in the original, making it memory-efficient

This knowledge is essential for optimizing deep learning models and understanding memory management in PyTorch.

https://www.linkedin.com/posts/goyalpramod_memory-allocation-in-python-activity-7273940017410928640-FZji?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADbSv4QB6z8hG-KISdXHiYSLLfD-84W0wuQ


https://medium.com/analytics-vidhya/pytorch-contiguous-vs-non-contiguous-tensor-view-understanding-view-reshape-73e10cdfa0dd
https://blog.ezyang.com/2019/05/pytorch-internals/


### AutoGrad

https://www.linkedin.com/posts/goyalpramod_autograd-mechanics-activity-7277299201581981696-yVmr?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADbSv4QB6z8hG-KISdXHiYSLLfD-84W0wuQ



Understanding PyTorch Autograd: A Deep Dive into Automatic Differentiation

As machine learning practitioners, understanding how neural networks learn is crucial. At the heart of this learning process lies PyTorch's Autograd system - a powerful implementation of reverse-mode automatic differentiation.

Let me break down how Autograd works:

At its core, Autograd is an automatic differentiation system that computes gradients by applying the chain rule in reverse order through a computation graph. When we create a tensor with requires_grad=True, we essentially tell PyTorch to track all operations performed on this tensor.
Here's what happens under the hood:

PyTorch constructs a computational graph for each operation
When we call .backward(), the multiplication function retrieves the context from this graph

The next_functions in the graph represent tensor connections:
AccumulateGrad points to tensor 'a'
None points to tensor 'b' (when requires_grad=False)

Let's look at a practical example:
For a simple computation where c = a * b:

dc/dc = 1 (derivative of a value with respect to itself)
dc/da = 3 (as dc/da = da/dab = 1*3 = 3)
If tensor 'b' had requires_grad=True, dc/db would be 2 (as dc/db = adb/db = 2*1 = 2)

The beauty of Autograd lies in its ability to handle complex computational graphs while maintaining computational efficiency. All gradients are automatically stored in the computation graph when requires_grad is enabled, making backpropagation seamless.

http://youtube.com/watch?v=MswxJw-8PvE

### Computation Graph
https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/
https://weiliu2k.github.io/CITS4012/pytorch/computational_graph.html
https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
https://huggingface.co/blog/andmholm/what-is-automatic-differentiation


https://jingnanshi.com/blog/autodiff.html
https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

### Forward Mode Automatic Differentiation 
https://liqimai.github.io/blog/Forward-Automatic-Differentiation/



### Reverse Mode Automatic Differentiation 

### Broadcasting


### Dispatcher

https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/

### Torch.Compile
https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

### Bottleneck 

https://horace.io/brrr_intro.html

## JIT 

## CUDA


## Triton

https://huggingface.co/docs/diffusers/en/optimization/fp16


## Einsum

https://rockt.ai/2018/04/30/einsum
https://eli.thegreenplace.net/2025/understanding-numpys-einsum/
https://ajcr.net/Basic-guide-to-einsum/
https://ejenner.com/post/einsum/
https://theaisummer.com/einsum-attention/

What is nn.Module? 
Why do we always do super()__init__() whenever we start a new class in PyTorch

## The Questions, with their answers

1. Convert nf4 / BnB 4bit to Triton
2. Make FSDP2 work with QLoRA
3. Remove graph breaks in torch.compile
4. Help solve Unsloth issues!
5. Memory Efficient Backprop

 -->
