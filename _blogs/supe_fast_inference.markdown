<!-- ---
layout: blog
title: "Super Fast Inference"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

Throughout my circle I am known as the optimization guy, for my ability to squeeze out performance from each aspect of my life. Hence, I felt it was a damn shame I did not know how to do the same with my ML models. 

And hence we start out on this journey together to learn inference engine optimization. We will start by optimizing a GEMM (General Matrix Multiplication) as that is the backbone of everything ML/AI. Then we will move to Diffusion models, first image then video. (I believe there are plenty resources for text i.e LLMs, I will link them in the references for you to check out!)

Let us begin our endevour! 

## GEMM 

I will go with the assumption that you are familiar with your Linear Algebra and move forward. 
Let us first write a for loop that multiplies two 2d matrices of size aXb and bXc

The first is an extremely rudimentary 

```python
import numpy as np

a = 5
b = 10
c = 5

GEMM_1 = np.random.rand(a, b)
GEMM_2 = np.random.rand(b, c)

# 3-loop version: scalar multiply-accumulate, exposes every memory access
ANS_triple = np.zeros((a, c))

for i in range(a):
    for j in range(c):        #notice we iterate over c here
        for k in range(b):
            ANS_triple[i, j] += GEMM_1[i, k] * GEMM_2[k, j]

# Both should match numpy's built-in matmul
assert np.allclose(ANS_triple, GEMM_1 @ GEMM_2)
```



```python
# 2-loop version: vectorized dot product per output element
ANS_double = []

for i in range(a):
    temp_arr = []
    for j in range(c):
        temp_arr.append(np.sum(GEMM_1[i, :] * GEMM_2[:, j]))
    ANS_double.append(temp_arr)

ANS_double = np.array(ANS)  # shape (a, c)
assert np.allclose(ANS_double, GEMM_1 @ GEMM_2)
```

We can optimize it further by calling the internal numpy matmul 

```python
ANS = GEMM_1@GEMM_2
assert np.allclose(ANS, GEMM_1 @ GEMM_2)
```

This is the extent of optimization most people will go with, but we are not most people. Let's go further down the rabit hole and optimize this further! 

(Some might say we skipped over the internals and just used numpy implementations, those some will be correct. To understand more of the internal optimization I recommend checking out these excellent posts [Blog 1](https://siboehm.com/articles/22/Fast-MMM-on-CPU) & [Blog 2](https://salykova.github.io/gemm-cpu))

> Aside: If you would like to understand how numpy shapes work better I will recommend reading this [blog series](https://ajcr.net/stride-guide-part-1/), this knowlege will prove to be essential as we will move forward!

**Blog 3: Super Fast Inference** *(VIRGIL-relevant)*

1. Hook: Start with a GEMM
   - Naive triple loop → optimized → why this matters for everything that follows

2. How to profile inference — measure before you optimize
   - torch.profiler, nsys, memory_profiler
   - https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
   - https://pytorch.org/docs/stable/torch.cuda.html#memory-management

3. The fundamental bottleneck — memory bandwidth vs compute
   - This is the mental model for everything else
   - https://horace.io/brrr_intro.html ← read this first, it's the best thing on this list

4. dtypes in practice
   - fp32 → bf16 → fp16 → fp8 → int8 → int4, with actual VRAM measurements
   - https://huggingface.co/docs/diffusers/optimization/fp16
   - https://pytorch.org/docs/stable/amp.html

5. Quantization
   - bitsandbytes: https://huggingface.co/docs/bitsandbytes
   - torchao (what HF is pushing for Flux): https://github.com/pytorch/ao
   - GGUF for diffusion models: https://github.com/city96/ComfyUI-GGUF

6. Attention optimization
   - Why vanilla attention is memory bandwidth bound (derive the complexity)
   - Flash Attention — read intro + section 2: https://arxiv.org/abs/2205.14135
   - flash-attn repo: https://github.com/Dao-AILab/flash-attention
   - xformers: https://github.com/facebookresearch/xformers

7. torch.compile
   - Graph capture model, what causes graph breaks, how to debug them
   - https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
   - https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

8. CPU offloading & model sharding
   - What `enable_model_cpu_offload()` actually does under the hood
   - https://huggingface.co/docs/diffusers/optimization/memory

9. Triton — write your first custom kernel
   - https://triton-lang.org/main/getting-started/tutorials/ (do all in order)
   - https://github.com/linkedin/Liger-Kernel (production kernels to read)

10. CUDA — going deeper
    - https://siboehm.com/ (SGEMM post specifically)
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (ch 1-4)
    - https://www.youtube.com/@cudamode

---

Blog 3 is the one to write first — it's directly VIRGIL-relevant and has the tightest narrative arc (GEMM → profiling → bottleneck theory → practical optimizations → kernel writing). The other two are supporting material you'll have naturally by the time you're done.

## CUDA 

Lets first start by understanding how a GPU usually looks like and what it's components are 

What is important to understand is, There are grids, grids have blocks inside of them and blocks have threads. 

the row moves along y axis and column moves along x axis 

It is stored in row major format 

If we write a simple kernel (A gpu function) for matrix multiplication it will look something like this 

```cpp
__global__ void simple_matmul(const float* X, const float* y,float* output, int M, int N, int K){
   int gid = threadIdx.x + blockDim.x*blockIdx.x; //-> Important to understand 
 
   int temp = 0

   for(int i = 0; i<K; i++){
      temp += X[gid*N + i]*y[i*N + gid]; // -> a good heuristic to remember is (row*width + col)
   }

   if(gid<M*N){ // -> Very important, because threads can be more than...
      output[gid] = temp;
   }
}
```

This may seem complex initially but once you start working with it. It starts getting easier to make sense. 

Now lets look at how a GPU is to understand the problem with our kernel and how we can optimize it 

The first problem is that when a thread calculates the matrix multiplication it has to do a value look up again and again from the DRAM which is inefficient. It would be much better if it was present inside the block shared memory. The look up is significantly faster in that. 

And it is quite complex to work with a 2d matrix and representing it in 1d. So we will make our representation 2d too using dim3 

Now the optimized kernel will look something like this 

```cpp
#define TILE_SIZE 16

__global__ void tiled_matmul(const float* X, const float* y, float* output, int M, int N, int K){
   int row = blockIdx.y*TILE_SIZE;
   int col = blockIdx.x*TILE_SIZE;

   int temp = 0;

   __shared__ A_TILE[TILE_SIZE][TILE_SIZE];
   __shared__ B_TILE[TILE_SIZE][TILE_SIZE];

   int numTiles = (K + TILE_SIZE - 1)/TILE_SIZE;

   for(int t = 0;t<numTiles;t++){
      //DANG FORGOT THE tile element. FIX IT!
      if(row < M&& col <K){
         A_TILE[row][col] = A[];
      }else{
         A_TILE[row][col] = 0;
      }if(row < K && col <N){
         B_TILE[row][col] = B[];
      }else{
         B_TILE[row][col] = 0;
      }

      __syncthreads();

      for(int kk = 0;kk<K;k++){
         temp += A_TILE[][]*B[][];
      }

      __syncthreads();
   }

   if(){
      output[] = temp
   }

}
```

This is pretty good, but it has problems too. The main problem being that each thread is only doing computation for one output. It would be more efficient if it did it for more than one 

Why you ask? 

This is why... 

Ok, now lets code that out! 

```cpp

```

Okay that was good, if you understand everything we did so far. YOU ARE AMAZING, but if you didnt. Its okay, Even reaching this point took me quite some time.  -->