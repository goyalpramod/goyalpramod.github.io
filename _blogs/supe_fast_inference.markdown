---
layout: blog
title: "Super Fast Inference"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

## Notes on CUDA #1 

These are my notes as I learn CUDA, here I will try to breakdown things and represent them the way it makes the most sense to me. I like to truly understand things internally, so I will try to explain it enough length to make complete sense of it. 

I believe the best way to go about this is to actually write CUDA kernels ourselves and improve over time. 

I will like these series to be exhaustive enough to bring any newbie upto SOTA level. I will mention blogs/books/videos as I keep learning and growing. 

> Note: This is an adaptive blog and I will keep adding (and sometimes removing) as I gain a better understanding of things. 

Now let us begin with... understanding the hardware. Now hear me out, when it comes to CUDA. Understanding what GPU you have and how it works is equally important as understaning the code. Because these are tightly coupled. 

NOTES: 
1. Mention device and host 
2. Mention how data is allocated and freed using CudaMalloc etc 
3. Define and differentiate between and kernel and so on! 
4. [ADD PART OF COMPUTE AND MEMORY PROBLEM WHEN IT COMES TO GPUs]


### Understanding the GPU 

Now one obvious question arrises is why do we even have GPUs, aren't CPUs enough? Can we not combine them*? Why have a separate module at all?

*interestingly this is exactly what apple did, you can understand more about it [here](https://discussions.apple.com/thread/255191914?sortBy=rank) (I remember watching a beautiful video explaining this in greater depth, I forgot about it. If you know what I am talking about, please reach out!).

This is how a CPU looks like 

![Image of CPU Internal](/assets/blog_assets/supe_fast_inference/notes_on_cuda_1.webp)
(Inspired form [PMPP](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/))

the different parts are 

DRAM -> Dynamic Random Access Memory, this is where data gets stored before computation

CACHE -> Temporary memory space to store on the fly computational values

CONTROL -> Determines where to send the computation, where to store data. It's the control center! 

ALU -> Arithmetic Logic Unit, this is the part that takes care of computation. 

> Note: This is a gross over simpliffication of how CPUs look like (even GPUs when we get to it), this is meant to help you understand the core components and how they work. As we go through the blog we will gradudally break down these high level components to their individual sub parts and understand how they work! 

Now this is great if you want to do things in sequence,i.e one after the other. In CPUs we even have multiple cores so you can run multiple computation in parallel (multi-threading, parallelism ,and async are all different ideas consider [reading](https://stackoverflow.com/questions/27435284/multiprocessing-vs-multithreading-vs-asyncio) this to understand the difference.)

Now imagine a matrix multiplication, the core of most of AI. It is an operation which if you think about can be run in parallel, each output value can be calculated independently of the other output values all you need is the row and column vector for that i and j values. 

![Image of Matrix Multiplication](/assets/blog_assets/supe_fast_inference/notes_on_cuda_7.webp)

And to enable this what would we need differently from the CPU... well it isn't hard to answer more ALUS!!! because we want to compute these values ASAP and that is why a GPU in general looks like this 

![Image of GPU Internal](/assets/blog_assets/supe_fast_inference/notes_on_cuda_2.webp)
(Inspired form [PMPP](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/))

>NOTE: Again, this GPU architecture is an oversimplification. But it is necessary info to get the point across. As we get more advanced, we will add on to our existing knowledge and make the diagrams more complex!

As you can see above, we have way more ALUs. Let's understand them better by looking at what the inidividual parts are called. We will explore them a bit more in detail unlike the CPU section above as this blog is all abount understanding GPUs and CUDA. 

![Memory layout of the internal of a GPU](/assets/blog_assets/supe_fast_inference/notes_on_cuda_5.webp)
Image inspired from this [blog](https://damek.github.io/random/basic-facts-about-gpus/#fn:12)

The most basic fact that we need to understand is that, the higher the memory storage, slower the speed. And vice versa. (I do not completely understand the reason behind it right now, but when I do. I will write it!).

Global Memory or VRAM is the advertised GPU storage, an SM (or streaming multiprocessor) has multiple parts to it like tensor cores, threads, warp scheduler and much more stuff. 

For this current blog, we need not dive that much into it! So we will look at the core ideas for now. The most important thing to understand is that SMs have blocks inside of them, these blocks have threads in them, the threads of a block has access to the shared memory of that block ONLY. 

All threads are arranged in a 32 thread warp! Essentially a warp runs all the threads simultaneously. 
(If this does not make a lot of sense right now, do not worry. As we move forward it will start making more sense!)

![Data transfer from VRAM to SM](/assets/blog_assets/supe_fast_inference/notes_on_cuda_6.webp)

The transfer of data from global memory to an SM is an extremely inefficient operation, [horace he](https://horace.io/) has an amazing blog "[Making GPUs go Brrr](https://horace.io/brrr_intro.html)" that explains it quite well. Check it out. So ideally we would like to take our data, give it to the SM do all the necessary computation there. And only send it back once we are done computing. 

![Data transfer from VRAM to SM](/assets/blog_assets/supe_fast_inference/notes_on_cuda_4.webp)

The above image is a simplification of how an SM looks like. Now we have a good general understanding of why GPUs exist and how they look like! This knowledge will prove to be useful when we start getting really knee deep into this.

### Understanding CUDA 

Now we can start understaning the internals of CUDA itself.

In CUDA we have grids, which have blocks inside of them, and the blocks have threads. They can be laid out in a 3d manner as presented below, but generally everyone just works with a 2d layout so we will use that majority of the time. 

As I find the 1d model easier to understand as a beginner, I will use that in this part. And we will start with the multi-dimentional part starting next blog. 

![Internals of CUDA code](/assets/blog_assets/supe_fast_inference/notes_on_cuda_3.webp)

Now the above diagram can be a lot to take in while starting out, but let's break it down component by component. 

We have a host where we write the CUDA kernel which is run in the device. The host is the CPU, a kernel is essentially a function and the GPU is the device. 

In a kernel we define the number of blocks in a grid, and then the number of threads in each block. 

To traverse the blocks and grids, we have the dimensions and indexes. (Look carefully, you will see both are very different thing. One helps move in a direction the other defines the length of that direction)

### Simple MatMul 

Now let us write simple matrix multiplication in Python or CPU code, then we will write a CUDA kernel using what we have learned so far!

```python
import numpy as np

a = 5
b = 10
c = 5

GEMM_1 = np.random.rand(a, b)
GEMM_2 = np.random.rand(b, c)

ANS_triple = np.zeros((a, c))

for i in range(a):
    for j in range(c):      
        for k in range(b):
            ANS_triple[i, j] += GEMM_1[i, k] * GEMM_2[k, j]

# Both should match numpy's built-in matmul
assert np.allclose(ANS_triple, GEMM_1 @ GEMM_2)
```

The simplest idea that we have to keep in mind while working with cuda is that we have multiple threads, running at once, and we want to get them running simultaneously. 

The worst matmul that you can write is 

```cpp
// A -> M X K 
// B -> K X N
// output -> M X N

__global__ void super_bad_matmul_kernel(const float* A, const float* B, float* output, int M, int N, int K){
   float temp_val = 0;

   for(int i = 0; i<M; i++){
      for(int j = 0; j<N; j++){
         for(int k = 0; k<K; k++){
            temp_val += A[i*K + k]*B[K*k + j];
         }
         output[i*N + j] = temp_val;
      }
   }
}

// Fix the below code
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (N*C*H*W + threadsPerBlock - 1)/threadsPerBlock;
                      }
```

The above code is terrible, the main reason being we are not utilising the fact the the code can be parallalized and multiple threads can each calculate one output value. 

Let's write the naive CUDA solution, and then I will explain what each part does and why it looks the way it does!

```cpp
// A -> M X K 
// B -> K X N
// output -> M X N

__global__ void naive_matmul(const float* A, const float* B, float* output, int M, int N, int K){
   int gid = threadIdx.x + blockDim.x*blockIdx.x;
   if(gid>= M*N) return;

   int row = gid/N;
   int col = gid%N;

   float temp_val = 0;
   for(int i = 0;i<K;i++){
      temp_val += A[row*K + i]*B[i*N + col];
   }

   output[gid] = temp_val;
}
```

The above one although not a great implementation shows the value. There is one idea that we have not talked about so far which sits at the center of CUDA. And that is the actual memory layout of the data is one dimentional and it is stored in a row major form. 

We know that our output will be of the shape, MxN. But in memory we cannot have 2d layouts, we only have 1d. So instead of a MxN layout we have M rows of N values stacked after each other, which looks something like the below image. 

![how row major happens in memory](/assets/blog_assets/supe_fast_inference/notes_on_cuda_8.webp)

(You can imagine the similar for 3d)

Hence, if we breakdown gid (I like to call it global id, tid id thread id and is the number of the thread within a block. Here we have defined it as 256, so that will never exceed 256). It consists of threadIdx, blockDim, and blockIdx. Idx means index and dim means dimension. 

It is important you visualize and understand how this is working; threadIdx gives you the current thread you are on in a block. Multiplying the index of the block (blockIdx) with the dimension tells you how many previous blocks you have completed. 

A good way to understand is by thinking backwards. We want each thread to calculate one value. So we need MxN threads. Which is not possible. 

So we set 256 threads and then we define the number of blocks based on this, that is where the ceiling equation comes from 

`blocksPerGrid = (M*N + threadsPerBLock - 1)/threadsPerBlock;`

this will give us enough blocks with enough threads to take care of this computation, now as this is a ceiling function we can have number of threads which exceed MxN. And hence the reason we have added a check 

`if(gid>= M*N) return;`

read this part a few more times, try to think it in your terms and it should make sense! 

### Where do we go from here? 

If you would like to put the knowledge you have aquired to the test, I will recommend checking out 

* [GPU Puzzles](https://github.com/srush/gpu-puzzles)
* [LeetGPU](https://leetgpu.com/)

<!-- ## NOtes on CUDA #2  

In our last blog we understood how CUDA works and how a general GPU will look like 

Now let us optimize it 

The few things we will learn are 

* Warp divergence 
* Bank conflicts 
* Shared memory 

https://damek.github.io/random/basic-facts-about-gpus/ -> Good overview of GPU -->



<!-- Throughout my circle I am known as the optimization guy, for my ability to squeeze out performance from each aspect of my life. Hence, I felt it was a damn shame I did not know how to do the same with my ML models. 

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
