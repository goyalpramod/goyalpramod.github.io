---
layout: blog
title: "Transformers Laid Out"
date: 2024-03-15 12:00:00 +0530
categories: [personal, technology]
---

# Transformers Laid Out

I have encountered that there are mainly three types of blogs/videos/tutorials talking about transformers

- Explaining how a transformer works (One of the best is [Jay Alammar's blog](https://jalammar.github.io/illustrated-transformer/))
- Explaining the "Attention is all you need" paper ([The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/))
- Coding tranformers in PyTorch ([Coding a ChatGPT Like Transformer From Scratch in PyTorch](https://www.youtube.com/watch?v=C9QSpl5nmrY))

And each follow an amazing pedigogy, Helping one understand a singluar concept from multiple point of views.

But this hindered my own learning process, hence I have created this blog. Which will do the following

<!-- add redirects to each section  -->

- Give an intition of how transformers work
- Explain what each section of the paper means and how you can understand and implement it
- Code it down using PyTorch from a beginners perspective

![Meme](https://imgs.xkcd.com/comics/standards_2x.png)
{add this as a foot note} meme taken from [xkcd](https://xkcd.com/)
<!-- {change this to make there are 14 transformers tutorial} -->

## How to use this blog

First I will give you a quick overview of how the transformer works and why it was developed in the first place.

Once we have a baseline context setup, We will dive into the code itself.

I will mention the section from the paper and the part of the transformer that we will be coding, along with that I will give you a sample code block with hints and links to documentation like the following:

```python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Args:
            optimizer: Optimizer to adjust learning rate for
            d_model: Model dimensionality
            warmup_steps: Number of warmup steps
        """
        # Your code here
        # lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        # {add pytorch documentation links here}

    def step(self, step_num):
        """
        Update learning rate based on step number
        """
        # Your code here - implement the formula

```

I will recommend you copy this code block and try to implement that function by your self.

To make it easier for you, before we start coding I will explain that part in detail, If you are still unable to solve it by yourself, come back and see my code implementation of that specific part.

Subsequently after each completed code block I will keep a FAQ section where I will write down my own questions that I had while writing the transformer as well as some questions that I believe are important to understand the concepts.

## Understanding the Transformer

The original transformers was made for machine translation task and that is what we shall do as well.
We will try to translate from English to Hindi.

Let us begin with a single sentence and work from there

"I like Pizza", first the sentence is broken down into it's respective words\* and each word is embedded using an embeddings matrix that is trained along with the transformer.

Now these positional information is added to these embeddings,
The reason we need to do this is because Transformers take all the information in parallel i.e. at once, so they lose the positional
information which RNN or LSTM capture.

And positional information is important because "I like Pizza" =/= "Pizza like I" (It just gets weirder with longer sentences)

Now these embeddings are passed to an "encoder" block which essentially does two things

- Applies self-attention (about which we will be learning in the next section) to understand the relationship of individual words with respect to the other words present
- {continue}

The decoder block takes the output from the encoder, runs it through it self. Produces an ouput and sends it back to itself to create the next word

think of it like this.

The encoder understands your language let's call it e and another language called z
The decoder understands z and the language you are trying to transfor e to, lets call it d.

So z acts as the common language that both the encoder and decoder speak to produce the final output.

\*We are using words for easier understanding, most modern LLMs do not work with words. But rather "Tokens"

## Understanding Self-attention

We have all heard of the famous trio, "Query, Key and Values". I absolutely lost my head trying to understand how the team came up behind this idea
Was Q,K,Y related to dictionaries? (or maps in traditional CS) Was it inspired by a previous paper? if so how did they come up with?

Let us first build an intuition behind the convention (then get rid of this convention to make more sense of it)
Sentence (S): "Pramod loves pizza"

Questions:

1. Who loves pizza?

You can come up with as many Questions (the queries) for the sentence as you want.
Now for each query, you will have one specific piece of information (the key) that will give you the desired answer (the value)

Query:

1. Q ->Who loves pizza?
   K -> pizza, Pramod, loves (it will actually have all the words with different degree of importance)
   V -> pizza (The value is not directly the answer, but a representation as a matrix of something similar to the answer)

This is an over simplification really, but it helps understand that the queries, keys and values all can be created only using the sentences

Let us simplify this, I will help you understand it the best way it helped me understand.
Forget Q,K,V. Lets just call them matrices for now. m1,m2 and m3.

{here just make a matrix}
m1 -> matrix representing query
pramod: embedding  
loves: embedding
pizza: embedding

m2 -> matrix representing keys
pramod:
loves:
pizza:

m3 -> {I am not sure about this}

Now forget multi-head attention, attention blocks and all the HUGE BIG JARGON.
Lets say you are in point A and want to go to B in a huge city
Do you think there is only one path to go their? of course not, there are thousands of way to reach that point

so a single matrix multiplication will obviously not get you the best representation of query and key
Multiple queries can be made, multiple keys can be done for each of these query
That is the reason we do so many matrix multiplication to try and get the best key for a query that is relevant to the question asked by the user

That is all the reason there is to it. Have a look at the different illustrations to better understand it.

## Understanding The Encoder and Decoder Block

If everything so far has made sense, this is going to be a cake walk for you. Because this is where we put everything together.

A single tranformer can have multiple encoder, as well as decoder blocks.

Let's start with the encoder part first.

Our input sentence is first converted into tokens

Then it is embedded through the embeddings matrix, then the positional encoding is added.

Now all the tokens are processed in parallel, they go through the first encoder block, then the second till the nth(n here being any arbitrary number of blocks defined by you) block

What this tries to do is capture all the semantic meaning between the words, the richness of the sentence, the grammar (originally transformers were created for machine translation. So that can help you understand better)

Then this final output is given to all the decoder blocks as they process the data, the decoder block is auto-regressive. Meaning it outputs one after the other and takes its own output as an input

That is all the high level understanding you need to have, to be able to write a transformer of your own. Now let us look at the paper as well as the code

## Coding the transformer

For the following section I will recommend you have 3 tabs open. This blog, a jupyter notebook
and the [original paper](https://arxiv.org/pdf/1706.03762)

### Abstract & Introduction

This section brings you upto speed about what the paper is about and why it was made in the first place.

There are some concepts that can help you learn new things, [RNNs](https://www.youtube.com/watch?v=AsNTP8Kwu80), [Convolution neural network](https://www.youtube.com/watch?v=HGwBXDKFk9I) and about [BLEU](https://en.wikipedia.org/wiki/BLEU).

Also it is important to know that transformers were originally created for text to text translation. I.E from one language to another.

Hence they have an encoder section and a decoder section. They pass around information and it is known as cross attention (more on the difference between self-attention and cross attention later)

### Background

This section usually talks about the work done previously in the field, known issues and what people have used to fix them.
One very important thing for us to understand to keep in mind is.

"Keeping track of distant information". Transformers are amazing for multitude of reasons but one key one is that they can remember distant relations.

Solutions like RNNs and LSTMs lose the contextual meaning as the sentence gets longer. But transformers do not run into such problem. (A problem tho, hopefully none existent when you read it is. The context window length. This fixes how much information the transformer can see)

### Model Architecture

The section all of us had been waiting for. I will divert a bit from the paper here. Because I find it easier to follow the data.

{here make the names clickable to the section}
We will first start with the Multi-Head Attention, then the feed forward network, followed by the positional encoding, Using these we will finish the Encoder Layer, subsequently we will move to the Decoder Layer, After which we will write the Encoder & Decoder block, and finally end it with writing the training loop for an entire Transformer on real world data.

The full notebook can be found [here](https://github.com/goyalpramod/transformer_from_scratch/blob/main/transformers.ipynb)

![Image of a transformer](/assets/transformers.svg)

Necessary imports

```python
import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax
```

#### Multi-Head Attention

By now you should have good grasp of how attention works, so let us first start with coding the scaled dot-product attention (as MHA is basically multiple scaled dot-product stacked together). Reference section is 3.2.1 Scaled Dot-Product Attention

{add a copy button for these code blocks}

```python
# try to finish this function on your own
def scaled_dot_product_attention(query, key, value, mask=None):
      """
      Args:
          query: (batch_size, num_heads, seq_len_q, d_k)
          key: (batch_size, num_heads, seq_len_k, d_k)
          value: (batch_size, num_heads, seq_len_v, d_v)
          mask: Optional mask to prevent attention to certain positions
      """
      # get the size of d_k using the query or the key

      # calculate the attention score using the formula given. Be vary of the dimension of Q and K. And what you need to transpose to achieve the desired results.

      #YOUR CODE HERE

      # hint 1: batch_size and num_heads should not change
      # hint 2: nXm @ mXn -> nXn, but you cannot do nXm @ nXm, the right dimension of the left matrix should match the left dimension of the right matrix. The easy way I visualize it is as, who face each other must be same

      # add inf is a mask is given, This is used for the decoder layer. You can use help for this if you want to. I did!!
      #YOUR CODE HERE


      # get the attention weights by taking a softmax on the scores, again be wary of the dimensions. You do not want to take softmax of batch_size or num_heads. Only of the values. How can you do that?
      #YOUR CODE HERE

      # return the attention by multiplying the attention weights with the Value (V)
      #YOUR CODE HERE

```

Some helpful documentation (I will add these after the code block that you are supposed to write, but I will recommend you do your own research first. That is the first steps to becoming a [cracked engineer](https://news.ycombinator.com/item?id=41848448))

- [Tensor size](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html)
- [Matrix multiplication](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [Masked fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_)

```python
# my implementation
def scaled_dot_product_attention(query, key, value, mask=None):
      """
      Args:
          query: (batch_size, num_heads, seq_len_q, d_k)
          key: (batch_size, num_heads, seq_len_k, d_k)
          value: (batch_size, num_heads, seq_len_v, d_v)
          mask: Optional mask to prevent attention to certain positions
      """
      # Shape checks
      assert query.dim() == 4, f"Query should be 4-dim but got {query.dim()}-dim"
      assert key.size(-1) == query.size(-1), "Key and query depth must be equal"
      assert key.size(-2) == value.size(-2), "Key and value sequence length must be equal"

      d_k = query.size(-1)

      # Attention scores
      scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

      if mask is not None:
          scores = scores.masked_fill(mask == 0, float('-inf'))

      attention_weights = softmax(scores, dim=-1)

      return torch.matmul(attention_weights, value)
```

Using this let us complete the MHA, Section 3.2.2

```python
class MultiHeadAttention(nn.Module):
    #Let me write the initializer just for this class, so you get an idea of how it needs to be done
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" #think why?

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Note: use integer division //

        # Create the learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model) #think why we are doing from d_model -> d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None):
      #YOUR IMPLEMENTATION HERE

    def forward(self, query, key, value, mask=None):
      #get batch_size and sequence length
      #YOUR CODE HERE

      # 1. Linear projections
      #YOUR CODE HERE

      # 2. Split into heads
      #YOUR CODE HERE

      # 3. Apply attention
      #YOUR CODE HERE

      # 4. Concatenate heads
      #YOUR CODE HERE

      # 5. Final projection
      #YOUR CODE HERE
```
* I had a hard time understanding the difference between view and transpose. These 2 links should help you out, [When to use view,transpose & permute](https://www.reddit.com/r/learnmachinelearning/comments/17irzkc/why_do_we_use_view_and_then_transpose_when/) and [Difference between view & transpose](https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916)
* Contiguous and view, still eluded me. Till I read these, [Pytorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) and [Contiguous & Non-Contiguous Tensor](https://medium.com/analytics-vidhya/pytorch-contiguous-vs-non-contiguous-tensor-view-understanding-view-reshape-73e10cdfa0dd)
* [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

```python
#my implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Note: use integer division //

        # Create the learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None):
      """
      Args:
          query: (batch_size, num_heads, seq_len_q, d_k)
          key: (batch_size, num_heads, seq_len_k, d_k)
          value: (batch_size, num_heads, seq_len_v, d_v)
          mask: Optional mask to prevent attention to certain positions
      """
      # Shape checks
      assert query.dim() == 4, f"Query should be 4-dim but got {query.dim()}-dim"
      assert key.size(-1) == query.size(-1), "Key and query depth must be equal"
      assert key.size(-2) == value.size(-2), "Key and value sequence length must be equal"

      d_k = query.size(-1)

      # Attention scores
      scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

      if mask is not None:
          scores = scores.masked_fill(mask == 0, float('-inf'))

      attention_weights = softmax(scores, dim=-1)

      return torch.matmul(attention_weights, value)

    def forward(self, query, key, value, mask=None):
      batch_size = query.size(0)
      seq_len = query.size(1)

      # 1. Linear projections
      Q = self.W_q(query)  # (batch_size, seq_len, d_model)
      K = self.W_k(key)
      V = self.W_v(value)

      # 2. Split into heads
      Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
      K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
      V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

      # 3. Apply attention
      output = self.scaled_dot_product_attention(Q, K, V, mask)

      # 4. Concatenate heads
      output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

      # 5. Final projection
      return self.W_o(output)
```

#### Feed Forward Network

Section 3.3

```python
class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network

    Args:
        d_model: input/output dimension
        d_ff: hidden dimension
        dropout: dropout rate (default=0.1)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        #create a sequential ff model as mentioned in section 3.3

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        #YOUR CODE HERE
```

* [Dropout](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
* [Where to put Dropout](https://stackoverflow.com/questions/46841362/where-dropout-should-be-inserted-fully-connected-layer-convolutional-layer)
* [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)

```python
#my implementation
class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network

    Args:
        d_model: input/output dimension
        d_ff: hidden dimension
        dropout: dropout rate (default=0.1)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.model(x)
```

#### Positional Encoding

Section 3.5 

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()

        # Create matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # Create position vector
        position = torch.arange(0, max_seq_length).unsqueeze(1) # Shape: (max_seq_length, 1)

        # Create division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]  # Add positional encoding up to sequence length
```

```python 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()

        # Create matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # Create position vector
        position = torch.arange(0, max_seq_length).unsqueeze(1) # Shape: (max_seq_length, 1)

        # Create division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]  # Add positional encoding up to sequence length
```












## Misc 
Here are some resources and more information that can help you out in your journey which I could not decide where to put 

[What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)









Congratulations for completing this tutorial/lesson/blog however you see it. It is by nature of human curosity that you must have a few questions now. 
Feel free to create issues in github for those questions, and I will add any questions that I feel most beginners would have here in an FAQ section. 

Cheers, 
Pramod 


P.S All the code as well as assets can be accessed from my github and are free to use and distribute, Consider citing this work though :)
