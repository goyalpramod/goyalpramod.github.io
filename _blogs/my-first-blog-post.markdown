---
layout: blog
title: "My First Blog Post"
date: 2024-03-15 12:00:00 +0530
categories: [personal, technology]
---

# Transformers Laid Out

I have encountered that there are mainly three types of blogs/videos/tutorials talking about transformers

* Explaining how a transformer works (One of the best is [Jay Alammar's blog](https://jalammar.github.io/illustrated-transformer/))
* Explaining the "Attention is all you need" paper ([The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/))
* Coding tranformers in PyTorch ([Coding a ChatGPT Like Transformer From Scratch in PyTorch](https://www.youtube.com/watch?v=C9QSpl5nmrY))

And each follow an amazing pedigogy, Helping one understand a singluar concept from multiple point of views.

But this hindered my own learning process, hence I have created this blog. Which will do the following

<!-- add redirects to each section  -->
* Give an intition of how transformers work
* Explain what each section of the paper means and how you can understand and implement it
* Code it down using PyTorch from a beginners perspective

![Meme](https://imgs.xkcd.com/comics/standards_2x.png)
{add this as a foot note} meme taken from [xkcd](https://xkcd.com/)

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

Let us begin with a single sentence and work from there 

"I like Pizza", first the sentence is broken down into it's respective words* and each word is embedded using an embeddings matrix that is trained along with the transformer. 










*We are using words for easier understanding, most modern LLMs do not work with words. But rather "Tokens"

























P.S All the code as well as assests can be accessed from my github and are free to use and distribute, Consider citing this work though :)