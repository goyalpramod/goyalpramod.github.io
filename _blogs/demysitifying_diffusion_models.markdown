<!-- ---
layout: blog
title: "Demystifying Diffusion Models"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---
Things left to do: 
-> Adding code for SD components 
-> explaining code 
-> Unifying the maths and explaining for reverse process
-> Maths for score based modeling 
-> Maths for VAE 
-> Appendix 
-> Explining The VAE 
-> Explaining CLIP/T5 text encoders 
-> Explaining Conditioning like CFG etc
-> Re-reading and fixing any mistakes I spot, and adding images where I see fit


{The order of things,

- Build the idea first, general stuff of what is going on.
- Build and code each component out
- Talk about the maths behind the whole idea, starting with score models, then moving to ddpm etc}

Diffusion models like Stable Diffusion, Flux, Dall-e etc were an enigma built upon multiple ideas and mathematical breakthroughs. So is the nature of it that most tutorials on the topic are extremely complicated or even when simplied talk a lot about it from a high level perspective.

There is a missing bridge between the beautiful simplification and more low level complex idea. That is the gap I have tried to fix in this blog.

- Starting with the simple **idea** behind diffusion models
- Understanding each component and **coding** it out
- And finally a full section dedicated to the **maths** for the curious minds

## The Genius Artist

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/1.webp)
Imagine you have a super special artist, whom you tell your ideas and he instantly generates amazing images out of it. Let's name him Dali

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/2.webp)
The way Dali starts his work is, that he first has a canvas, he listens to your instructions then creates an artwork.

But Dali has a big problem, that he cannot make big images, he tells you that he will only create images the size of your hand. This is obviously not desirable. As for practical purposes you may want images the size of a wall, or a poster etc.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/8.webp)
That is when a magic wand falls from the sky, and it has two modes Encoder(Compress size) and Decoder(Enlarge size). That gives you a great idea. You will start with the size of the canvas that you like, Encode it. Give the encoded canvas to Dali, he will make his art, And then you can decode the created art to get it back to the original shape you want.

This works and you are really happy.

But you are curious about how Dali works, so you ask him. "Dali why do you always start with this noisy canvas instead of pure white canvas? and how did you learn to generate so many amazing images?"

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/4.webp)
Dali is a kind nice guy, so he tells you about how he started out. When he was just a newbie artist. The world was filled with great art. Art so complex that I could not reproduce it, nobody could.

That is when I found a special wand as well, which let me add and fix mistakes in a painting.
![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/7.webp)

I would start with an image, add a bunch of mistakes to it, and using my wand I would reverse them.

After a while, I added so many mistakes to the original image, that they looked like pure noise. The way my canvas do, and using my special wand. I just gradually found mistakes and removed them. Till I got back the original image.

This idea sounds fascinating, but you being you have quite a question "that sounds amazing, so did you learn what the full of mistakes image will look like for all the images in the world? Otherwise how do you know what will be the final image be from a noisy image?"

[INSERT_IMAGE]
"Great question!!!" Dali responds. "That is what my brothers used to do, They tried to learn the representation of all the images in the world and failed. What I did differently was, instead of learning all the images. I learnt the general idea of different images. For example, instead of learning all the faces. I learnt how do human faces look in general"

Satisfied with his answers you were about to leave, when Dali stops you and asks, "Say friend, that wand of yours truly is magical. It can make my art popular worldwide because everyone can create something of value using it. Will you be kind enough to explain how it works so I can make one for myself."

You really want to help Dali out, but unfortunately even you do not know how the wand works, as you are about to break the news to him. You are interrupted by a noise, "Gentlemen you wouldn't happen to have seen a magic wand around now would you? It is an artifact created with great toil and time"

[INSERT_IMAGE]
You being the kind soul you are, Tell the man that you found it on the street and wish to return it.
The man Greatly happy with your generosity, wishes to pay you back. You just say "Thank you, but I do not seek money. But it would really help my friend Dali out if you could explain how your magic wand works."

The man curious for what use anyone would have for his magic wand sees around Dali's studio, and understands that he is a great artist. Happy to help him he says. "My name is Auto, and I shall tell you about my magic wand."

## Understanding the diffferent components

Now that you have a general idea of how these image generation models work, lets build each specific component out.

Also, the respective code in each section is for understanding purposes. If you wish to run the entire pipeline, Go to this [repo]().

Additionally, The below work takes heavy inspiration from the following works

- [The annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Fast ai course by Jeremy Howard](https://course.fast.ai/Lessons/part2.html)

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/9.webp) [REPLACE_THIS_WITH_THE_EXPLAINING_PARTS_IMAGE]
![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/10.webp)

### Dali The Genius Artist (UNET)

- Architecture details
- Residual connections
- Attention mechanisms
- Time embeddings
- Training objective

You will be surprised to know UNETs were actually introduced in a [medical paper](https://arxiv.org/pdf/1505.04597) back in 2015. Primarily for the task of image segmentation.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/14.webp)

> Image Taken from the "U-Net: Convolutional Networks for Biomedical
> Image Segmentation" paper

The idea behind segmentation is, given an image "a". Create a map around the objects which need to be classified in the image.

And the Reason they are called U-Net is because, well the architecture looks like a "U".

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/12.webp)

> Image Taken from the "U-Net: Convolutional Networks for Biomedical
> Image Segmentation" paper

This looks quite complicated so let's break it down with a simpler image

Also, I will proceed with the believe you have an understanding of CNNs and how they work. If not, check the appendix for a quick overview and a guide to where you can learn more on the topic.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/11.webp)

The encoder side does [convolutions]() to extract features from images, then compresses them to only focus on the relevant parts.

The decoder then does [Transpose Convolutions]() to decode these extracted parts back into the original image size.

To understand it in our context, think instead of segmenting objects, we are segmenting the noise. Trying to find out the particular places where noise is present.

To prevent the U-net from losing important information while downsampling, skip connections are added. This sends back the compressed encoded image back to the decoder so they have context from their as well.

Another thing present inside Diffusion Model U-Nets is a cross attention layer, which focuses on the important parts {explain this in greater detail}

#### Coding the original U-Net

They are easier to understand when we write them down in code. So let us do that. (We start with coding the original U-Net out first, then add the complexities of the one used in Stable Diffusion)

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

This is a simple convolution, This is done to extract relevant features from the image.

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
```

A simple Down block, that compresses the size of the image. This makes sure we only focus on the relevant part. Imagine it like this Given most images, like pictures of dogs, person in a beach, Photo of the moon etc. The most interesting part (the dog,person,moon) usually take up a small or half the photo

```python
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

{add explanation}

```python
lass UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Correct channel progression:
        self.inc = DoubleConv(3, 64)  # Initial convolution

        # Encoder path (feature maps halve, channels double)
        self.down1 = Down(64, 128)    # Output: 128 channels
        self.down2 = Down(128, 256)   # Output: 256 channels
        self.down3 = Down(256, 512)   # Output: 512 channels
        self.down4 = Down(512, 1024)  # Output: 1024 channels

        # Decoder path (feature maps double, channels halve)
        self.up1 = Up(1024, 512)      # Input: 1024 + 512 = 1536 channels
        self.up2 = Up(512, 256)       # Input: 512 + 256 = 768 channels
        self.up3 = Up(256, 128)       # Input: 256 + 128 = 384 channels
        self.up4 = Up(128, 64)        # Input: 128 + 64 = 192 channels

        # Final convolution
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Store encoder outputs for skip connections
        x1 = self.inc(x)         # [B, 64, H, W]
        x2 = self.down1(x1)      # [B, 128, H/2, W/2]
        x3 = self.down2(x2)      # [B, 256, H/4, W/4]
        x4 = self.down3(x3)      # [B, 512, H/8, W/8]
        x5 = self.down4(x4)      # [B, 1024, H/16, W/16]

        # Decoder path with skip connections
        x = self.up1(x5, x4)     # Use skip connection from x4
        x = self.up2(x, x3)      # Use skip connection from x3
        x = self.up3(x, x2)      # Use skip connection from x2
        x = self.up4(x, x1)      # Use skip connection from x1

        # Final 1x1 convolution
        logits = self.outc(x)    # [B, num_classes, H, W]

        return logits

```

{add explanation}

#### Stable Diffusion U-Net

Now let us code out the U-Net used in Stable Diffusion

### Dali's mistake fixing wand (Scheduler)

A quick note to the reader, This part is mostly pure mathematics. I have described each part in greater detail and simplification in the [maths section]() of the blog.

This here is mostly a quick idea that one will need to understand how scheduler's work. If you are interested in how these came to be, I urge you to check out the mathematics behind it, because it is quite beautiful.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/6.webp)

> Taken from the {add paper}

The above image looks quite complex, But it is really simple if you understand what is going on.

We start with an image and call it $X_0$ we then keep adding noise to it till we have pure stochatic Gausian Noise $X_T$

"$q(X_t|X(t-1))$ {fix this} is the conditional probability over the Probability Density Function"

Well wasn't that a mouthful, dont worry. I won't throw such a big sentence at you without explaining what it means.

Let's again stary with our original image $X_0$ and then we add a bit of noise to it, this is now $X_1$, then we add noise to this image that becomes $X_2$ and so on.

[INSERT_IMAGE]

That scary looking equation basically says if we have the image on the right $X_(t-1)$ we can add noise to it and get image at the next timestep and represent that as $X_t$
(This is a slight oversimplification and we dive into greater detail about it in the math section)

So now we have a single image, and we are able to add noise to it.

What we want to do is, the reverse process. Take noise and get an image out of it.

You may ask why do we not simply do what we did earlier but the otherway around so something like

$$q(X_(t-1)|X_t)$$

Well the above is simply not computationally possible because we will need to learn how the noise of all the images in the world looks like (remember how in the [idea]() section Dali said his brothers tried to do this and failed)

So we need to learn to approximate it, learn how the images might look like given the noise.

and that is given by the other equation $p(theta)$ [ADD_EQUATION]

Now above I mentioned that we add noise, but never described how.

That is done by this equation

$$q = N(mean, variance) $$

We already know what the left hand side means, lets understand the right hand side.

The RHS represents a Normal distribution $N$ with mean $43234$ and Variance $4234$, we pick out noise at time t from this distribution to add to our image.

There is one slight problem though, gradually adding so many different noise at different values of t is very computationally expensive.

Using the "nice property" we can make another equation

$$write euqation here$$

This basically means, now we can add noise at any time t just using the original image. This is amazing, why? well you will understand in a while.

You need to understand a few more things the $beta$ term in the above equation is a _variance shedule_ it basically controlls the curve the noise is added in

[ADD_IMAGE_OF_CURVE]
[ADD_IMAGE_OF_HOW_NOISE_CHANGES]

Now that we understand how we can add noise to the images, how we can control the different kinds of noise, we need an objective or loss function to train over

That is given by

$\|\epsilon - \epsilon_\theta(x_t,t)\|_2 = \|\epsilon - \epsilon_\theta(\bar{\alpha}_t x_0 + (1-\bar{\alpha}_t)\epsilon,t)\|_2$

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/5.webp)

> Image Taken from {add paper}

This greatly simplifies are training, which can be written as the above image.

"""
In other words:

- we take a random sample $\mathbf{x}_0$ from the real unknown and possibly complex data distribution $q(\mathbf{x}_0)$
- we sample a noise level $t$ uniformly between $1$ and $T$ (i.e., a random time step)
- we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ (using the nice property defined above)
- the neural network is trained to predict this noise based on the corrupted image $\mathbf{x}_t$ (i.e. noise applied on $\mathbf{x}_0$ based on known schedule $\beta_t$)

In reality, all of this is done on batches of data, as one uses stochastic gradient descent to optimize neural networks.
"""

### Instructions, because everyone needs guidance (Conditioning)

#### Text Encoder

- CLIP/T5 architecture
- Token embedding
- Text preprocessing
- Integration with conditioning

#### Different Conditioning

- Cross-attention mechanics
- Embedding integration
- Multi-modal conditioning
- Classification guidance

"""
The Text-encoder

The text-encoder is responsible for transforming the input prompt, e.g. "An astronaut riding a horse" into an embedding space that can be understood by the U-Net. It is usually a simple transformer-based encoder that maps a sequence of input tokens to a sequence of latent text-embeddings.

Inspired by Imagen, Stable Diffusion does not train the text-encoder during training and simply uses an CLIP's already trained text encoder, CLIPTextModel.
"""

### The Magical Wand (Variational Auto-Encoder)

This [video](https://www.youtube.com/watch?v=qJeaCHQ1k2w&t=1s) helped me immensely while writing this part.

Unfortunately for the both of us, This part too is very maths heavy. So again I will leave the intuition and derivation for the [maths section]() of the blog and just talk about the idea, show the equations and write out the code.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/13.webp)
The above image is actually what happens inside of an Auto-Encoder but if you are anything like me. It probably doesn't make any sense.

So let's look at a simpler representation and come back to this when it makes more sense.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/15.webp)


{add difference between VAE and AE}

"""
The VAE model has two parts, an encoder and a decoder. The encoder is used to convert the image into a low dimensional latent representation, which will serve as the input to the U-Net model. The decoder, conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent representations (latents) of the images for the forward diffusion process, which applies more and more noise at each step. During inference, the denoised latents generated by the reverse diffusion process are converted back into images using the VAE decoder. As we will see during inference we only need the VAE decoder.
"""

### Putting it all together

- Component interaction
- Training workflow
- Inference workflow
- Optimization strategies

**A quicky Summary**

"""
Before we get hands on with the code, let’s refresh how inference works for a diffuser.

- We input a prompt to the diffuser.

- This prompt is given a mathematical representation (an embedding) through the text encoder.

- A latent comprised of noise is produced.
  The U-Net predicts the noise in the latent in conjunction with the prompt.
- The predicted noise is subtracted from the latent in conjunction with the scheduler.
- After many iterations, the denoised latent is decompressed to produce our final generated image.

The main components in use are:

- a text encoder,
- a U-Net,
- and a VAE decoder.
  """

## The Dreaded Mathematics

This part was heavily influenced by the following works

- [Lil'Log blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [yang song](https://yang-song.net/blog/2021/score/)

As the above works were way too hard to understand. The following 3 videos really helped me out understand them

- [Diffusion Models From Scratch | Score-Based Generative Models Explained | Math Explained](https://www.youtube.com/watch?v=B4oHJpEJBAA)
- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [Denoising Diffusion Probabilistic Models | DDPM Explained](https://www.youtube.com/watch?v=H45lF4sUgiE&t=1583s)

As is the nature of Understanding Stable Diffusion, it is going to be mathematics heavy. I have added an appendix at the bottom where I explain each mathematical ideas as simply as possible.

It will take too much time and distract us from the understanding of the topic being talked at hand if I describe the mathematical ideas as well as the idea of the process in the same space.

## Maths of the Forward Diffusion process

Imagine you have a large dataset of images, we will represent this real data distribution as $q(x)$ and we take an image from it (data point) $x_0$.
(Which is mathematically represented as $x_0 \sim q(x)$).

In the forward diffusion process we add small amounts of Gaussian noise to the image ($x_0$) in $T$ steps. Which produces a bunch of noisy images as each step which we can label as $x_1,\ldots,x_T$. These steps are controlled by a variance schedule given by $\beta_t$. The value of $\beta_t$ ranges from 0 to 1 (i.e it can take values like 0.002, 0.5,0.283 etc) for $t, \ldots, T$. (Mathematically represented as ${\beta_t \in (0,1)}_{t=1}^T$)

There are many reasons we choose Gaussian noise, but it's mainly due to the properties of normal distribution. (about which you can read more here)

Now let us look at the big scary forward diffusion equation and understand what is going on

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I}) \tag{1}$$
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}) \tag{2}$$

$q(x_t|x_{t-1})$ means that given that I know $q(x_{t-1})$ what is the probability of $q(x_t)$ This is also knows as [bayes theorem]().

To simplify it, think of it as. given $q(x_0)$ (for value of $t$ = 1) I know the value of $q(x_1)$.

The right handside of equation 1 represents a normal distribution.

Now A question that I had was how can a probability and distribution be equal, well the Left Hand Side(LHS) of equation(eq) 1 represents a Probability Density Function ([PDF]())

For the Right Hand Side(RHS) of eq 1. When we write $N(x; μ, σ²)$, we're specifying that $x$ follows a normal distribution with mean $μ$ and variance $σ²$

This can be written as

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

As $t$ becomes larger. And eventually when $T \to \infty$ (This means as $T$ approaches infinity, or just a really large number). The initial data sample $x_0$ loses its features and turns into an isotropic Gaussian Distribution.

{explain equation 2 as well}

Let's talk about an interesting property - we can actually sample $x_t$ at any arbitrary time step. This means we don't need to go through the diffusion process step by step to get to a specific noise level.

First, let's understand something fundamental about normal distributions. Any normal distribution can be represented in the following form:

$$X = \mu + \sigma \epsilon$$

where $\epsilon \sim \mathcal{N}(0,1)$ (This means $\epsilon$ is sampled from a normal distribution with mean 0 and variance 1)

Taking our equation from before:
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

We can rewrite this using the above form as:
$$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}$$

To make our equations simpler, let's define $\alpha_t = 1-\beta_t$. This gives us:
$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}$$

Now, we can substitute the expression for $x_{t-1}$ in terms of $x_{t-2}$:
$$x_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}$$

A key property of normal distributions is that when we add two normal distributions, their means and variances can be combined. Using this property and some algebraic manipulation, we get:

$$x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2}$$

If we continue this process all the way back to our original image $x_0$, and define $\bar{\alpha}_t$ as the product of all $\alpha$s from 1 to t ($\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$), we arrive at:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

This final equation is quite powerful. It allows us to directly sample $x_t$ at any timestep $t$ using just:

- The original image $x_0$
- The cumulative product of alphas up to time $t$ ($\bar{\alpha}_t$)
- A sample from a standard normal distribution ($\epsilon$)

This makes our implementation much more efficient as we can directly jump to any noise level without calculating all the intermediate steps.

{explain about alpha as well, and rewrite this in your tone a bit more}

"""\
Usually, we can afford a larger update step when the sample gets noisier, so $\beta_1 < \beta_2 < \cdots < \beta_T$ and therefore $\bar{\alpha}_1 > \cdots > \bar{\alpha}_T$.\
"""

"""\
**Connection with stochastic gradient Langevin dynamics**\
Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, stochastic gradient Langevin dynamics (Welling & Teh 2011) can produce samples from a probability density $p(x)$ using only the gradients $\nabla_x \log p(x)$ in a Markov chain of updates:
$$x_t = x_{t-1} + \frac{\delta}{2}\nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t, \text{ where } \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$
where $\delta$ is the step size. When $T \to \infty, \delta \to 0$, $x_T$ equals to the true probability density $p(x)$.
Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.\
"""

Let me help you understand Langevin dynamics and its connection to diffusion models. This is a fascinating bridge between physics and machine learning!
First, let's understand what Langevin dynamics is trying to do. Imagine you're trying to find the lowest point in a hilly landscape while blindfolded. If you only walk downhill (like regular gradient descent), you might get stuck in a small valley that isn't actually the lowest point. Langevin dynamics solves this by occasionally taking random steps - like sometimes walking uphill - which helps you explore more of the landscape.
The key equation is:
xt = xt−1 + (δ/2)∇xlog⁡p(xt−1) + δϵt
Let's break this down piece by piece:

xt and xt−1 represent our position at the current and previous steps
∇xlog⁡p(xt−1) is the gradient term - it tells us which direction to move to increase the probability
δ is our step size - how far we move in each step
ϵt is our random noise term, sampled from a normal distribution

The equation combines two behaviors:

A "deterministic" part: (δ/2)∇xlog⁡p(xt−1) which moves us toward higher probability regions
A "random" part: δϵt which adds noise to help us explore

What makes this special is that when we run this process for a long time (T→∞) and with very small steps (δ→0), we're guaranteed to sample from the true probability distribution p(x). This is similar to how diffusion models gradually denoise images - they're following a similar kind of path, but in reverse!
The connection to standard gradient descent is interesting - regular SGD would only have the gradient term, but Langevin dynamics adds that noise term ϵt. This noise prevents us from getting stuck in bad local minima, just like how shaking a jar of marbles helps them settle into a better arrangement.

Let me explain why we use the logarithm of the probability (log p(x)) instead of just the probability p(x) itself. This is a fundamental concept in probability and optimization that's used across many areas of machine learning.
First, let's understand why logarithms are useful in general with probabilities:

Numerical Stability
When we work with probabilities, we're dealing with numbers between 0 and 1. When we multiply many small probabilities together (which happens often in machine learning), we can end up with extremely small numbers that computers struggle to handle accurately. For example:
0.01 × 0.01 × 0.01 = 0.000001

But if we take logarithms:
log(0.01) + log(0.01) + log(0.01) = -13.82
The logarithm transforms these tiny numbers into more manageable negative numbers that computers can handle better.

Converting Multiplication to Addition
In probability theory, we often need to multiply many probabilities together. Taking the logarithm converts multiplication into addition, which is computationally simpler and more stable:
log(a × b) = log(a) + log(b)
Gradient Properties
Now, here's the key insight for why we use log p(x) in Langevin dynamics: When we take the gradient of log probability (∇log p(x)), we get something called the "score function". This score function has a special property - it points in the direction where the probability increases most rapidly.

Let's see why through calculus:
∇log p(x) = ∇(log p(x)) = (1/p(x))∇p(x)
This division by p(x) acts as an automatic scaling factor. When p(x) is small, it makes the gradient larger, and when p(x) is large, it makes the gradient smaller. This natural scaling helps our sampling process explore the probability space more efficiently.

What is P(x) though and why are we taking that. Traditionally in SGD do we not take, del(error)/del(weight)

In traditional SGD for neural networks, we're trying to minimize an error function (or loss function), so we use ∂(error)/∂(weight) as you correctly noted. We're trying to find the weights that make our predictions as accurate as possible.
But in Langevin dynamics, we're doing something fundamentally different. Here, p(x) represents a probability distribution that we want to sample from. Think of it this way:
Imagine you have a dataset of faces, and you want to generate new faces that look real. The probability p(x) would represent how likely it is that a particular image x is a real face. Areas of high p(x) would correspond to images that look like real faces, while areas of low p(x) would be images that don't look like faces at all.
So when we take ∇log p(x), we're asking: "In which direction should I move to make this image look more like a real face?"
Let's compare:

Traditional SGD in Neural Networks:

Goal: Minimize error
Gradient: ∂(error)/∂(weight)
Updates weights to make predictions more accurate

Langevin Dynamics:

Goal: Sample from a probability distribution
Gradient: ∇log p(x)
Updates the sample itself to look more like real data

This is why Langevin dynamics is particularly relevant to diffusion models. Remember how diffusion models start with noise and gradually transform it into an image? The ∇log p(x) term tells us how to modify our noisy image at each step to make it look more like real data.

"""

TRAINING THE MODEL

Since we can't have x₀ during generation, we train a model pθ(xₜ₋₁|xₜ) to approximate q(xₜ₋₁|xₜ,x₀). This model learns to predict the denoising step without needing the original image.
The training process works like this:

Take a clean image x₀
Sample a random timestep t
Add noise to get xₜ using our "nice property" formula
Train the model to predict the noise that was added
The model learns to do this by minimizing the difference between its prediction and the actual noise

## Maths of Reverse diffusion process

Now what we want to do is take a noisy image $x_t$ and get the original image $x_0$ from it. And to do that we need to do a reverse diffusion process.

Essentially we want to sample from $q(x_{t-1}|x_t)$, Which is quite tough as there can be millions of noisy images for actual images. To combat this we create an approximation (why do they work and how do they work in a minute) $p_\theta$ to approximate these conditional probabilities in order to run the _reverse diffusion process_.

Which can be represented as
$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$

Unfortunately it is tough to even sample from this approximate model because it is the same as our previous model, so we modify it by adding the original image $x_0$ to it as such.

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t,x_0), \tilde{\beta}_t\mathbf{I})$$

Now this is tractable (Exaplain what this word means), let us first understand the proof for how it is tractable. Later moving on to understand how they thought of this idea in the first place

Using Bayes' rule, we have:

$$
\begin{aligned}
q(x_{t-1}|x_t,x_0) &= \frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} \\
&\propto \exp(-\frac{1}{2}(\frac{(x_t-\alpha_tx_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\bar{\alpha}_{t-1}x_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t-\bar{\alpha}_tx_0)^2}{1-\bar{\alpha}_t})) \\
&= \exp(-\frac{1}{2}(\frac{x_t^2-2\alpha_tx_tx_{t-1}+\alpha_tx_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2-2\bar{\alpha}_{t-1}x_0x_{t-1}+\bar{\alpha}_{t-1}x_0^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t-\bar{\alpha}_tx_0)^2}{1-\bar{\alpha}_t})) \\
&= \exp(-\frac{1}{2}((\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^2 - (\frac{2\alpha_t}{\beta_t}x_t+\frac{2\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)x_{t-1} + C(x_t,x_0)))
\end{aligned}
$$

where $C(x_t,x_0)$ is some function not involving $x_{t-1}$ and details are omitted. Following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$):

$$
\begin{aligned}
\tilde{\beta}_t &= 1/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= 1/(\frac{\alpha_t-\bar{\alpha}_t+\beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}) \\
&= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t
\end{aligned}
$$

$$
\begin{aligned}
\tilde{\mu}_t(x_t,x_0) &= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t \\
&= \frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\bar{\alpha}_{t-1}\beta_t}{1-\bar{\alpha}_t}x_0
\end{aligned}
$$

Thanks to the nice property, we can represent $x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)$ and plug it into the above equation and obtain:

$$\tilde{\mu}_t = \frac{1}{\alpha_t}(x_{t-1}-\frac{\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t)$$

{add color coding for the above to make it easier to understand}

"""
As demonstrated in Fig. 2., such a setup is very similar to VAE and thus we can use the variational lower bound to optimize the negative log-likelihood.

$$
\begin{aligned}
-\log p_\theta(x_0) &\leq -\log p_\theta(x_0) + D_{KL}(q(x_{1:T}|x_0)\|p_\theta(x_{1:T}|x_0)); \text{ KL is non-negative} \\
&= -\log p_\theta(x_0) + \mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}[\log \frac{q(x_{1:T}|x_0)p_\theta(x_{0:T})}{p_\theta(x_0)}] \\
&= -\log p_\theta(x_0) + \mathbb{E}_q[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0)] \\
&= \mathbb{E}_q[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}]
\end{aligned}
$$

Let $\mathcal{L}_{VLB} = \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] \geq -\mathbb{E}_{q(x_0)}\log p_\theta(x_0)$

It is also straightforward to get the same result using Jensen's inequality. Say we want to minimize the cross entropy as the learning objective,

$$
\begin{aligned}
\mathcal{L}_{CE} &= -\mathbb{E}_{q(x_0)}\log p_\theta(x_0) \\
&= -\mathbb{E}_{q(x_0)}\log(\int p_\theta(x_{0:T})dx_{1:T}) \\
&= -\mathbb{E}_{q(x_0)}\log(\int \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}p_\theta(x_{0:T})dx_{1:T}) \\
&= -\mathbb{E}_{q(x_0)}\log(\mathbb{E}_{q(x_{1:T}|x_0)}\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}) \\
&\leq -\mathbb{E}_{q(x_{0:T})}\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
&= \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] = \mathcal{L}_{VLB}
\end{aligned}
$$

[Continued in next message...]

To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several KL-divergence and entropy terms:

$$
\begin{aligned}
\mathcal{L}_{VLB} &= \mathbb{E}_{q(x_{0:T})}[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}] \\
&= \mathbb{E}_q[\log \frac{\prod_{t=1}^T q(x_t|x_{t-1})}{p_\theta(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=1}^T \log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=2}^T \log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
&= \mathbb{E}_q[-\log p_\theta(x_T) + \sum_{t=2}^T \log \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} \cdot \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} + \log \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}] \\
&= \mathbb{E}_q[D_{KL}(q(x_T|x_0)\|p_\theta(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)]
\end{aligned}
$$

Let's label each component in the variational lower bound loss separately:

$$\mathcal{L}_{VLB} = L_T + L_{T-1} + \cdots + L_0$$

where:

$$
\begin{aligned}
L_T &= D_{KL}(q(x_T|x_0)\|p_\theta(x_T)) \\
L_t &= D_{KL}(q(x_t|x_{t+1},x_0)\|p_\theta(x_t|x_{t+1})) \text{ for } 1 \leq t \leq T-1 \\
L_0 &= -\log p_\theta(x_0|x_1)
\end{aligned}
$$

Every KL term in $\mathcal{L}_{VLB}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in closed form. $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $x_T$ is a Gaussian noise. Ho et al. 2020 models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(x_0; \mu_\theta(x_1,1), \Sigma_\theta(x_1,1))$.
"""

## Maths of VAE

### Unet

You will be surprised to know that the idea of Unets comes from a [medical paper](https://arxiv.org/pdf/1505.04597)

As is the nature of things, I am going with the assumption you understand what CNNs are
and the different kinds of task in image classification. Below I have created a simple visualization for the different tasks.

But if you do not know CNNs Let me give a quick overview below otherwise, consider reading my CV blog(If the link doesn't take you anywhere, it's still under development) or read this.

Helpful docs

[Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)\
[BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html1)\
[MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)\
[Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)\
[torch.cat](https://pytorch.org/docs/main/generated/torch.cat.html)\
[ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)\
[Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)

### VAE

### CLIP

### DDPM

### DDIM

## The code

## Understanding the metrics

This is interesting as well because... how do you tell a computer which is a good image and which is a bad image without actually doing a vibe check.

This really makes you appreaciate how the loss function was created doesnt it now!!

## Things to talk about from the fast ai notebooks:

- [Stable diffsion components](https://forbo7.github.io/forblog/posts/13_implementing_stable_diffusion_from_its_components.html) Build SD from taking components from HF
-

## Appendix

### PDF

### KL Divergence

### Bayes' rule -->
