<!-- ---
layout: blog
title: "Demystifying Diffusion Models"
date: 2025-02-3 12:00:00 +0530
categories: [CV, ML, Maths, Code]
image: assets/blog_assets/demystifying_diffusion_models/temp_meme_img.webp
---

Diffusion models like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Dall-e](https://openai.com/index/dall-e-3/) etc are an enigma built upon multiple ideas and mathematical breakthroughs. So is the nature of it that most tutorials on the topic are extremely complicated or even when simplified talk a lot about it from a high level perspective.

There is a missing bridge between the beautiful simplification and more low level complex idea. That is the gap I have tried to fix in this blog.

- Starting with the simple **idea** behind diffusion models
- A full section dedicated to the **maths** for the curious minds
- Understanding each component and **coding** it out

Each section of the blog has been influenced by works by pioneering ML practitioners and the link to their blog/video/article is linked in the very beginning of the respective section.

## How this Blog is Structured

First we talk about a very high level idea of diffusion models about how they work. In doing so we will be personifying each component of the whole pipeline.

Once we have a general idea of the pipeline, We will dive into the ML side of those sections.

Many sections of the diffusion model pipeline is mathematics heavy, hence I have added a completely different section for that. Which is included after we understand the ML components. You can understand how diffusion models work (if you believe in some assumptions without looking at the proof) along with the code, without the maths. But I will still recommend going through the Mathematical ideas behind it, because they are essential for developing further for diffusion model research.

After Understanding everything, we will code it out. As it is substantially harder to keep the blog to readable length and maintain it's quality while giving the entire code for Stable Diffusion, I will link to the exact code (with definition for each function)
Wherever I do not explicitly code a section out.

Inference with Diffusion model deserves an entirely different blog of it's own, as I hope to finish this blog in a reasonable time. I have added links in the end ([Misc]()) to where you can further learn how to make the best diffusion model art and get better at it.

Let us begin!!

## The Genius Artist

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/1.webp)
Imagine you have a super special artist friend, whom you tell your ideas and he instantly generates amazing images out of it. Let's name him Dali

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/2.webp)
The way Dali starts his work is, that he first has a canvas, he listens to your instructions then creates an artwork. (The canvas looks a lot like noise rather than the traditional white, more on this later)

But Dali has a big problem, that he cannot make big images, he tells you that he will only create images the size of your hand. This is obviously not desirable. As for practical purposes you may want images the size of a wall, or a poster etc.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/8.webp)
That is when a magic wand falls from the sky, and it has two modes Encoder(Compress size) and Decoder(Enlarge size). That gives you a great idea. You will start with the size of the canvas that you like, Encode it. Give the encoded canvas to Dali, he will make his art, And then you can decode the created art to get it back to the original shape you want.

This works and you are really happy.

But you are curious about how Dali works, so you ask him. "Dali why do you always start with this noisy canvas instead of pure white canvas? and how did you learn to generate so many amazing images?"

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/4.webp)
Dali is a kind nice guy, so he tells you about how he started out. When he was just a newbie artist. The world was filled with great art. Art so complex that I could not reproduce it, nobody could.

That is when I found a special wand as well, which let me add and fix mistakes in a painting.
![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/7.webp)

I would start with an existing artwork, add a bunch of mistakes to it, and using my wand I would reverse them.

After a while, I added so many mistakes to the original artwork, that they looked like pure noise. The way my canvas do, and using my special wand. I just gradually found mistakes and removed them. Till I got back the original image.

This idea sounds fascinating, but you being you have quite a question "that sounds amazing, so did you learn what the "full of mistakes" image will look like for all the images in the world? Otherwise how do you know what will be the final image be from a noisy image?"

"Great question!!!" Dali responds. "That is what my brothers used to do, They tried to learn the representation of all the images in the world and failed. What I did differently was, instead of learning all the images. I learnt the general idea of different images. For example, instead of learning all the faces. I learnt how do human faces look in general"

Satisfied with his answers you were about to leave, when Dali stops you and asks, "Say friend, that wand of yours truly is magical. It can make my art popular worldwide because everyone can create something of value using it. Will you be kind enough to explain how it works so I can make one for myself."

You really want to help Dali out, but unfortunately even you do not know how the wand works, as you are about to break the news to him. You are interrupted by a noise, "Gentlemen you wouldn't happen to have seen a magic wand around now would you? It is an artifact created with great toil and time"

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/28.webp)

You being the kind soul you are, Tell the man that you found it on the street and wish to return it.
The man Greatly happy with your generosity, wishes to pay you back. You just say "Thank you, but I do not seek money. But it would really help my friend Dali out if you could explain how your magic wand works."

The man curious for what use anyone would have for his magic wand sees around Dali's studio, and understands that he is a great artist. Happy to help him he says. "My name is Auto, and I shall tell you about my magic wand."

## Understanding the diffferent components

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/24.webp)

Now that you have a general idea of how these image generation models work, lets build each specific component out.

Also, the respective code in each section is for understanding purposes. If you wish to run the entire pipeline, Go to this [repo]().

Additionally, The below work takes heavy inspiration from the following works

- [The annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Fast ai course by Jeremy Howard](https://course.fast.ai/Lessons/part2.html)

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/9.webp)

If you look closely you will see how similar both these images are.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/10.webp)

The above is an oversimplification and has a few mistakes. But by the end of this blog you will have a complete understanding of how diffusion models work and how the seemingly complex model above, is quite simple.

### Dali The Genius Artist (U-Net)

Our genius artist is called a U-Net in ML terms, now if we go back to our story. Dali was responsible for figuring out the noise. The removal and addition of which was done by his magic wand. That is what the U-Net does. It predicts the noise in the image, it DOES NOT REMOVE IT.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/25.webp)

Let's understand how it works, You will be surprised to know U-Nets were actually introduced in a [medical paper](https://arxiv.org/pdf/1505.04597) back in 2015. Primarily for the task of image segmentation.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/14.webp)

> Image Taken from the ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)

The idea behind segmentation is, given an image "a". Create a map "b" around the objects which need to be classified in the image.

And the Reason they are called U-Net is because, well the architecture looks like a "U".

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/12.webp)

> Image Taken from the ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597)

This looks quite complicated so let's break it down with a simpler image

Also, I will proceed with the assumption you have an understanding of [CNNs]() and how they work. If not, check the [appendix]() for a quick overview and a guide to where you can learn more on the topic.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/11.webp)

The encoder side does [convolutions]() to extract features from images, then compresses them to only focus on the relevant parts.

The decoder then does [Transpose Convolutions]() to decode these extracted parts back into the original image size.

To understand it in our context, think instead of segmenting objects, we are segmenting the noise. Trying to find out the particular places where noise is present.

To prevent the U-net from losing important information while downsampling, skip connections are added. This sends back the compressed encoded image back to the decoder so they have context from their as well.

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
        # Choose between two upsampling methods
        if bilinear:
            # Method 1: Simple bilinear interpolation
            # Doubles the spatial dimensions using interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Method 2: Learnable transposed convolution
            # Doubles spatial dimensions while halving channels
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Apply two consecutive convolutions after combining features
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Step 1: Upsample the lower resolution feature map (x1)
        x1 = self.up(x1)

        # Step 2: Handle size mismatches between upsampled and skip connection
        # Calculate the difference in height and width
        diff_y = x2.size()[2] - x1.size()[2]  # Height difference
        diff_x = x2.size()[3] - x1.size()[3]  # Width difference

        # Add padding to make sizes match
        # The padding is distributed evenly on both sides
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])

        # Step 3: Combine features from upsampled path and skip connection
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel dimension

        # Step 4: Process combined features through double convolution
        return self.conv(x)
```

```python
class UNet(nn.Module):
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

By putting together the Down and Up blocks we have the final U-Net as coded above.

#### Stable Diffusion U-Net

The Diffusion Model U-Net is quite different. But it follows the same principles as discussed above.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/22.webp)

First, let's understand what goes into this U-Net:

Input Components

- Noisy Latents: The noisy image we're trying to denoise
- Prompt Embedding: Text information converted into a numerical representation
- Time Step Embedding: Information about which denoising step we're on

The ResNet and Attention blocks work together in a complementary way to process this information:

**ResNet Blocks**\
These blocks receive three inputs that are combined:

1. The main feature path (coming from previous layers)
2. The time step embedding
3. The residual skip connection (from earlier in the network)

Inside a ResNet Block (Pseudo-code):

```python
# Simplified ResNet block structure
def resnet_block(features, time_embedding, skip_connection):
    # Time embedding projection
    time_features = project_time_embedding(time_embedding)

    # Combine features with time information
    combined = features + time_features

    # Apply convolutions with residual connection
    output = conv1(combined)
    output = activation(output)
    output = conv2(output)

    # Add skip connection
    final = output + skip_connection
    return final
```

If you are new to ResNets, consider reading more [here](https://medium.com/towards-data-science/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96)

The ResNet blocks are crucial because they:

- Maintain spatial information about the image
  \*Help the model understand how features should change based on the denoising step
- Prevent vanishing gradients through residual connections

**Attention Blocks**\
Attention blocks receive:

- The feature maps from ResNet blocks
- The prompt embedding (indirectly through cross-attention)

Inside an attention block(Pseudo-code):

```python
# Simplified attention block structure
def attention_block(features, prompt_embedding):
    # Self-attention: Image features attending to themselves
    q, k, v = project_to_qkv(features)
    self_attention = compute_attention(q, k, v)

    # Cross-attention: Image features attending to text
    q_cross = project_query(features)
    k_cross, v_cross = project_kv(prompt_embedding)
    cross_attention = compute_attention(q_cross, k_cross, v_cross)

    # Combine both attention results
    output = self_attention + cross_attention
    return output
```

To read more about attention, consider reading my blog on the topic [here](https://goyalpramod.github.io/blogs/Transformers_laid_out/#understanding-self-attention)

The attention blocks are essential because they:

- Help the model focus on relevant parts of the image based on the text prompt
- Allow the model to understand relationships between different parts of the image
- Enable text-image alignment during the generation process

Why This Architecture Works So Well

1. Progressive Refinement

   - The U-Net structure allows the model to work at multiple scales
   - Early layers capture broad structure
   - Middle layers refine details
   - Later layers add fine details

2. Information Flow

   - ResNet blocks ensure gradient flow and feature preservation
   - Attention blocks align image generation with text description
   - Skip connections preserve spatial information

3. Controlled Generation

   - Time step embeddings guide the denoising process
   - Prompt embeddings guide the semantic content
   - Their combination enables precise control over the generation

For example, when generating an image of "a red cat sitting on a blue chair":

1. ResNet blocks handle the basic structure and progressive denoising
2. Attention blocks ensure:

   - The cat is actually red
   - The chair is actually blue
   - The cat is positioned correctly relative to the chair

3. Skip connections preserve spatial details throughout the process

#### Diffusion Transformer (DiT)

Remember how earlier we talked about U-Nets being the brain behind image generation? Well, there's another architecture that's becoming increasingly popular - the Diffusion Transformer (DiT). Think of it as giving Dali a different kind of artistic training, one that's more about seeing the whole canvas at once rather than focusing on different parts separately.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/38.webp)

> Image taken from ["Scalable Diffusion Models with Transformers"](https://arxiv.org/pdf/2212.09748)

To understand DiTs, let's first recall what made U-Nets special:

1. They look at images at different scales (like an artist looking at both details and the big picture)
2. They have skip connections (like keeping notes about earlier versions of the painting)
3. They process information in a hierarchical way (working from rough sketch to fine details)

DiTs take a completely different approach. Instead of processing the image in this hierarchical way, they treat the image more like a sequence of patches - imagine cutting up the canvas into small squares and looking at how each square relates to all other squares.

Here's how it works:

1. **Patch Embedding**:

   - The noisy image is divided into small patches (usually 16×16 pixels)
   - Each patch is converted into a sequence of numbers (like translating visual information into a language DiT can understand)
   - If this reminds you of how CLIP processes images, you're spot on!

2. **Global Attention**:

   - Unlike U-Net where each part mainly focuses on its neighbors, in DiT every patch can directly interact with every other patch
   - It's like the artist being able to simultaneously consider how every part of the painting relates to every other part
   - This is done through transformer blocks, similar to what powers ChatGPT but adapted for images

3. **Time and Prompt Integration**:
   - The noise level (time step) is embedded directly into the sequence
   - Text prompts are also converted into embeddings and can influence how patches interact
   - This creates a unified way for the model to consider all the information at once

The advantages of DiTs are:

- Better at capturing long-range relationships in images (like ensuring consistency across the entire image)
- More flexible in handling different types of inputs
- Often easier to scale up to larger models

But there's no free lunch - DiTs typically require more computational resources than U-Nets and can be trickier to train. This is why many popular models still use U-Nets or hybrid approaches.

A practical example of DiT's power is in handling global consistency. Let's say you're generating an image of "a symmetrical face":

- A U-Net might need to work hard to ensure both sides of the face match
- A DiT can more easily maintain symmetry because it's always looking at the relationship between all parts of the face simultaneously

The future is likely to see more models using DiT architectures or hybrid approaches combining the best of both U-Net and DiT worlds. After all, even great artists can learn new techniques!

### Dali's mistake fixing wand (Scheduler)

> A quick note, This part is mostly purely Mathematical. And as mentioned earlier, everything is described in greater detail in the maths section.\
> This here is mostly a quick idea that one will need to understand how scheduler's work. If you are interested in how these came to be, I urge you to check out the mathematics behind it, because it is quite beautiful.\
> Also, if at any point during the explanation, if it becomes too complex. Take a break and come back, Each part alone took me weeks to write. Do not assume you can understand it in one sitting, and the idea only becomes simpler as you read more about it.

As mentioned earlier, The U-Net does not remove the noise, it just predicts it. The job of removing it comes down to the scheduler.

Put simply, the scheduler is just a mathematical equation that takes an image, predicted noise. And outputs another image with some noise removed from it. (I am saying images, but actually matrices are passed around)

![Denoising process of an image](/assets/blog_assets/demystifying_diffusion_models/6.webp)

> Image taken from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

The above image looks quite complex, But it is really simple if you understand what is going on.

We start with an image and call it $X_0$ we then keep adding noise to it till we have pure [Stochastic]()(random) [Gaussian]()(Normal Distribution) Noise $X_T$.

$$q(x_t|x_{t-1})$$

"The above equation is the conditional probability over the Probability Density Function"

Well wasn't that a mouthful, dont worry. I won't throw such a big sentence at you without explaining what it means.

Let's again stary with our original image $X_0$ and then we add a bit of noise to it, this is now $X_1$, then we add noise to this image that becomes $X_2$ and so on.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/29.webp)

That scary looking equation basically says if we have the image an image $X_{t-1}$ we can add noise to it and get image at the next timestep and represent that as $X_t$
(This is a slight oversimplification and we dive into greater detail about it in the math section)

So now we have a single image, and we are able to add noise to it.

What we want to do is, the reverse process. Take noise and get an image out of it.

You may ask why do we not simply do what we did earlier but the otherway around so something like

$$q(X_{t-1}|X_t)$$

Well the above is simply not computationally possible because we will need to learn how the noise of all the images in the world looks like (remember how in the [idea]() section Dali said his brothers tried to do this and failed)

So we need to learn to approximate it, learn how the images might look like given the noise.

and that is given by the other equation

$$p_\theta(x_{t-1}|x_t)$$

Now above I mentioned that we add noise, but never described how.

That is done by this equation

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

We already know what the left hand side means, lets understand the right hand side.

The RHS represents a Normal distribution $\mathcal{N}$ with mean $\sqrt{1-\beta_t}x_{t-1}$ and variance $\beta_t\mathbf{I}$, where we sample noise at time $t$ from this distribution to add to our image.

There is one slight problem though, gradually adding so many different noise at different values of t is very computationally expensive.

Using the "nice property" we can make another equation

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$
where $\bar{\alpha}t = \prod{s=1}^t \alpha_s$ and $\alpha_t = 1-\beta_t$

This basically means, now we can add noise at any time t just using the original image. This is amazing, why? well you will understand in a while.

You need to understand a few more things, the $\beta$ term in the above equation is a _variance shedule_ it basically controls the curve the noise is added into the image.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/30.webp)

> Image taken from ["Improved Denoising Diffusion Probabilistic Models"](https://arxiv.org/pdf/2102.09672)

The above image represents how value of $\beta$ is varied.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/31.webp)

> Image taken from ["Improved Denoising Diffusion Probabilistic Models"](https://arxiv.org/pdf/2102.09672)

> Top is nosie being added by a linear variance scheduler, notice how after only a few steps the image starts looking like complete noise

> Bottom is noise being added by a cosine variance scheduler.

Now that we understand how we can add noise to the images & how we can control the different kinds of noise. But there is something much more important that we need to talk about, that is. WHY ARE WE DOING THIS and WHY DOES THIS WORK?

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/33.webp)

Images that look like images actually lie is a very specific region of all possible images in the world. An easy way to think about it will be like this, Most humans only have 2 eyes. But if you are given an infinite space of images, the pictures of humans can have n number of eyes. But you only want the images which has 2. So that significantly limits the space from where you want to get your images.

So we initially when we are adding noise to an image, we are taking it from this very specific space, to the more random gaussian space. (This is done, so we can learn the reverse process. Given any random point in space, get back to this very specific space)

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/32.webp)

> On Left, complex initial image

> Red line represents guassian noise being added

> On right, final Normal curve

The power of using normal (Gaussian) distributions in diffusion models comes from a fundamental property called the "stability property" of normal distributions. Here's how it works:

When we start with any distribution (like our complex image) and add Gaussian noise to it repeatedly, the resulting distribution gradually becomes more and more Gaussian. This is due to the [_Central Limit Theorem_](https://en.wikipedia.org/wiki/Central_limit_theorem), one of the most important principles in probability theory.

Think of it like mixing paint colors: If you start with any color (our original image distribution) and keep adding white paint (Gaussian noise) in small amounts, eventually your color will become consistently whitish, regardless of what color you started with. Similarly, adding Gaussian noise gradually transforms our complex image distribution into a simple Gaussian distribution.

we need an objective or loss function to train over

That is given by

$\|\epsilon - \epsilon_\theta(x_t,t)\|_2 = \|\epsilon - \epsilon_\theta(\bar{\alpha}_t x_0 + (1-\bar{\alpha}_t)\epsilon,t)\|_2$

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/37.webp)

> Image Taken from {add paper}

This greatly simplifies are training, which can be written as the above image.

In summary:

- **Original Artwork ($\mathbf{x}_0$)**: We start with a clean image from our dataset.
- **Progressive Damage (t)**: We simulate different levels of damage by choosing a random time step t. It's like choosing how degraded we want our image to be.
- **Adding Known Damage ($\mathbf{x}_t$)**: We add a specific amount of Gaussian noise to our image based on t. This is like deliberately damaging the artwork in a controlled way, where we know exactly what damage we added.
- **Training the Restorer**: Our neural network (like our art restorer) looks at the damaged image and tries to identify what damage was added. The loss function $|\epsilon - \epsilon_\theta(x_t,t)|_2$ measures how well the network identified the damage.

This process is efficient because:

We can jump to any level of noise directly (thanks to the "nice property")\
We know exactly what noise we added, so we can precisely measure how well our model predicts it\
By learning to identify the noise at any degradation level, the model implicitly learns how to restore images

This training happens in batches, where the model learns from multiple examples simultaneously, gradually improving its ability to identify and later remove noise from images.

Above we discussed mainly about DDPM, but there are many kinds of schedulers. You can check few of the popular one's [here](https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers)

To know more about the differences during inference. Check this [blog](https://stable-diffusion-art.com/samplers/)

### Instructions, because everyone needs guidance (Conditioning)

So far we have talked about how to generate images but have conveniently skipped over how to describe the kinds of images we want. This was another major revolution for Diffusion models, because back even when we could generate high quality images using models like [GANs](), it was tough to tell them what we want them to generate. Let us focus on that now.

Over the years the field of image gen has substantially improved and now we are not only limited to texts as a means of helping us generate images.

We can use image sources as guidance, a drawing of a rough idea, structure of an image etc. Some examples are shown below.

As Text based conditioning was the first that gained public popularity. Let's understand more on that.

#### Text Encoder

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/18.webp)

The idea is relatively simple, we take texts, convert them into embeddings and send them to the U-Net layer for conditioning.

The how is more interesting if you think about it in my opinion. Throughout our discussion of diffusion models, we never talked about image description or any means to teach a model about an image.

All the diffusion model understands is how a image looks like, without any idea about what an image is and what it contains. It's just really good at creating images which well... look like images.

Then how can we guide it using texts about what we want it to do.

That is where CLIP comes in, first let's understand what it does, then moving on to understand how it does it.

As I described initially, CLIP simply takes the text and converts it into embeddings.

These embeddings do not actually represent semantic meaning of text as they usually do in NLP, here they represent image structure, depth, and overall idea of an image.

These details are fed into the U-Net while the model tries to denoise the input image. With guidance from clip.

So the magic is introduced by CLIP, let us understand how CLIP was made.

##### CLIP (Contrastive Language–Image Pre-training)

It was originally created as a image classification tool, Given an image, Describe what it is talking about

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/19.webp)

> Image taken from [OpenAI's article on CLIP](https://openai.com/index/clip/)

Contrastive Language-Image Pre-training or CLIP pre-trains an image encoder and a text encoder which is used to predict which images are paired with which texts.

The image encoder takes images and converts them into embeddings, as you can see in the above image $I_1$ represents image 1 embeddings, $I_2$ represents image 2 embeddings and so on.

The text encoder takes captions of the images and converts them into embeddings similarily, $T_1$ for text 1 embeddings, $T_2$ for text 2 embeddings and so on.

Now as shown above, The matrix comprises of dot product of these text and image encoding. The diagnol of the matrix is maximised whereas everything else is minimised.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/20.webp)

> Image taken from [OpenAI's article on CLIP](https://openai.com/index/clip/)

Now CLIP was originally trained for zero-shot image classification. (which is a complex way of saying that "given an image, tell what it is. Without any clues".)

As you can see from the above image, when given an image and a dataset. CLIP returns the word which has the highest dot-product (The dot-product measures the similarity) with the image encoding.

Now we primarily talked about CLIP, But there is another text encoder that is used called [T5](<https://en.wikipedia.org/wiki/T5_(language_model)>) created by Google. The idea is more or less similar the only difference is

T5 (Text-to-Text Transfer Transformer) differs from CLIP's text encoder in several key ways:

1. **Architecture & Purpose**:

   - CLIP's text encoder is specifically designed for image-text alignment
   - T5 is a general-purpose text model that treats every NLP task as a text-to-text problem

2. **Training Approach**:

   - CLIP learns through contrastive learning between image-text pairs
   - T5 is trained through a masked language modeling approach, where it learns to generate text by completing various text-based tasks

3. **Output Use**:
   - In Stable Diffusion, T5 creates richer text embeddings that capture more nuanced language understanding
   - These embeddings provide more detailed semantic information to guide the image generation process

The main reason Stable Diffusion uses T5 is its superior ability to understand and represent complex text descriptions, helping create more accurate and detailed image generations.

#### Image to Image

Image to Image is of multiple types, you can have FaceSwap, Inpainting, ControlNet etc. But all of them follow a fairly simple method. If you have understood everything so far, this part will be a ride in the park.

##### Control-Net

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/34.webp)

> Image taken from ["Adding Conditional Control to Text-to-Image Diffusion Models"](https://arxiv.org/pdf/2302.05543)

Control-Net is a popular method in the world of diffusion model, where you can take a reference image and based on that add different conditioning to achieve amazing and beautiful results. Let us understand how it works.

This part was inspired by this [blog](https://blog.bria.ai/exploring-controlnet-a-new-perspective)

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/36.webp)

Control-Net essentially has two components ->

1. The diffusion model used for generating images
2. The Control-Net used for conditioning

The process itself is rather simple, we start with an image convert it into depth, canny or HUD representation.

Then give this representation to the Control-Net which conditions the DIffusion model during the denoising process.

Everything will make more sense when we see the internal architecture and understand how the control-net model is trained.

**Note**: As mentioned, a complete Control-Net model is trained for a diffusion model, so a Control-Net model trained for one model won't work on another. For example, a control net model trained for SD1.5 cannot be used for SDXL.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/21.webp)

> Image taken from ["Adding Conditional Control to Text-to-Image Diffusion Models"](https://arxiv.org/pdf/2302.05543)

Training a Control-Net model consists of the following steps:

1. Cloning the pre-trained parameters of a Diffusion model, such as Stable Diffusion's latent UNet(The part on the right referred to as "b"), while also maintaining the pre-trained parameters separately(The part on the left referred to as "a").

2. "a" is kept locked, i.e not trained while training a Control-Net model to preserve the knowledge of the Diffusion model.

3. The trainable blocks of "b" are trained to learn features specific to an image.

4. The two copies ("a" and "b") are connected through "zero convolution" layers, which are trained to bridge the locked knowledge from "a" with the new conditions being learned in "b".

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/26.webp)

As you can see above. The SD U-Net takes in the text encoding and the noisy image. And then generates the predicted noise.

The loss calculated then is used to train the Control-Net model (inside green block)

The Control-net model consists of two parts, The transformers and the Control U-Net. The control U-Net is very similar to our original unet that we started with a few important changes. (More on this in a while)

The transformer's job is to take whatever condition we give it (depth map, edge detection etc.) and convert it into something that our U-Net can understand (latent space representation).

Think of it like this, if you are reading a book in Spanish but only know English, you will need a translator. Similarly, our U-Net only understands latent space representation, so the transformer acts as a translator converting our condition into this representation.

Let's understand how this translation happens:

1. First Contact:

   - The condition (like a depth map) goes through multiple convolution layers
   - These layers are like filters that extract important features from our input
   - For example, in a depth map, it might identify where objects start and end

2. Creating Understanding:
   - These features are organized into a special format called a tensor
   - Think of a tensor like a very organized filing cabinet where each drawer (dimension) stores specific types of information
   - This tensor has information about:
     - How many images we're processing (batch size)
     - The height and width of our features
     - Different types of features we extracted (channels)

{I believe the transformer is a DiT that we should talk more about later in improvements}

### The ControlNet U-Net Component

Let's first understand what a Hyper-Network is:

Imagine you have a very talented artist (our base model) who is amazing at drawing pictures based on descriptions. Now, what if you want them to draw pictures based on descriptions AND reference sketches? Instead of retraining the artist (which would be time-consuming and might make them forget their original skills), we give them an assistant (hyper-network) who understands sketches and can guide the artist.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/39.webp)

> Image taken from [HyperNetworks](https://arxiv.org/pdf/1609.09106)

This is exactly what ControlNet does! Instead of modifying our powerful Stable Diffusion model (the artist), it creates a smaller network (the assistant) that:

- Takes in our conditions (like depth maps, sketches)
- Processes them to extract useful information
- Guides the main model in using this information

Now, how does this guidance actually work?

Remember our U-Net from earlier? The ControlNet U-Net connects to it in a very simple way:

1. It processes our condition (like a depth map)
2. At each step in the U-Net, it adds its processed information to the main model's features
3. This addition subtly guides the main model without changing its learned knowledge

Think of it like this: Our artist (main model) is painting based on a description, and the assistant (ControlNet) is constantly whispering suggestions about the reference sketch, helping create an artwork that follows both the description AND the sketch.

For implementation of the original controlnet consider reading this [blog](https://huggingface.co/blog/controlnet), the original [repo](https://github.com/lllyasviel/ControlNet) and [paper](https://arxiv.org/pdf/2302.05543)

##### Inpainting

Let's quickly understand inpainting - it's like having an artist who can fill in missing or damaged parts of a photo while keeping everything else exactly the same.

![Image showing inpainting concept](/assets/blog_assets/demystifying_diffusion_models/41.webp)

> Image taken from ["Image inpainting based on sparse representations with a perceptual metric"](https://www.researchgate.net/publication/269588939_Image_inpainting_based_on_sparse_representations_with_a_perceptual_metric)

Looking at the image above:

- In (a), we have our original image of a person jumping
- In (b), we have a mask showing which part we want to keep (in white)
- In (c), we see how we can use this information to remove the person from the scene

The magic of inpainting in diffusion models is that it only applies the denoising process to the masked areas while keeping the rest of the image unchanged. Think of it like having an artist who:

1. Gets a photo with some parts erased
2. Looks at the surrounding context (trees, water, buildings)
3. Carefully fills in the erased parts to match perfectly with the rest

#### How Does It Actually Work?

Behind the scenes, inpainting uses a clever modification of our regular diffusion process. Let's understand the technical bits:

1. **Masked Diffusion**:

```python
   # x is our image, mask is 1 for areas to keep, 0 for areas to fill
   def add_noise_to_masked_area(x, mask, noise_level):
       noise = generate_gaussian_noise(x.shape)
       # Only add noise to masked areas (where mask = 0)
       noisy_x = x * mask + noise * (1 - mask)
       return noisy_x
```

We only add noise to the areas we want to fill in
Original parts of the image stay untouched
This preserves exact details in unmasked regions

Conditional Denoising

```python
def denoise_step(x_t, t, mask, original_x):
    # Predict noise for the entire image
    predicted_noise = model(x_t, t)

    # Only apply denoising to masked regions
    x_t_denoised = denoise(x_t, predicted_noise)
    result = x_t_denoised * (1 - mask) + original_x * mask
    return result
```

The model sees both the masked and unmasked regions
This helps it understand the context and maintain consistency
But updates only happen in masked areas

Additional Conditioning:

The model gets extra information:

The mask itself (where to fill)
The surrounding context (what to match)
Any text prompts (what to create)

This helps it generate content that fits seamlessly

Here's how these pieces work together in practice:

```python
def inpaint_image(original, mask, prompt):
    # Initialize with original image
    x = original.clone()

    # Step 1: Add noise only to masked region
    x_noisy = add_noise_to_masked_area(x, mask, noise_level=1.0)

    # Step 2: Gradually denoise while respecting mask
    for t in reversed(range(num_steps)):
        # Predict and denoise while maintaining original pixels
        x = denoise_step(x_noisy, t, mask, original)

        # Apply any prompt guidance
        x = apply_guidance(x, prompt, guidance_scale)

        # Ensure masked areas stay unchanged
        x = x * (1 - mask) + original * mask

    return x
```

This powerful technique is used for:

Removing unwanted objects from photos
Restoring damaged parts of old images
Extending images beyond their original boundaries
Creating variations of specific parts while keeping the rest intact

The best part? The same diffusion process we learned about earlier handles this naturally - it just needs to know which parts to leave alone and which parts to work on!

## LoRA (Low-Rank Adaptation)

Remember how earlier we talked about Dali learning the general idea of images rather than specific ones? Well, what if Dali wanted to learn a very specific style - like drawing in the style of Van Gogh, or creating anime characters? Teaching the entire model from scratch would be like making Dali relearn everything just to add one style. That would be quite inefficient!

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/40.webp)

> Image taken from ["LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS"](https://arxiv.org/pdf/2106.09685)

This is where LoRA comes in. Instead of retraining the whole model, LoRA only modifies a tiny but crucial part - the cross-attention mechanism. Think of cross-attention as Dali's ability to understand and connect your instructions with what he's drawing.
Let's break this down with an example:
When you say "A man with blue eyes", two important things happen:

The model needs to understand that "blue" and "eyes" go together
It needs to know where and how to draw these blue eyes in the image

The cross-attention mechanism handles these connections. It's like Dali's artistic intuition that knows when you say "blue eyes", you want the eyes to be blue, not the skin or the background.
LoRA works by making small adjustments to this cross-attention mechanism. Instead of teaching Dali completely new techniques, it's like giving him a small set of notes about a specific style. These notes are much smaller (often less than 1% of the original model size!) but can significantly change how Dali draws.
For example:

Want Dali to draw in anime style? Add a LoRA that tweaks how he interprets and draws faces and eyes
Want more realistic portraits? A LoRA can adjust how he handles skin textures and lighting
Want a specific artist's style? LoRA can modify how he approaches colors and brush strokes

What makes LoRA especially clever is how it achieves these changes. Instead of storing full-sized modifications, it uses something called "low-rank decomposition" - think of it as finding a clever shorthand way to write down the changes. This is why LoRA files are so small compared to full models, yet can create dramatic style changes.

### The Magical Wand (Variational Auto-Encoder)

This [video](https://www.youtube.com/watch?v=qJeaCHQ1k2w&t=1s) helped me immensely while writing this part.

Unfortunately for the both of us, This part too is very maths heavy. So again I will leave the intuition and derivation for the [maths section]() of the blog and just talk about the idea, show the equations and write out the code.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/13.webp)
The above image is actually what happens inside of an Variational Auto-Encoder but if you are anything like me. It probably doesn't make any sense.

So let's look at a simpler representation and come back to this when it makes more sense.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/15.webp)

On the left side we have something called the pixel space, these are images that humans understand.

The reason it is called pixel space is pretty self-explanatory. In a computer images are made up of pixels.

The encoder takes these pixels, Yes pixels. Not the images directly. Because if we take all the pixels of an image we can form a distribution. This is how such a distribution may look like only using red, green and blue.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/35.webp)

Now we take this distribution, pass it to the encoder which converts this into a latent space which has it's own distribution.

The reason we need it is quite simple.

An HD image can be of the size 1080x1920, which is equal to {calculate} pixels. But in the latent space a representation of the same image (a representation, or in simpler terms a replica. Not the original) can be in 128X128 pixels a reduction by a factor of {}X

Then the decoder returns this representation back to pixel image so we can see a picture. Which is more or less like the original one we started with.

The reason we do this is, This makes computation substantially easier, and it also lets Dali, Or The U-Net to have to do less computation to calculate the noise.

There is a difference between Auto-Encoders and Variational Auto-encoders. Which is explained in greater detail in the Maths section.

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/27.webp)

To expand on this idea, imagine a cluster of emojis—faces, animals, and other familiar icons—all grouped together in the pixel space because of their similar visual style.
Now, let’s take this to the latent space. We can see that the birds are grouped together, the emojis are cultured together in another space, with similar emojis together.
This demonstrates how the VAE learns to map out objects in the latent space, organizing them based on their visual or stylistic characteristics.

### Putting it all together

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/17.webp)

**A quicky Summary**

Before we start with the code, let's have a quick look at everything we have understood so far.

1. We begin with a prompt. (A delicious pizza)
2. This prompt is converted into a text embedding using a text encoder.
3. A latent noisy image is given to the U-net along with the text embeddings.
4. The U-Net predicts the noise in the latents.
5. The predicted noise is subtracted from the latent using the scheduler.
6. After many iterations, the denoised latent is decoded using the decoder to produce our final generated image.

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

Additionaly, we will begin with the same idea that we started with when we first talked about the diffusion process. To really drive the idea home.

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

Now a question that I had was how can a probability and distribution be equal, well the Left Hand Side (LHS) of equation (eq) 1 represents a Probability Density Function ([PDF]())

For the Right Hand Side (RHS) of eq 1. When we write $N(x; μ, σ²)$, we're specifying that $x$ follows a normal distribution with mean $μ$ and variance $σ²$

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

## Maths of Reverse diffusion process

### Reverse diffusion process

Now what we want to do is take a noisy image $x_t$ and get the original image $x_0$ from it. And to do that we need to do a reverse diffusion process.

Essentially we want to sample from $q(x_{t-1}|x_t)$, Which is quite tough as there can be millions of noisy images for actual images. To combat this we create an approximation (why do they work and how do they work in a minute) $p_\theta$ to approximate these conditional probabilities in order to run the _reverse diffusion process_.

Which can be represented as
$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$

Unfortunately it is tough to even sample from this approximate model because it is the same as our previous model, so we modify it by adding the original image $x_0$ to it as such.

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; {\color{Blue}{}\tilde{\mu}(x_t,x_0)}, {\color{red}{}\tilde{\beta}_t\mathbf{I}})$$

Now this is tractable (Exaplain what this word means), let us first understand the proof for how it is tractable. Later moving on to understand how they thought of this idea in the first place

Using Bayes' rule, we have:

$$
\begin{aligned}
q(x_{t-1}|x_t,x_0) &= \frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} & \text{[Bayes' Rule: } P(A|B) = \frac{P(B|A)P(A)}{P(B)} \text{]}
\end{aligned}
$$

Since $q(x)$ represents a normal distribution, we can expand using the general form of a normal distribution:

$$
\mathcal{N}(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

Therefore:

$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$

where $C(x_t,x_0)$ is some function not involving $x_{t-1}$, hence details can be omitted.

If we have a look again at our equation of normal distribution:

$$
\mathcal{N}(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

Expanding the squared term in the exponential:

$$
\exp(-\frac{1}{2\sigma^2}(x^2 - 2\mu x + \mu^2))
$$

The coefficient of $x^2$ is $\frac{1}{2\sigma^2}$, and the coefficient of $x$ is $-\frac{\mu}{\sigma^2}$. All other terms not involving $x$ can be collected into the normalization constant.

Therefore, comparing this with our previous equation:

$$
\color{red}{(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}})} = \frac{1}{2\sigma^2}
$$

and

$$
\color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0)} = \frac{2\mu}{\sigma^2}
$$

Following the above logice, the mean ($\tilde{\mu}_t(x_t,x_0)$) and variance ($\tilde{\beta}_t$) can be parameterized as follows (recall that $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$):

$$
\begin{aligned}
\tilde{\beta}_t &= 1/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= 1/(\frac{\alpha_t-\bar{\alpha}_t+\beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}) \\
&= \color{yellow}{}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t

\end{aligned}
$$

$$
\begin{aligned}
\tilde{\mu}_t(x_t,x_0) &= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)/(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}) \\
&= (\frac{\alpha_t}{\beta_t}x_t+\frac{\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t-1}}x_0)\color{yellow}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t \\
&= \frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\bar{\alpha}_{t-1}\beta_t}{1-\bar{\alpha}_t}x_0
\end{aligned}
$$

Thanks to the nice property, we can represent $x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)$ and replacing $x_0$ in the above equation we get:

$$\tilde{\mu}_t = \frac{1}{\alpha_t}(x_{t-1}-\frac{\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t)$$

This is great, we now have the mean in terms of $x_{t-1}$ and it does not depend on the original image $x_0$

> **Note**: Constants like 2,1/2,K etc have been omitted in many places as they do not hold much significance to the final equation

Now we have the mean, which can help us denoise the image. But we still need a training objective, using which the model gradually learns the approximation function.

### Training Loss ($L_t$)

Our original objective was to create an approcimate conditional probability distribution using which we could train a neural network to reverse the diffusion process.

$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$.

We wish to train $\boldsymbol{\mu}_\theta$ to predict $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t)$. Because $\mathbf{x}_t$ is available as input at training time.

we can instead reparameterize the Gaussian noise term to make it predict $\boldsymbol{\epsilon}_t$ from the input $\mathbf{x}_t$ at time step $t$:

$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))$

Thus $\mathbf{x}_{t-1} = \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$

The loss term $L_t$ is parameterized to minimize the difference from $\tilde{\boldsymbol{\mu}}$:

$L_t = \mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{1}{2\|\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\|_2^2}\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2\right]$

This scary looking equation is simply the Mean Squared Error for an [estimator](https://en.wikipedia.org/wiki/Mean_squared_error#Estimator)

Also given as,
$\text{MSE}(\hat{\theta}) = \mathbb{E}_{\theta}\left[(\hat{\theta} - \theta)^2\right]$

$= \mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{1}{2\|\boldsymbol{\Sigma}_\theta\|_2^2}\|\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t) - \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))\|^2\right]$

$= \mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\|\boldsymbol{\Sigma}_\theta\|_2^2}\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$

$= \mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\|\boldsymbol{\Sigma}_\theta\|_2^2}\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2\right]$

## Simplification

Ho et al. in ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) found that training the diffusion model works better with a simplified objective that ignores the weighting term:

$L_t^{\text{simple}} = \mathbb{E}_{t\sim[1,T],\mathbf{x}_0,\boldsymbol{\epsilon}_t}[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2]$

$= \mathbb{E}_{t\sim[1,T],\mathbf{x}_0,\boldsymbol{\epsilon}_t}[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t,t)\|^2]$

The final simple objective is:

$L^{\text{simple}} = L_t^{\text{simple}} + C$

where $C$ is a constant not depending on $\theta$.

Hence the equations simply become

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/37.webp)

Congratulations, you have a complete understanding of how we came to these equations now. Do not take from granted to how these equations were reached. I have furter added the mathematical backing to the ideas which led to the creation of these equations. Consider checking them out in the [Appendix]()

### Score Based Modeling

> **"**
> Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, stochastic gradient Langevin dynamics (Welling & Teh 2011) can produce samples from a probability density $p(x)$ using only the gradients $\nabla_x \log p(x)$ in a Markov chain of updates:
> $$x_t = x_{t-1} + \frac{\delta}{2}\nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t, \text{ where } \epsilon_t \sim \mathcal{N}(0,\mathbf{I})$$
> where $\delta$ is the step size. When $T \to \infty, \delta \to 0$, $x_T$ equals to the true probability density $p(x)$.
> Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima.\
> **"**

> From [Lil's Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics)

Before we continue further we need to understand Score based modeling.
This is a fascinating bridge between physics and machine learning!
First, let's understand what Langevin dynamics is trying to do. Imagine you're trying to find the lowest point in a hilly landscape while blindfolded. If you only walk downhill (like regular gradient descent), you might get stuck in a small valley that isn't actually the lowest point. Langevin dynamics solves this by occasionally taking random steps - like sometimes walking uphill - which helps you explore more of the landscape.
The key equation is:

$$x_t = x_{t-1} + \frac{\delta}{2}\nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t$$

Let's break this down piece by piece:

$x_t$ and $x_{t-1}$ represent our position at the current and previous steps.\
$\nabla_x \log p(x_{t-1})$ is the gradient term - it tells us which direction to move to increase the probability\
δ is our step size - how far we move in each step
ϵt is our random noise term, sampled from a normal distribution

The equation combines two behaviors:

A "deterministic" part: $\frac{\delta}{2}\nabla_x \log p(x_{t-1})$ which moves us toward higher probability regions\
A "random" part: $\sqrt{\delta}\epsilon_t$ which adds noise to help us explore

What makes this special is that when we run this process for a long time (T→∞) and with very small steps (δ→0), we're guaranteed to sample from the true probability distribution p(x). This is similar to how diffusion models gradually denoise images - they're following a similar kind of path, but in reverse!
The connection to standard gradient descent is interesting - regular SGD would only have the gradient term, but Langevin dynamics adds that noise term ϵt. This noise prevents us from getting stuck in bad local minima, just like how shaking a jar of marbles helps them settle into a better arrangement.

This is already immensely helpful, Because if we recall our previous discussion. Our biggest issue had been how do we create an approximate of our distribution because it is computationally expensive.

Now, here's the key insight of Langevin dynamics: When we take the gradient of log probability (∇log p(x)), we get something called the "_score function_". This score function has a special property - it points in the direction where the probability increases most rapidly.

Let's see why through calculus:
∇log p(x) = ∇(log p(x)) = (1/p(x))∇p(x)
This division by p(x) acts as an automatic scaling factor. When p(x) is small, it makes the gradient larger, and when p(x) is large, it makes the gradient smaller. This natural scaling helps our sampling process explore the probability space more efficiently.

What is P(x) though and why are we taking that. Traditionally in SGD do we not take, del(error)/del(weight)

In traditional SGD for neural networks, we're trying to minimize an error function (or loss function), so we use ∂(error)/∂(weight). We're trying to find the weights that make our predictions as accurate as possible.

But in Langevin dynamics, we're doing something fundamentally different. Here, p(x) represents a probability distribution that we want to sample from. Think of it this way:

Imagine you have a dataset of faces, and you want to generate new faces that look real. The probability p(x) would represent how likely it is that a particular image x is a real face. Areas of high p(x) would correspond to images that look like real faces, while areas of low p(x) would be images that don't look like faces at all.
So when we take ∇log p(x), we're asking: "In which direction should I move to make this image look more like a real face?"

This is why Langevin dynamics is particularly relevant to diffusion models. Remember how diffusion models start with noise and gradually transform it into an image? The ∇log p(x) term tells us how to modify our noisy image at each step to make it look more like real data.

To learn more about Score Based Modeling, consider reading this [blog by Yang Song](https://yang-song.net/blog/2021/score/)

### Mathematics of Guidance in Diffusion Models

Let's understand how we can mathematically guide our diffusion process to better follow our prompts. There are two main approaches: Classifier Guidance and Classifier-Free Guidance (CFG).

#### Classifier Guidance

First, recall that our diffusion process tries to predict and remove noise. The base noise prediction is:

$$\nabla_{x_t}\log q(x_t) = -\frac{1}{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)$$

When we want to incorporate class or prompt information $y$, we can write the score function for the joint distribution $q(x_t,y)$ as:

$$\nabla_{x_t}\log q(x_t,y) = \nabla_{x_t}\log q(x_t) + \nabla_{x_t}\log q(y|x_t)$$

This looks complex, but the idea is simple - we're combining:

- How the image should evolve naturally ($\nabla_{x_t}\log q(x_t)$)
- How it should change to better match our prompt ($\nabla_{x_t}\log q(y|x_t)$)

We can approximate this using a classifier $f_\phi(y|x_t)$:

$$\nabla_{x_t}\log q(x_t,y) \approx -\frac{1}{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t) + \nabla_{x_t}\log f_\phi(y|x_t)$$

This gives us our classifier-guided noise predictor:

$$\bar{\epsilon}_\theta(x_t,t) = \epsilon_\theta(x_t,t) - (1-\bar{\alpha}_t)\nabla_{x_t}\log f_\phi(y|x_t)$$

And to control how strongly we follow the classifier's guidance, we add a weight $w$:

$$\bar{\epsilon}_\theta(x_t,t) = \epsilon_\theta(x_t,t) - (1-\bar{\alpha}_t)w\nabla_{x_t}\log f_\phi(y|x_t)$$

#### Classifier-Free Guidance

CFG takes a different approach. Instead of training a separate classifier, we use the same model to predict noise both with and without conditioning.

Let's say we have:

- $\epsilon_\theta(x_t,t)$: Unconditional noise prediction
- $\epsilon_\theta(x_t,t,y)$: Conditional noise prediction (using prompt)

The gradient of an implicit classifier can be written as:

$$\nabla_{x_t}\log p(y|x_t) = \nabla_{x_t}\log p(x_t|y) - \nabla_{x_t}\log p(x_t)$$

$$= -\frac{1}{1-\bar{\alpha}_t}(\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t))$$

When we plug this into our guidance formula, we get:

$$\bar{\epsilon}_\theta(x_t,t,y) = \epsilon_\theta(x_t,t,y) + w(\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t))$$

Which simplifies to:

$$\bar{\epsilon}_\theta(x_t,t,y) = (w+1)\epsilon_\theta(x_t,t,y) - w\epsilon_\theta(x_t,t)$$

This final equation is what most modern diffusion models use. The weight $w$ (often called the CFG scale) controls how much we want our generation to follow the prompt:

- $w = 0$: Pure unconditional generation
- $w = 1$: Normal conditional generation
- $w > 1$: Increased adherence to the prompt

The beauty of CFG is its simplicity - we don't need a separate classifier, just the difference between conditional and unconditional predictions from our main model.

## stuff

There is still a lot of things that we can discuss like LDMs, Distillation etc. But now you have the essentially idea for majority of how SD maths work. So you can tackle it on your own, you can check more about it [here]()

## Maths of VAE

This part of blog takes inspiration from Lilian's blog on [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) and [Jaan's Blog](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/#glossary) on the topic.

To understand the mathematics behind VAE lets begin by setting some conventions first:

- $x$ represents a real input data point (an image)
- $z$ represents the latent space variables
- $p(z)$: The prior - what we think z should look like before seeing any data
- $p(x|z)$: The likelihood - how to generate data from the latent representation
- $p(z|x)$: The posterior - given data, what's its latent representation?

The central idea is we have a latent space $z$ and when we sample a data point $x^*_i$ from it, we want it to be as close as possible to the original datapoint $x_i$

This is represented mathematically as $p(z|x)$ (Think right to left: given x, we want to get z)

From Bayes' formula we can write:

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

We can calculate $p(x)$ as:

$$p(x) = \int p(x|z)p(z)dz$$
(Where did this come from?)

Unfortunately, this integral is computationally very expensive as we need to evaluate it over all possible configurations of latent variables.

Hence instead of calculating $p(z|x)$ directly, we calculate an approximation $q_{\lambda}(z|x)$, parameterized by $\lambda$. For example, if $q$ were Gaussian, $\lambda$ would represent the mean and variance of the latent variables for each datapoint: $\lambda_{x_i} = (\mu_{x_i}, \sigma^2_{x_i})$.

Now how do we know how close this approximation is to our original data distribution? That is where KL Divergence comes in.

> KLD deserves an entire blog on its own. Due to time & length constraints, I am recommending the following two blogs, which will help you get the whole idea ASAP: [Blog 1](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) & [Blog 2](https://colah.github.io/posts/2015-09-Visual-Information/).

Using KLD we can write:

$$KL(q_{\lambda}(z|x)||p(z|x)) = E_q[\log q_{\lambda}(z|x)] - E_q[\log p(x,z)] + \log p(x)$$

We want to find a value of $\lambda$ that minimizes the difference between these two distributions. Which we can write as:

$$q^*_{\lambda}(z|x) = \arg\min_{\lambda} KL(q_{\lambda}(z|x)||p(z|x))$$

Sadly, even this is intractable (hard to compute) as we again have our $p(z|x)$ term in it.

So we introduce ELBO (Evidence Lower BOund), which provides us with a tractable way to optimize our model. ELBO represents the lower bound on the evidence (log probability) of our observed data. It is written as:

$$ELBO(\lambda) = E_q[\log p(x,z)] - E_q[\log q_{\lambda}(z|x)]$$

Notice how we can combine this with the equation of KLD we wrote earlier and get the following formula:

$$\log p(x) = ELBO(\lambda) + KL(q_{\lambda}(z|x)||p(z|x))$$

By Jensen's inequality, the KLD is always greater than or equal to zero. Hence if we minimize KLD we will maximize ELBO. This makes our lives easier as we can indeed calculate the ELBO.

We can decompose the ELBO for a single data point as following:

$$ELBO_i(\lambda) = E_{q_{\lambda}(z|x_i)}[\log p(x_i|z)] - KL(q_{\lambda}(z|x_i)||p(z))$$

We can see that this is equivalent to our previous definition of ELBO through the following derivation:

Starting with our original ELBO:
$$ELBO(\lambda) = E_q[\log p(x,z)] - E_q[\log q_{\lambda}(z|x)]$$

We can expand $\log p(x,z)$ using the chain rule of probability:
$$\log p(x,z) = \log p(x|z) + \log p(z)$$

Substituting this back:
$$ELBO(\lambda) = E_q[\log p(x|z) + \log p(z)] - E_q[\log q_{\lambda}(z|x)]$$

Using the linearity of expectation:
$$ELBO(\lambda) = E_q[\log p(x|z)] + E_q[\log p(z)] - E_q[\log q_{\lambda}(z|x)]$$

Rearranging terms:
$$ELBO(\lambda) = E_q[\log p(x|z)] - (E_q[\log q_{\lambda}(z|x)] - E_q[\log p(z)])$$

The term in parentheses is exactly the KL divergence between $q_{\lambda}(z|x)$ and $p(z)$, giving us:
$$ELBO(\lambda) = E_q[\log p(x|z)] - KL(q_{\lambda}(z|x)||p(z))$$

Which is the same as our single-point ELBO formula.

## The Reparameterization Trick

![Image of super special artist](/assets/blog_assets/demystifying_diffusion_models/42.webp)

> Image taken from ["From Autoencoder to Beta-VAE by Lillian Weng"](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick)
> ]

There's a critical problem we haven't addressed yet. Remember our ELBO formula:

$$ELBO_i(\theta,\phi) = E_{q_{\theta}(z|x_i)}[\log p_{\phi}(x_i|z)] - KL(q_{\theta}(z|x_i)||p(z))$$

To optimize this, we need to calculate gradients through the entire process. However, sampling from $q_{\theta}(z|x_i)$ is a random operation, and we can't backpropagate through random sampling!

### The Problem

1. Our encoder outputs parameters for a probability distribution (usually mean $\mu$ and variance $\sigma^2$ for a Gaussian)
2. We sample $z$ from this distribution
3. We can't compute gradients through this random sampling step

### The Solution: Reparameterization Trick

Instead of directly sampling $z$, we:

1. Sample a random noise $\epsilon$ from a standard normal distribution $\mathcal{N}(0,1)$
2. Transform it using our distribution parameters:

$$z = \mu + \sigma \odot \epsilon$$

where $\odot$ represents element-wise multiplication.

This is equivalent to sampling from $\mathcal{N}(\mu, \sigma^2)$, but now the randomness is separated from the network parameters!

### Why This Works

- The random sampling ($\epsilon$) is now independent of the network parameters
- $\mu$ and $\sigma$ are direct outputs of our encoder network
- We can backpropagate through this transformation
- During backpropagation, $\epsilon$ is treated as a constant

In practice, our VAE now looks like this:

1. Encoder outputs $\mu$ and $\sigma$
2. Sample $\epsilon \sim \mathcal{N}(0,1)$
3. Compute $z = \mu + \sigma \odot \epsilon$
4. Feed $z$ into decoder
5. Calculate loss and backpropagate

This trick is what makes VAEs trainable in practice!

## Connecting to Neural Networks

Now that we understand the mathematical foundation, let's see how this translates to actual neural networks.

The key insight is that we can implement our probability distributions using neural networks:

1. **The Encoder Network (Inference Network)**

   - Takes input data $x$ and outputs parameters $\lambda$
   - Implements our approximate posterior $q_{\theta}(z|x)$
   - Parameters $\theta$ are the weights and biases of this network

2. **The Decoder Network (Generative Network)**
   - Takes latent variables $z$ and reconstructs the data
   - Implements our likelihood $p_{\phi}(x|z)$
   - Parameters $\phi$ are the weights and biases of this network

With these networks, we can rewrite our ELBO formula to include the network parameters:

$$ELBO_i(\theta,\phi) = E_{q_{\theta}(z|x_i)}[\log p_{\phi}(x_i|z)] - KL(q_{\theta}(z|x_i)||p(z))$$

This formula serves as our loss function (technically its negative):
$$Loss_i(\theta,\phi) = -ELBO_i(\theta,\phi)$$

Let's break down what each term means in practice:

1. **First Term**: $E_{q_{\theta}(z|x_i)}[\log p_{\phi}(x_i|z)]$

   - This is our reconstruction loss
   - How well can we reconstruct the input after encoding and decoding?
   - Think of it as "How close is the output to the input?"

2. **Second Term**: $KL(q_{\theta}(z|x_i)||p(z))$
   - This is our regularization term
   - Keeps our latent space well-behaved
   - Makes sure our encoded representations don't deviate too far from our prior

We optimize this loss function using stochastic gradient descent, which:

- Updates encoder parameters ($\theta$) to better approximate the posterior
- Updates decoder parameters ($\phi$) to better reconstruct the input

> Note: This is called variational EM (Expectation Maximization) because we're maximizing the expected log-likelihood of our data with respect to both sets of parameters.

And that's it! We've connected the dots between probability theory and neural networks. The VAE is essentially:

1. An encoder that compresses data into a well-behaved latent space
2. A decoder that reconstructs data from this latent space
3. A loss function that ensures both good reconstruction and well-structured latent representations

## Essential Papers in Diffusion Models

Here's a curated list of papers that shaped the field of diffusion models, arranged chronologically to show how the technology evolved.

### Foundational Papers

[**Auto-Encoding Variational Bayes**](https://arxiv.org/pdf/1312.6114) (2013)

- Introduced the VAE framework that later became crucial for latent diffusion models
- Key innovation: Reparameterization trick for training deep generative models
- Impact: Created the foundation for modern generative models

[**Denoising Diffusion Probabilistic Models (DDPM)**](https://arxiv.org/pdf/2006.11239) (2020)

- First major breakthrough in making diffusion models practical
- Key innovation: Showed how to train diffusion models efficiently using a simple noise prediction objective
- Impact: Set the basic framework that most modern diffusion models build upon

### Architecture Innovations

[**Denoising Diffusion Implicit Models (DDIM)**](https://arxiv.org/pdf/2010.02502) (2020)

- Solved the slow sampling problem in DDPMs
- Key innovation: Developed a non-Markovian sampling process that needs fewer steps
- Impact: Made diffusion models much faster and more practical for real applications

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/pdf/2112.10752) (2022)

- Introduced Stable Diffusion, making diffusion models accessible to everyone
- Key innovation: Performing diffusion in compressed latent space instead of pixel space
- Impact: Revolutionized the field by making high-quality image generation possible on consumer hardware

[**Scalable Diffusion Models with Transformers (DiT)**](https://arxiv.org/pdf/2212.09748) (2022)

- Reimagined diffusion model architecture using transformers
- Key innovation: Replaced U-Net with a transformer-based architecture
- Impact: Showed how transformer architectures could be effectively used for image generation

### Guidance and Control

[**CLASSIFIER-FREE DIFFUSION GUIDANCE**](https://arxiv.org/pdf/2207.12598) (2022)

- Solved the need for separate classifiers in guided diffusion
- Key innovation: Using the difference between conditional and unconditional generations for guidance
- Impact: Became the standard approach for controlling diffusion models

[**Diffusion Models Beat GANs on Image Synthesis**](https://arxiv.org/pdf/2105.05233) (2021)

- Proved diffusion models could outperform GANs
- Key innovation: Combined classifier guidance with architectural improvements
- Impact: Helped shift the field's focus from GANs to diffusion models

### Score-Based Methods

[**Generative Modeling by Estimating Gradients of the Data Distribution**](https://arxiv.org/pdf/1907.05600) (2019)

- Introduced score-based modeling perspective
- Key innovation: Connected noise-conditional score networks with diffusion
- Impact: Provided theoretical foundations for understanding diffusion models

[**SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS**](https://arxiv.org/pdf/2011.13456) (2020)

- Unified score-based models and diffusion models
- Key innovation: Continuous-time formulation of generative modeling
- Impact: Created a theoretical framework connecting different approaches

### Recent Advances

[**PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS**](https://arxiv.org/pdf/2202.00512) (2022)

- Addressed the slow sampling speed of diffusion models
- Key innovation: Student models that can generate high-quality samples in few steps
- Impact: Made diffusion models more practical for real-time applications

[**Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**](https://arxiv.org/pdf/2205.11487) (2022)

- Introduced Imagen, pushing boundaries of text-to-image generation
- Key innovation: Using large language models for better text understanding
- Impact: Showed the importance of strong text encoders in text-to-image models

[**Elucidating the Design Space of Diffusion-Based Generative Models**](https://arxiv.org/pdf/2206.00364) (2022)

- Comprehensive analysis of diffusion model design choices
- Key innovation: Systematic study of architecture and training decisions
- Impact: Provided practical guidelines for building better diffusion models

## The code [INCOMPLETE]

[Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)\
[BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html1)\
[MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)\
[Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)\
[torch.cat](https://pytorch.org/docs/main/generated/torch.cat.html)\
[ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)\
[Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)

## Understanding the metrics [INCOMPLETE]

This is interesting as well because... how do you tell a computer which is a good image and which is a bad image without actually doing a vibe check.

This really makes you appreaciate how the loss function was created doesnt it now!!

## Misc

- civitai
- comfyui
- https://stable-diffusion-art.com/author/andrew/ The blogs by this guy are absolutely mind boggling, if you are really intersted in this space. Check this out.

## How to help out

- **Share**: Consider clicking on any of the links below to share this blog, It reaches more people and they get to learn something new. Which make's me happy :), also. Consider tagging me from my socials.
- **Translate**: This blog is written in english, but there are a lot of people who do not speak english. I will be really glad if you are willing to translate this work in the language you prefer. (Consider raising a PR and I will attach your translation at the top)
- **Corrections & Feedback**: I am human, and I too can err. If you find I have made any mistakes, again feel free to raise a PR. Or contact me through my socials.

## Personal Note

Back in 2022 when I first saw Dall-e, It absolutely blew my mind. That was the moment I realised I really wanted to pursue ML.

I have put my heart and soul into writing this, I really hope it is something that can ignite the love for the amazing worls of ML in your heart as well.

I wrote this because when I first tried to understand Stable Diffusion I was bombarded with complex Code, Maths, ML Ideas that I did not understand. Heck Linear Regression was tough for me. This is something that I have written that I wish my past self had access to.

If you are reading this, then you most probably finished the blog. Hope you enjoyed it and learned something new. Thank you for reading!! -->
