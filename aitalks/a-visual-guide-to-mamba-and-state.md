A Visual Guide to Mamba and State Space Models
==============================================

### An Alternative to Transformers for Language Modeling

[](https://substack.com/@maartengrootendorst)

[Maarten Grootendorst](https://substack.com/@maartengrootendorst)

Feb 19, 2024

##### Translations _\- [Korean](https://tulip-phalange-a1e.notion.site/05f977226a0e44c6b35ed9bfe0076839)  

_UPDATE 🔥_\- Now **with** **animations**!_

The Transformer architecture has been a major component in the success of Large Language Models (LLMs). It has been used for nearly all LLMs that are being used today, from open-source models like Mistral to closed-source models like ChatGPT.

To further improve LLMs, new architectures are developed that might even outperform the Transformer architecture. One of these methods is _Mamba_, a _State Space Model_.

![](https://substackcdn.com/image/fetch/$s_!cam8!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5547a54a-5f47-41c0-a2ee-2ba5bcceefb8_1920x1080.gif)


Mamba was proposed in the paper [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).[1](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state?utm_source=profile&utm_medium=reader2#footnote-1-141228095) You can find its official implementation and model checkpoints in its [repository](https://github.com/state-spaces/mamba).

_To see a **Table of Contents** (ToC), click on the stack of lines on the left-hand side._

Thanks for reading _Exploring Language Models_! Subscribe to receive new posts on _Gen AI_ and the book: **[Hands-On Large Language Models](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/)**

In this post, I will introduce the field of State Space Models in the context of language modeling and explore concepts one by one to develop an intuition about the field. Then, we will cover how Mamba might challenge the Transformers architecture.

In this visual guide, there are more than **50 custom visuals** to help you develop an intuition about Mamba and State Space Models!

Part 1: **The Problem with Transformers**
===========================================

To illustrate why Mamba is such an interesting architecture, let’s do a short re-cap of transformers first and explore one of its disadvantages.

A Transformer sees any textual input as a _sequence_ that consists of _tokens_.




A major benefit of Transformers is that whatever input it receives, it can look back at any of the earlier tokens in the sequence to derive its representation.

![](https://substackcdn.com/image/fetch/$s_!CBPZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2c01c75-1105-4aeb-a608-f00c85bbe5f7_1776x532.png)



The Core Components of Transformers
-------------------------------------

Remember that a Transformer consists of two structures, a set of encoder blocks for representing text and a set of decoder blocks for generating text. Together, these structures can be used for several tasks, including translation.

![](https://substackcdn.com/image/fetch/$s_!ZjET!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62a21d60-2e84-4c19-a6fb-d2eff501af1c_1776x952.png)

We can adopt this structure to create generative models by using only decoders. This Transformer-based model, _Generative Pre-trained Transformers_ (GPT), uses decoder blocks to complete some input text.

![](https://substackcdn.com/image/fetch/$s_!NpX9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58e51959-d4ef-4fa9-a1c7-dab0e5ca4dc0_1776x1012.png)


Let’s take a look at how that works!

A Blessing with Training…
---------------------------

A single decoder block consists of two main components, masked self-attention followed by a feed-forward neural network.

![](https://substackcdn.com/image/fetch/$s_!9-0q!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4b5af9c5-5266-4c2b-b583-b20a19f19fcc_1776x464.png)


Self-attention is a major reason why these models work so well. It enables an uncompressed view of the entire sequence with fast training.

So how does it work?

It creates a matrix comparing each token with every token that came before. The weights in the matrix are determined by how relevant the token pairs are to one another.

![](https://substackcdn.com/image/fetch/$s_!tr9q!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F167cfe80-2863-47c8-a969-cb2eeedbd353_1776x860.png)


During training, this matrix is created in one go. The attention between “_My_” and “_name_” does not need to be calculated first before we calculate the attention between “_name_” and “_is_”.

It enables **parallelization**, which speeds up training tremendously!

And the Curse with Inference!
-------------------------------

There is a flaw, however. When generating the next token, we need to re-calculate the attention for the _entire sequence_, even if we already generated some tokens.

![](https://substackcdn.com/image/fetch/$s_!-uKS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb66f1965-fc44-4a61-b9c6-912c8120ecad_2420x580.png)


Generating tokens for a sequence of length _L_ needs roughly _L²_ computations which can be costly if the sequence length increases.

![](https://substackcdn.com/image/fetch/$s_!zx4V!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F405074ed-aa8c-44b4-88a5-bae1dad0412e_2072x392.png)


This need to recalculate the entire sequence is a major bottleneck of the Transformer architecture.

Let’s look at how a “classic” technique, Recurrent Neural Networks, solves this problem of slow inference.

Are RNNs a Solution?
----------------------

Recurrent Neural Networks (RNN) is a sequence-based network. It takes two inputs at each time step in a sequence, namely the input at time step _**t**_ and a hidden state of the previous time step _**t-1**_, to generate the next hidden state and predict the output.

RNNs have a looping mechanism that allows them to pass information from a previous step to the next. We can “unfold” this visualization to make it more explicit.

![](https://substackcdn.com/image/fetch/$s_!VDgL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fddc71706-8da8-4c28-921b-675e9164c7ab_2404x872.png)


When generating the output, the RNN only needs to consider the previous hidden state and current input. It prevents recalculating all previous hidden states which is what a Transformer would do.

In other words, RNNs can do inference fast as it scales linearly with the sequence length! In theory, it can even have an _infinite context length_.

To illustrate, let’s apply the RNN to the input text we have used before.

![](https://substackcdn.com/image/fetch/$s_!1eET!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb2484541-f7b1-4950-b04f-5a3177596fbb_2228x808.png)


Each hidden state is the aggregation of all previous hidden states and is typically a compressed view.

There is a problem, however…

Notice that the last hidden state, when producing the name “_Maarten_” does not contain information about the word “_Hello_” anymore. RNNs tend to forget information over time since they only consider one previous state.

Although RNNs could be fast for both training and inference, they lacked the accuracy that the Transformer models could offer.

Instead, we look at State Space Models to efficiently use RNNs (and sometimes use convolutions).

Part 2: The **State Space Model (SSM)**
=========================================

A State Space Model (SSM), like the Transformer and RNN, processes sequences of information, like text but also signals. In this section, we will go through the basics of SSMs and how they relate to textual data.

What is a State Space?
------------------------

A State Space contains the minimum number of variables that fully describe a system. It is a way to mathematically represent a problem by defining a system's possible states.

Let’s simplify this a bit. Imagine we are navigating through a maze. The “_state space_” is the map of all possible locations (states). Each point represents a unique position in the maze with specific details, like how far you are from the exit.

The “_state space representation_” is a simplified description of this map. It shows where you are (current state), where you can go next (possible future states), and what changes take you to the next state (going right or left).

Although State Space Models use equations and matrices to track this behavior, it is simply a way to track where you are, where you can go, and how you can get there.

The variables that describe a state, in our example the X and Y coordinates, as well as the distance to the exit, can be represented as “_state vectors_”.

![](https://substackcdn.com/image/fetch/$s_!4TzR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F42c79eba-2559-4d9d-8999-bee33666f2e3_2364x736.png)

Sounds familiar? That is because embeddings or vectors in language models are also frequently used to describe the “state” of an input sequence. For instance, a vector of your current position (state vector) could look a bit like this:

![](https://substackcdn.com/image/fetch/$s_!m9J4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9ff8812a-64d2-4fc6-8e54-eb86222333b0_1496x444.png)

In terms of neural networks, the “state” of a system is typically its hidden state and in the context of Large Language Models, one of the most important aspects of generating a new token.

What is a State Space Model?
------------------------------

SSMs are models used to describe these state representations and make predictions of what their next state could be depending on some input.

Traditionally, at time _**t**_, SSMs:

*   map an input sequence _**x(t)**_ — (e.g., moved left and down in the maze)
    
*   to a latent state representation _**h(t)**_ — (e.g., distance to exit and x/y coordinates)
    
*   and derive a predicted output sequence _**y(t)**_ — (e.g., move left again to reach the exit sooner)
    

However, instead of using _discrete_ _sequences_ (like moving left once) it takes as input a _continuous_ _sequence_ and predicts the output sequence.

![](https://substackcdn.com/image/fetch/$s_!V1I6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5788c3e4-8794-4492-af87-3a45f7a6aa70_1992x624.png)

SSMs assume that dynamic systems, such as an object moving in 3D space, can be predicted from its state at time _**t**_ through two equations.

![](https://substackcdn.com/image/fetch/$s_!nrEV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F32401c6d-39b6-4619-a75e-6b33d3268bca_2520x388.png)

By solving these equations, we assume that we can uncover the statistical principles to predict the state of a system based on observed data (input sequence and previous state).

Its goal is to find this state representation _**h(t)**_ such that we can go from an input to an output sequence.

![](https://substackcdn.com/image/fetch/$s_!pAQE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ca5c7ae-3dbe-44d8-8b13-7f4dcc14a29b_2008x624.png)


These two equations are the core of the State Space Model.

The two equations will be referenced throughout this guide. To make them a bit more intuitive, they are **color-coded** so you can quickly reference them.

The **state equation** describes how the state changes (through _matrix A_) based on how the input influences the state (through _matrix B_).

![](https://substackcdn.com/image/fetch/$s_!SFe0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0876819d-8a46-4187-9826-14391bfd47b9_1796x624.png)

As we saw before, _**h(t)**_ refers to our latent state representation at any given time _**t**_, and _**x(t)**_ refers to some input.

The **output equation** describes how the state is translated to the output (through _matrix C_) and how the input influences the output (through _matrix D_).

![](https://substackcdn.com/image/fetch/$s_!alZ1!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb2e87708-9676-4a1c-b32c-d703026f64d9_1796x624.png)


> **NOTE**: Matrices _A_, _B_, _C_, and _D_ are also commonly refered to as _parameters_ since they are learnable.

Visualizing these two equations gives us the following architecture:


![](https://substackcdn.com/image/fetch/$s_!kKud!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc358439e-c507-49f1-ac2e-5dedaccc2a8b_1728x364.png)


Let’s go through the general technique step-by-step to understand how these matrices influence the learning process.

Assume we have some input signal _**x(t)**_, this signal first gets multiplied by _matrix B_ which describes how the inputs influence the system.


![](https://substackcdn.com/image/fetch/$s_!d8Ts!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcd6f8dae-2281-47af-8ba3-06bbdc594d1c_1956x360.png)

The updated state (akin to the hidden state of a neural network) is a latent space that contains the core “knowledge” of the environment. We multiply the state with _matrix A_ which describes how all the internal states are connected as they represent the underlying dynamics of the system.


![](https://substackcdn.com/image/fetch/$s_!Ckm-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1cedc98a-d200-4fe4-b311-6d68dcaa50af_1956x572.png)


As you might have noticed, _matrix A_ is applied before creating the state representations and is updated after the state representation has been updated.

Then, we use _matrix C_ to describe how the state can be translated to an output.


![](https://substackcdn.com/image/fetch/$s_!sgay!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8599487f-1023-4069-be7a-8056e63b0574_1956x572.png)



Finally, we can make use of _matrix D_ to provide a direct signal from the input to the output. This is also often referred to as a _skip-connection_.


![](https://substackcdn.com/image/fetch/$s_!Pjsq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcf79721f-5cef-44da-98c5-f63a7839ebc3_1956x756.png)


Since _matrix D_ is similar to a skip-connection, the SSM is often regarded as the following without the skip-connection.


![](https://substackcdn.com/image/fetch/$s_!Y8IQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ca1d511-7d31-42a0-9220-2fb85b256efd_1956x864.png)


We can also go through each step in more detail:

Going back to our simplified perspective, we can now focus on matrices _A_, _B_, and _C_ as the core of the SSM.

Which can be shown as:


![](https://substackcdn.com/image/fetch/$s_!r4iU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4e52f4f0-d7ad-453d-a741-6dfa4a998964_1728x352.png)


We can update the original equations (and add some pretty colors) to signify the purpose of each matrix as we did before.


![](https://substackcdn.com/image/fetch/$s_!HTOq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F55df8ede-3a16-4473-8ea9-872fe199d3a1_1904x676.png)


Together, these two equations aim to predict the state of a system from observed data. Since the input is expected to be continuous, the main representation of the SSM is a **continuous-time representation**.

From a Continuous to a Discrete Signal


----------------------------------------

Finding the state representation _**h(t)**_ is analytically challenging if you have a continuous signal. Moreover, since we generally have a discrete input (like a textual sequence), we want to discretize the model.

To do so, we make use of the _Zero-order hold technique._ It works as follows. First, every time we receive a discrete signal, we hold its value until we receive a new discrete signal. This process creates a continuous signal the SSM can use:

![](https://substackcdn.com/image/fetch/$s_!jd_L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7d9d77e2-1456-4eef-96b1-b943014bed1e_1920x638.gif)



How long we hold the value is represented by a new learnable parameter, called the _step size_ **∆**. It represents the resolution of the input.

Now that we have a continuous signal for our input, we can generate a continuous output and only sample the values according to the time steps of the input.

![](https://substackcdn.com/image/fetch/$s_!v9P_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe692b224-bc6b-4c8a-a4a7-ddd92173b25e_1918x630.gif)



These sampled values are our discretized output!

Mathematically, we can apply the Zero-order hold as follows:

![](https://substackcdn.com/image/fetch/$s_!_TjK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff6df4b59-6f76-4f13-a201-7b69e59df164_6200x1176.png)



Together, they allow us to go from a continuous SSM to a discrete SSM represented by a formulation that instead of a _function-to-function_, _**x(t)**_ → _**y(t)**_, is now a _sequence-to-sequence, **x**_**ₖ** → _**y**_**ₖ**:

![](https://substackcdn.com/image/fetch/$s_!QVa5!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc29cfbbb-ae41-4dc2-b899-9e0a81cba34d_1980x1012.png)


Here, matrices _A_ and _B_ now represent discretized parameters of the model.

We use _**k**_ instead of _**t**_ to represent discretized timesteps and to make it a bit more clear when we refer to a continuous versus a discrete SSM.

> **NOTE:** We are still saving the continuous form of _Matrix A_ and not the discretized version during training. During training, the continuous representation is discretized.

Now that we have a formulation of a discrete representation, let’s explore how we can actually _compute_ the model.

The Recurrent Representation


------------------------------

Our discretized SSM allows us to formulate the problem in specific timesteps instead of continuous signals. A recurrent approach, as we saw before with RNNs is quite useful here.

If we consider discrete timesteps instead of a continuous signal, we can reformulate the problem with timesteps:


![](https://substackcdn.com/image/fetch/$s_!XAz6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F83b70ba4-b068-44d1-8641-9b224d103c51_1980x548.png)



At each timestep, we calculate how the current input (_**Bx**_**ₖ**) influences the previous state (**Ahₖ₋₁**) and then calculate the predicted output (_**Ch**_**ₖ**).


![](https://substackcdn.com/image/fetch/$s_!yujB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb4d0412-87fb-4507-bedb-4793588bd465_2116x788.png)



This representation might already seem a bit familiar! We can approach it the same way we did with the RNN as we saw before.


![](https://substackcdn.com/image/fetch/$s_!l2el!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91ca51f7-9b9b-4f17-bccb-32a5a96f3339_2184x868.png)



Which we can unfold (or unroll) as such:


![](https://substackcdn.com/image/fetch/$s_!v2ZT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd1084e8a-a70d-450b-beb0-f18117ade5ed_2184x876.png)



Notice how we can use this discretized version using the underlying methodology of an RNN.

The Convolution Representation


--------------------------------

Another representation that we can use for SSMs is that of convolutions. Remember from classic image recognition tasks where we applied filters (_kernels_) to derive aggregate features:


![](https://substackcdn.com/image/fetch/$s_!twIZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F47f05950-bfad-4013-b854-679c9a47ada9_3216x2144.png)



Since we are dealing with text and not images, we need a 1-dimensional perspective instead:


![](https://substackcdn.com/image/fetch/$s_!GCKD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb943872f-de72-43e8-b2f1-cb8213f120a3_3216x1296.png)



Using techniques from a different field makes for an interesting pipeline:

The kernel that we use to represent this “filter” is derived from the SSM formulation:


![](https://substackcdn.com/image/fetch/$s_!_5jO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05049821-2352-4c04-8fb2-07fe15c20a9c_2620x824.png)



Let’s explore how this kernel works in practice. Like convolution, we can use our SSM kernel to go over each set of tokens and calculate the output:


![](https://substackcdn.com/image/fetch/$s_!-75V!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9007d03b-c1c9-4b37-8c83-f27bfe8318f4_2620x1080.png)



This also illustrates the effect padding might have on the output. I changed the order of padding to improve the visualization but we often apply it at the end of a sentence.

In the next step, the kernel is moved once over to perform the next step in the calculation:


![](https://substackcdn.com/image/fetch/$s_!wdkl!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F82ed71fb-f237-4173-bb23-bd1bf02ff123_2620x1080.png)



In the final step, we can see the full effect of the kernel:


![](https://substackcdn.com/image/fetch/$s_!soIJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7b984ef9-5f56-4bb3-bc21-f66d7ca3cf98_2560x1440.gif)



A major benefit of representing the SSM as a convolution is that it can be trained in parallel like Convolutional Neural Networks (CNNs). However, due to the fixed kernel size, their inference is not as fast and unbounded as RNNs.

The Three Representations


---------------------------

These three representations, _continuous_, _recurrent_, and _convolutional_ all have different sets of advantages and disadvantages:


![](https://substackcdn.com/image/fetch/$s_!FWd6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F682187d6-f402-44aa-8097-8a2e5b6179a7_2072x744.png)



Interestingly, we now have efficient inference with the recurrent SSM and parallelizable training with the convolutional SSM.

With these representations, there is a neat trick that we can use, namely choose a representation depending on the task. During training, we use the convolutional representation which can be parallelized and during inference, we use the efficient recurrent representation:


![](https://substackcdn.com/image/fetch/$s_!77LQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9c43c82d-9735-4d55-97bb-8ad6f504909e_1960x1008.png)



This model is referred to as the [Linear State-Space Layer (LSSL)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/05546b0e38ab9175cd905eebcc6ebb76-Abstract.html).[2](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state?utm_source=profile&utm_medium=reader2#footnote-2-141228095)

These representations share an important property, namely that of _**Linear Time Invariance**_ (LTI). LTI states that the SSMs parameters, _A_, _B_, and _C_, are fixed for all timesteps. This means that matrices _A_, _B_, and _C_ are the same for every token the SSM generates.

In other words, regardless of what sequence you give the SSM, the values of _A_, _B_, and _C_ remain the same. We have a static representation that is not content-aware.

Before we explore how Mamba addresses this issue, let’s explore the final piece of the puzzle, _matrix A_.

The Importance of Matrix _A_


------------------------------

Arguably one of the most important aspects of the SSM formulation is _matrix A_. As we saw before with the recurrent representation, it captures information about the _previous_ state to build the _new_ state.


![](https://substackcdn.com/image/fetch/$s_!PbYt!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b2a1eb3-72f3-4baf-ac89-4b16b6b8ef41_1840x778.gif)



In essence, _matrix_ _A_ produces the hidden state:

![](https://substackcdn.com/image/fetch/$s_!M-uM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F47635355-6b9a-4981-af3a-7ee6a12b87b3_1412x468.png)



Creating _matrix A_ can therefore be the difference between remembering only a few previous tokens and capturing every token we have seen thus far. Especially in the context of the Recurrent representation since it only _looks back_ _at the previous state_.

So how can we create _matrix A_ in a way that retains a large memory (context size)?

We use Hungry Hungry Hippo! Or [HiPPO](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)[3](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state?utm_source=profile&utm_medium=reader2#footnote-3-141228095) for **Hi**gh-order **P**olynomial **P**rojection **O**perators.


![](https://substackcdn.com/image/fetch/$s_!OpoS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F07985a64-fc26-4ee8-9ec2-c488e4cb709a_1492x488.png)



HiPPO attempts to compress all input signals it has seen thus far into a vector of coefficients.

It uses _matrix A_ to build a state representation that captures recent tokens well and decays older tokens. Its formula can be represented as follows:


![](https://substackcdn.com/image/fetch/$s_!cuQ1!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4bc7c768-7f0c-4983-a21e-70a4f587e6aa_2520x628.png)



Assuming we have a square _matrix A_, this gives us:


![](https://substackcdn.com/image/fetch/$s_!uc-k!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fef8f5de9-6448-43d5-9878-c8cd1d938b7c_1436x708.png)


Building _matrix A_ using HiPPO was shown to be much better than initializing it as a random matrix. As a result, it more accurately reconstructs _newer_ signals (recent tokens) compared to _older_ signals (initial tokens).

The idea behind the HiPPO Matrix is that it produces a hidden state that memorizes its history.

Mathematically, it does so by tracking the coefficients of a [Legendre polynomial](https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html) which allows it to approximate all of the previous history.[4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state?utm_source=profile&utm_medium=reader2#footnote-4-141228095)

HiPPO was then applied to the recurrent and convolution representations that we saw before to handle long-range dependencies. The result was [Structured State Space for Sequences (S4)](https://arxiv.org/abs/2111.00396), a class of SSMs that can efficiently handle long sequences.[5](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state?utm_source=profile&utm_medium=reader2#footnote-5-141228095)

It consists of three parts:

*   State Space Models
    
*   HiPPO for handling **long-range dependencies**
    
*   Discretization for creating **recurrent** and **convolution** representations
    

![](https://substackcdn.com/image/fetch/$s_!mnMc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb055ec5-f8c7-4862-ab88-f4fb38abf042_1892x844.png)


This class of SSMs has several benefits depending on the representation you choose (recurrent vs. convolution). It can also handle long sequences of text and store memory efficiently by building upon the HiPPO matrix.

> **NOTE**: If you want to dive into more of the technical details on how to calculate the HiPPO matrix and build a S4 model yourself, I would HIGHLY advise going through the [Annotated S4](https://srush.github.io/annotated-s4/).

Part 3: **Mamba - A Selective SSM**


=====================================

We finally have covered all the fundamentals necessary to understand what makes Mamba special. State Space Models can be used to model textual sequences but still have a set of disadvantages we want to prevent.

In this section, we will go through Mamba’s two main contributions:

1.  A **selective scan algorithm**, which allows the model to filter (ir)relevant information
    
2.  A **hardware-aware algorithm** that allows for efficient storage of (intermediate) results through _parallel scan_, _kernel fusion_, and _recomputation_.
    

Together they create the _selective SSM_ or _S6_ models which can be used, like self-attention, to create _Mamba blocks_.

Before exploring the two main contributions, let’s first explore why they are necessary.

What Problem does it attempt to Solve?


----------------------------------------

State Space Models, and even the S4 (Structured State Space Model), perform poorly on certain tasks that are vital in language modeling and generation, namely _the ability to focus on or ignore particular inputs_.

We can illustrate this with two synthetic tasks, namely **selective copying** and **induction heads**.

In the **selective copying** task, the goal of the SSM is to copy parts of the input and output them in order:


![](https://substackcdn.com/image/fetch/$s_!Z-Di!,w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1e086c20-18c0-4d5f-9efd-348ffd4f9858_1554x358.gif)


However, a (recurrent/convolutional) SSM performs poorly in this task since it is _**Linear Time Invariant**_**.** As we saw before, the matrices _A_, _B_, and _C_ are the same for every token the SSM generates.

As a result, an SSM cannot perform _content-aware reasoning_ since it treats each token equally as a result of the fixed A, B, and C matrices. This is a problem as we want the SSM to reason about the input (prompt).

The second task an SSM performs poorly on is **induction heads** where the goal is to reproduce patterns found in the input:


![](https://substackcdn.com/image/fetch/$s_!U5YG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fef90ea52-25e3-4949-b026-b6ae562fb52f_1496x578.gif)



In the above example, we are essentially performing one-shot prompting where we attempt to “teach” the model to provide an “_**A:**_” response after every “_**Q:**_”. However, since SSMs are time-invariant it cannot select which previous tokens to recall from its history.

Let’s illustrate this by focusing on _matrix B_. Regardless of what the input _**x**_ is, _matrix B_ remains exactly the same and is therefore independent of _**x**_:


![](https://substackcdn.com/image/fetch/$s_!4xmC!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1ee2bd7a-b99b-4871-8396-69cb7dd13cf5_1480x808.png)


Likewise, _A_ and _C_ also remain fixed regardless of the input. This demonstrates the _static_ nature of the SSMs we have seen thus far.


![](https://substackcdn.com/image/fetch/$s_!9uCD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0fea51ca-8458-4216-bf1c-c880a23504b4_1412x484.png)



In comparison, these tasks are relatively easy for Transformers since they _dynamically_ change their attention based on the input sequence. They can selectively “look” or “attend” at different parts of the sequence.

The poor performance of SSMs on these tasks illustrates the underlying problem with time-invariant SSMs, the static nature of matrices _A_, _B_, and _C_ results in problems with _content-awareness_.

Selectively Retain Information


--------------------------------

The recurrent representation of an SSM creates a small state that is quite efficient as it compresses the entire history. However, compared to a Transformer model which does no compression of the history (through the attention matrix), it is much less powerful.

Mamba aims to have the best of both worlds. A small state that is as powerful as the state of a Transformer:


![](https://substackcdn.com/image/fetch/$s_!e1s4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F84b8a71a-6310-416b-8622-e9166593171e_1540x464.png)




As teased above, it does so by compressing data selectively into the state. When you have an input sentence, there is often information, like stop words, that does not have much meaning.

To selectively compress information, we need the parameters to be dependent on the input. To do so, let’s first explore the dimensions of the input and output in an SSM during training:


![](https://substackcdn.com/image/fetch/$s_!S5cc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9376222e-b232-4458-a72c-bd741d8a031c_1436x500.png)



In a Structured State Space Model (S4), the matrices _A_, _B_, and _C_ are independent of the input since their dimensions _**N**_ and _**D**_ are static and do not change.

![](https://substackcdn.com/image/fetch/$s_!2pRJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3e93b701-df74-4954-be5c-c0d83779d3df_1412x532.png)



Instead, Mamba makes matrices _B_ and _C,_ and even the _step size_ **∆**_,_ dependent on the input by incorporating the sequence length and batch size of the input:


![](https://substackcdn.com/image/fetch/$s_!6R-P!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdccefffd-5712-45ab-9821-c794bce65d7d_1412x596.png)



This means that for every input token, we now have different _B_ and _C_ matrices which solves the problem with content-awareness!

> **NOTE**: Matrix _A_ remains the same since we want the state itself to remain static but the way it is influenced (through _B_ and _C_) to be dynamic.

Together, they _selectively_ choose what to keep in the hidden state and what to ignore since they are now dependent on the input.

A smaller _step size_ **∆** results in ignoring specific words and instead using the previous context more whilst a larger _step size_ **∆** focuses on the input words more than the context:

![](https://substackcdn.com/image/fetch/$s_!D8RX!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F06b21aab-aa32-450a-ae02-b976a2c9f9d8_2520x616.png)


The Scan Operation


--------------------

Since these matrices are now _dynamic_, they cannot be calculated using the convolution representation since it assumes a _fixed_ kernel. We can only use the recurrent representation and lose the parallelization the convolution provides.

To enable parallelization, let’s explore how we compute the output with recurrence:

![](https://substackcdn.com/image/fetch/$s_!59tH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F902ab53f-c4f4-4cfe-86c6-e528cf0c2d73_1444x490.gif)


Each state is the sum of the previous state (multiplied by _A_) plus the current input (multiplied by _B_). This is called a _scan operation_ and can easily be calculated with a for loop.

Parallelization, in contrast, seems impossible since each state can only be calculated if we have the previous state. Mamba, however, makes this possible through the _[parallel scan](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)_ algorithm.

It assumes the order in which we do operations does not matter through the associate property. As a result, we can calculate the sequences in parts and iteratively combine them:

![](https://substackcdn.com/image/fetch/$s_!5pAi!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F191fdabe-6b38-4e6b-a9f0-8240feef0a9d_1640x1100.png)


Together, dynamic matrices _B_ and _C_, and the parallel scan algorithm create the _**selective scan algorithm**_ to represent the dynamic and fast nature of using the recurrent representation.

Hardware-aware Algorithm


--------------------------

A disadvantage of recent GPUs is their limited transfer (IO) speed between their small but highly efficient SRAM and their large but slightly less efficient DRAM. Frequently copying information between SRAM and DRAM becomes a bottleneck.


![](https://substackcdn.com/image/fetch/$s_!wxqd!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9a1d4fa3-526d-488e-8768-c7208f018eb4_1728x300.png)



Mamba, like Flash Attention, attempts to limit the number of times we need to go from DRAM to SRAM and vice versa. It does so through _kernel fusion_ which allows the model to prevent writing intermediate results and continuously performing computations until it is done.


![](https://substackcdn.com/image/fetch/$s_!CeHw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc46b75de-9896-4be1-baac-29c07c68dfd4_1920x448.gif)



We can view the specific instances of DRAM and SRAM allocation by visualizing Mamba’s base architecture:


![](https://substackcdn.com/image/fetch/$s_!2LcC!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F724eceb1-4356-4ac5-b44e-f7fabce3b472_1728x580.png)



Here, the following are fused into one kernel:

*   Discretization step with _step size_ **∆**
    
*   Selective scan algorithm
    
*   Multiplication with _C_
    

The last piece of the hardware-aware algorithm is _recomputation_.

The intermediate states are not saved but are necessary for the backward pass to compute the gradients. Instead, the authors recompute those intermediate states _during_ the backward pass.

Although this might seem inefficient, it is much less costly than reading all those intermediate states from the relatively slow DRAM.

We have now covered all components of its architecture which is depicted using the following image from its article:


![](https://substackcdn.com/image/fetch/$s_!KiA_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffc840fb8-2e24-4103-95c8-afa306ce0cfc_2409x743.png)



**The Selective SSM.** Retrieved from: Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." _arXiv preprint arXiv:2312.00752_ (2023).

This architecture is often referred to as a _**selective SSM**_ or _**S6**_ model since it is essentially an S4 model computed with the selective scan algorithm.

The Mamba Block
-----------------

The _selective SSM_ that we have explored thus far can be implemented as a block, the same way we can represent self-attention in a decoder block.

![](https://substackcdn.com/image/fetch/$s_!yKpU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6d754eec-5284-4599-ad1a-e27358f12419_972x948.gif)



Like the decoder, we can stack multiple Mamba blocks and use their output as the input for the next Mamba block:


![](https://substackcdn.com/image/fetch/$s_!fR9z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc94d349d-8620-45a9-8095-7c27de8b7865_1660x1356.png)



It starts with a linear projection to expand upon the input embeddings. Then, a convolution before the _Selective SSM_ is applied to prevent independent token calculations.

The _Selective SSM_ has the following properties:

*   _Recurrent SSM_ created through _discretization_
    
*   _HiPPO_ initialization on matrix _A_ to capture _long-range dependencies_
    
*   S_elective scan algorithm_ to selectively compress information
    
*   _Hardware-aware algorithm_ to speed up computation
    

We can expand on this architecture a bit more when looking at the code implementation and explore how an end-to-end example would look like:

![](https://substackcdn.com/image/fetch/$s_!PCm9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa67d7341-9a43-4c67-aa88-6e802c0902ae_1660x2040.png)



Notice some changes, like the inclusion of normalization layers and softmax for choosing the output token.

When we put everything together, we get both fast inference and training and even unbounded context. Using this architecture, the authors found it matches and sometimes even exceeds the performance of Transformer models of the same size!

**Conclusion**
================

This concludes our journey in State Space Models and the incredible Mamba architecture using a selective State Space Model. Hopefully, this post gives you a better understanding of the potential of State Space Models, particularly Mamba. Who knows if this is going to replace the Transformers but for now, it is incredible to see such different architectures getting well-deserved attention!
