The Illustrated DeepSeek-R1
===========================

### A recipe for reasoning LLMs

[](https://substack.com/@jayalammar)

[Jay Alammar](https://substack.com/@jayalammar)

Jan 27, 2025

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_f328be6c0512.png)



](https://substackcdn.com/image/fetch/$s_!fn53!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F623a9dbf-c76e-438c-ba69-43ae9613ebbe_2930x1496.png)

DeepSeek-R1 is the latest resounding beat in the steady drumroll of AI progress. For the ML R&D community, it is a major release for reasons including:

Thanks for reading Language Models & Co.! Subscribe for free to receive new posts and support my work.

Subscribe

1.  It is an open weights model with smaller, distilled versions and
    
2.  It shares and reflects upon a training method to reproduce a reasoning model like OpenAI O1.
    

In this post, we’ll see how it was built.

_**Translations**: [Chinese](https://zhuanlan.zhihu.com/p/21175143007), [Korean](https://tulip-phalange-a1e.notion.site/DeepSeek-R1-189c32470be2801c94b6e5648735447d), [Turkish](https://gist.github.com/gsamil/0a5ca3bf44e979151e6c5d33345ede16) (Feel free to translate the post to your language and send me the link to add here)_

Contents:

*   Recap: How LLMs are trained
    
*   DeepSeek-R1 Training Recipe
    
*   1- Long chains of reasoning SFT Data
    
*   2- An interim high-quality reasoning LLM (but worse at non-reasoning tasks).
    
*   3- Creating reasoning models with large-scale reinforcement learning (RL)
    
    *   3.1- Large-Scale Reasoning-Oriented Reinforcement Learning (R1-Zero)
        
    *   3.2- Creating SFT reasoning data with the interim reasoning model
        
    *   3.3- General RL training phase
        
*   Architecture
    

Most of the foundational knowledge you need to understand how such a model works is available in our book, [Hands-On Large Language Models](https://github.com/handsOnLLM/Hands-On-Large-Language-Models).

[

![Hands-On Large Language Models](https://substackcdn.com/image/fetch/$s_!KUww!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeca4378-59bd-4c16-8753-a91cbb3bb939_400x525.jpeg "Hands-On Large Language Models")



](https://substackcdn.com/image/fetch/$s_!KUww!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeca4378-59bd-4c16-8753-a91cbb3bb939_400x525.jpeg)

[Official website](https://www.llm-book.com/) of the book. You can order the book on [Amazon](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961). All code is uploaded to [GitHub](https://github.com/handsOnLLM/Hands-On-Large-Language-Models).

Recap: How LLMs are trained


-----------------------------

Just like most existing LLMs, DeepSeek-R1 generates one token at a time, except it excels at solving math and reasoning problems because it is able to spend more time processing a problem through the process of generating thinking tokens that explain its chain of thought.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_a0fe7510950b.gif)



](https://substackcdn.com/image/fetch/$s_!aHqo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5280089e-8989-45d7-8194-93396b25557d_613x152.gif)

The following figure, from Chapter 12 of our book shows the general recipe of creating a high-quality LLM over three steps:

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_c6e0b36c812e.png)



](https://substackcdn.com/image/fetch/$s_!Eopd!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa354473-6ae0-4ae7-a20c-e858c804d6c4_1600x477.png)

1) The language modeling step where we train the model to predict the next word using a massive amount of web data. This step results in a base model.

2) a supervised fine-tuning step that makes the model more useful in following instructions and answering questions. This step results in an instruction tuned model or a supervised fine -tuning / SFT model.

3) and finally a preference tuning step which further polishes its behaviors and aligns to human preferences, resulting in the final preference-tuned LLM which you interact with on playgrounds and apps.

DeepSeek-R1 Training Recipe


-----------------------------

DeepSeek-R1 follows this general recipe. The details of that first step come from a [previous paper for the DeepSeek-V3 model](https://arxiv.org/pdf/2412.19437v1). R1 uses the _base_ model (not the final DeepSeek-v3 model) from that previous paper, and still goes through an SFT and preference tuning steps, but the details of how it does them are what's different.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_506477127d19.png)



](https://substackcdn.com/image/fetch/$s_!gcwD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc66dff5b-8332-4696-b484-b2ddb029b78c_854x234.png)

There are three special things to highlight in the R1 creation process.

### 1- Long chains of reasoning SFT Data

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_a89bfb8d3f4b.png)



](https://substackcdn.com/image/fetch/$s_!aCYU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F26136780-897d-4f64-b1e5-45936b6078dd_854x434.png)

This is a large number of long chain-of-thought reasoning examples (600,000 of them). These are very hard to come by and very expensive to label with humans at this scale. Which is why the process to create them is the second special thing to highlight

### 2- An interim high-quality reasoning LLM (but worse at non-reasoning tasks).

This data is created by a precursor to R1, an unnamed sibling which specializes in reasoning. This sibling is inspired by a third model called _R1-Zero_ (that we’ll discuss shortly). It is significant not because it’s a great LLM to use, but because creating it required so little labeled data alongside large-scale reinforcement learning resulting in a model that excels at solving reasoning problems.

The outputs of this unnamed specialist reasoning model can then be used to train a more general model that can also do other, non-reasoning tasks, to the level users expect from an LLM.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_d8d38cb37323.png)



](https://substackcdn.com/image/fetch/$s_!J73r!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4caea6a5-52a1-4651-8c71-4586c0637f3e_924x427.png)

### 3- Creating reasoning models with large-scale reinforcement learning (RL)

This happens in two steps:

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_56c5c8705954.png)



](https://substackcdn.com/image/fetch/$s_!paM3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45ca8c84-6eb6-4879-ab53-035174b17ce1_1620x700.png)

#### **3.1 Large-Scale Reasoning-Oriented Reinforcement Learning (R1-Zero)**

Here, RL is used to create the interim reasoning model. The model is then used to generate the SFT reasoning examples. But what makes creating this model possible is an earlier experiment creating an earlier model called _DeepSeek-R1-Zero_.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_f7c853337bd3.png)



](https://substackcdn.com/image/fetch/$s_!PgMh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F69b9f117-caa3-42fd-a949-dc6433990d26_1526x506.png)

R1-Zero is special because it is able to excel at reasoning tasks without having a labeled SFT training set. Its training goes directly from a pre-trained base model through a RL training process (no SFT step). It does this so well that it’s competitive with o1.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_c080c94b2d37.png)



](https://substackcdn.com/image/fetch/$s_!Bktd!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7b5c964f-b654-49b2-ab5a-5618b256ef99_1588x418.png)

This is significant because data has always been the fuel for ML model capability. How can this model depart from that history? This points to two things:

1- Modern base models have crossed a certain threshold of quality and capability (this base model was trained on 14.8 trillion high-quality tokens).

2- Reasoning problems, in contrast to general chat or writing requests, can be automatically verified or labeled. Let’s show this with an example.

##### Example: Automatic Verification of a Reasoning Problem

This can be a prompt/question that is a part of this RL training step:

> Write python code that takes a list of numbers, returns them in a sorted order, but also adds 42 at the start.

A question like this lends itself to many ways of automatic verification. Say we present this this to the model being trained, and it generates a completion:

*   A software linter can check if the completion is proper python code or not
    
*   We can execute the python code to see if it even runs
    
*   Other modern coding LLMs can create unit tests to verify the desired behavior (without being reasoning experts themselves).
    
*   We can go even one step further and measure execution time and make the training process prefer more performant solutions over other solutions — even if they’re correct python programs that solve the issue.
    

We can present a question like this to the model in a training step, and generate multiple possible solutions.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_16c699a06eff.png)



](https://substackcdn.com/image/fetch/$s_!lygC!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8edd9db2-a071-4bba-9d14-bbdb076d6355_798x444.png)

We can automatically check (with no human intervention) and see that the first completion is not even code. The second one is code, but is not python code. The third is a possible solution, but fails the unit tests, and the forth is a correct solution.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_5b882d784aff.png)



](https://substackcdn.com/image/fetch/$s_!sUyG!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f9645a0-b1fb-4753-942c-583504297c25_972x517.png)

These are all signals that can be directly used to improve the model. This is of course done over many examples (in mini-batches) and over successive training steps.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_aae9c0d4d512.png)



](https://substackcdn.com/image/fetch/$s_!3-5E!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b595e04-bd57-4f78-8c9b-ab37797e9b66_955x543.png)

These reward signals and model updates are how the model continues improving on tasks over the RL training process as seen in Figure 2 from the paper.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_db7c3761bf8d.png)



](https://substackcdn.com/image/fetch/$s_!MIn4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe48af6fa-8956-44b0-84cf-915e607f3b5e_1546x884.png)

Corresponding with the improvement of this capability is the length of the generated response, where the model generates more thinking tokens to process the problem.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_5b465018cfcd.png)



](https://substackcdn.com/image/fetch/$s_!ziuE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcd2b7d78-62ac-408c-8bd7-e14053bb8a46_1518x912.png)

This process is useful, but the R1-Zero model, despite scoring high on these reasoning problems, confronts other issues that make it less usable than desired.

> Although DeepSeek-R1-Zero exhibits strong reasoning capabilities and autonomously develops unexpected and powerful reasoning behaviors, it faces several issues. For instance, DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing.

R1 is meant to be a more usable model. So instead of relying completely on the RL process, it is used in two places as we’ve mentioned earlier in this section:

1- creating an interim reasoning model to generate SFT data points

2- Training the R1 model to improve on reasoning and non-reasoning problems (using other types of verifiers)

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_56c5c8705954.png)



](https://substackcdn.com/image/fetch/$s_!paM3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45ca8c84-6eb6-4879-ab53-035174b17ce1_1620x700.png)

#### 3.2 Creating SFT reasoning data with the interim reasoning model

To make the interim reasoning model more useful, it goes through an supervised fine-tuning (SFT) training step on a few thousand examples of reasoning problems (some of which are generated and filtered from R1-Zero). The paper refers to this as cold start data”

> **2.3.1. Cold Start**  
> Unlike DeepSeek-R1-Zero, to prevent the early unstable cold start phase of RL training from the base model, for DeepSeek-R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor. To collect such data, we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1- Zero outputs in a readable format, and refining the results through post-processing by human annotators.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_56a9303e1c93.png)



](https://substackcdn.com/image/fetch/$s_!E5Rw!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a89a9a0-c08f-430d-b135-7f012c2810ba_1824x586.png)

But wait, if we have this data, then why are we relying on the RL process? It’s because of the scale of the data. This dataset might be 5,000 examples (which is possible to source), but to train R1, 600,000 examples were needed. This interim model bridges that gap and allows to synthetically generate that extremely valuable data.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_aa7a68a28bff.png)



](https://substackcdn.com/image/fetch/$s_!rbw2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F857e61c8-03e7-4bc7-bcbe-ca182f60a70e_3300x1170.png)

If you’re new to the concept of Supervised Fine-Tuning (SFT), that is the process that presents the model with training examples in the form of prompt and correct completion. This figure from chapter 12 shows a couple of SFT training examples:

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_4d8a3b283c22.png)



](https://substackcdn.com/image/fetch/$s_!xHUV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b630dbc-aaa4-4c27-804b-542055b0f298_2264x1324.png)

#### 3.3 General RL training phase

This enables R1 to excel at reasoning as well as other non-reasoning tasks. The process is similar to the the RL process we’ve seen before. But since it extends to non-reasoning applications, it utilizes a helpfulnes and a safety reward model (not unlike the Llama models) for prompts that belong to these applications.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_65ea0e0dbeb6.png)



](https://substackcdn.com/image/fetch/$s_!HRLo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0e5f9acf-b4ca-4ec4-9731-4845c8fc5515_902x394.png)

Architecture


--------------

Just like previous models from the dawn of [GPT2](https://jalammar.github.io/illustrated-gpt2/) and [GPT 3](https://jalammar.github.io/how-gpt3-works-visualizations-animations/), DeepSeek-R1 is a stack of [Transformer](https://jalammar.github.io/illustrated-transformer/) decoder blocks. It’s made up 61 of them. The first three are dense, but the rest are mixture-of-experts layers (See my co-author Maarten’s incredible intro guide here: [A Visual Guide to Mixture of Experts (MoE)](https://substack.com/home/post/p-148217245)).

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_d6fc46d6b333.png)



](https://substackcdn.com/image/fetch/$s_!YRrM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F199f326e-9a8d-4a95-8574-4778d5b7657b_538x413.png)

In terms of model dimension size and other hyperparameters, they look like this:

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_df9ef747ce0e.png)



](https://substackcdn.com/image/fetch/$s_!CUyJ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ee664ae-a544-4e19-a145-0ae87acc43fa_916x481.png)

More details about the model architecture are presented in their two earlier papers:

*   [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437v1)
    
*   [DeepSeekMoE: Towards Ultimate Expert Specialization in](https://arxiv.org/pdf/2401.06066)
    
    [Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)
    

### Conclusion

With this, you should now have the main intuitions to wrap your head around the DeepSeek-R1 model.

[

![](images/https_3A_2F_2Fsubstack-post-media.s3.amazonaws.com_7c4338ffa1fd.png)



](https://substackcdn.com/image/fetch/$s_!a0EN!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fed7fd8c3-7654-497c-a8e2-1f2e7930992e_3302x1438.png)

If you felt needed a little more foundational information to understand this post, I’d suggest you pick up a copy of [Hands-On Large Language Models](https://www.llm-book.com/) or read it online on [O’Reilly](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/) and check it out on [Github](https://github.com/handsOnLLM/Hands-On-Large-Language-Models).

[

![Book Cover](https://substackcdn.com/image/fetch/$s_!PPT9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd7beb5f-e943-4d2d-8b4c-eb1e80231670_582x768.png "Book Cover")



](https://substackcdn.com/image/fetch/$s_!PPT9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd7beb5f-e943-4d2d-8b4c-eb1e80231670_582x768.png)

Other suggested resources are:

*   [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms?) by 
    
    [Maarten Grootendorst](https://open.substack.com/users/14309499-maarten-grootendorst?utm_source=mentions)
    
*   [DeepSeek R1's recipe to replicate o1 and the future of reasoning LMs](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1) by 
    
    [Nathan Lambert](https://open.substack.com/users/10472909-nathan-lambert?utm_source=mentions)
    
*   [A Visual Guide to Mixture of Experts (MoE)](https://substack.com/home/post/p-148217245) by 
    
    [Maarten Grootendorst](https://open.substack.com/users/14309499-maarten-grootendorst?utm_source=mentions)
    
*   Sasha Rush’s YouTube video [Speculations on Test-Time Scaling (o1)](https://www.youtube.com/watch?v=6PEJ96k1kiw)
    
*   Yannis Kilcher’s [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Paper Explained)](https://www.youtube.com/watch?v=bAWV_yrqx4w)
    
*   [Open R1](https://github.com/huggingface/open-r1) is the HuggingFace project to openly reproduce DeepSeek-R1
    
*   [Putting RL back in RLHF](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)
    
*   While reading this paper, the [Galactica paper from 2022](https://arxiv.org/abs/2211.09085) came to mind. It had a lot of great ideas including a dedicated thinking token.
