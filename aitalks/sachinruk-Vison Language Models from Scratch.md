Vison Language Models from Scratch
==================================

[Deep Learning](https://sachinruk.github.io/blog.html#category=Deep%20Learning)

[LLM](https://sachinruk.github.io/blog.html#category=LLM)

[huggingface](https://sachinruk.github.io/blog.html#category=huggingface)

[Multimodal](https://sachinruk.github.io/blog.html#category=Multimodal)

Using Small Language Models to with Small vision models to generate captions

Author

Sachin Abeywardana

Published

August 11, 2024

Introduction
----------------------------------------------------------------------------------------------------

In this blog, we will explore the process of creating a captioning model by leveraging a small language model (LLM). It is important to note that many of the Visual Language Models (VLMs) available today are not genuinely multimodal. With the exception of models like [Apple’s 4M](https://openreview.net/pdf?id=TegmlsD8oQ), there are no general-purpose multimodal models currently available. Instead, we typically train vision models to “speak” English by generating textual descriptions from visual data. However, most VLMs are not capable of generating image tokens directly. This tutorial focuses on the conversion of visual information into text, specifically in a vision-to-text context.

For this tutorial, we will utilize the `HuggingFaceTB/SmolLM-135M-Instruct` as the LLM and `mobilenetv4_conv_medium.e500_r256_in1k` as the vision model. It’s worth noting that the LLM we are using is approximately 60 times smaller than the Llama3-8B model.

Full code for this blog can be found [here](https://www.kaggle.com/code/sachin/vlms-from-scratch-smollm-mobilenet-caption-model) (please upvote if useful).

Data
------------------------------------------------------------------------------------

For training our captioning model, we will be utilizing the COCO (Common Objects in Context) dataset. This dataset comprises approximately 118,000 images, with each image being associated with five different captions. During each training epoch, we randomly select one caption per image. This approach helps to ensure variability in the training process, which is crucial for improving the model’s generalization ability.

Vision Model
----------------------------------------------------------------------------------------------------

To maintain simplicity (and also because I am GPU poor) we will be using the `smol-LM` as our language model (LLM) and integrating it with `mobilenet_v4` as the vision model. The integration process involves discarding the classification head of mobilenet\_v4, allowing us to connect the vision model to the LLM effectively. This is accomplished using the `model.forward_features` method instead of the standard `model.forward` in timm-based models.

    image_features = self.image_model.forward_features(images)
    image_features = einops.rearrange(image_features, "bs dim w h -> bs (w h) dim")
    encoder_outputs = self.projector(image_features)

In the above code snippet, we utilize einops to restructure the tensor by flattening the width and height dimensions into a single product while simultaneously reordering the tensor’s dimensions. The resulting product represents the number of tokens, which is significantly smaller than the original pixel dimensions due to the downsampling steps inherent in pretrained convolutional neural networks (CNNs).

However, there remains a dimensionality mismatch between the output of the vision model and the expected input for the captioning model. To address this, we introduce several linear layers, augmented with `GELU` activations, to project the vision model’s output into a space that is compatible with the LLM. The final projection is carefully designed to match the dimensionality required by the LLM.

    class Projection(nn.Module):
        def __init__(self, d_in: int, d_out: int, p: float = 0.5, last_layer=False) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_in, d_out, bias=False)
            self.linear2 = nn.Linear(d_out, d_out, bias=False)
            self.layer_norm = nn.Identity() if last_layer else nn.LayerNorm(d_out)
            self.drop = nn.Dropout(p)
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            embed1 = self.linear1(x)
            embed2 = self.drop(self.linear2(F.gelu(embed1)))
            embeds = self.layer_norm(embed1 + embed2)
            return embeds
    
    def projection_layers(d_in: int, d_out: int, layers: int) -> list[nn.Module]:
        return [Projection(d_in, d_in), nn.GELU(), nn.LayerNorm(d_in)] * (layers - 1) + [
            Projection(d_in, d_out, last_layer=True)
        ]

This block of code defines the projection layers that are responsible for transforming the vision model’s output into a format suitable for input into the language model. The linear layers, along with dropout and normalization, ensure that the features are appropriately scaled and regularized, thereby enhancing the model’s performance and stability.

Combining the LLM with image tokens (projections)combining-the-llm-with-image-tokens-projections)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The core of our approach lies in integrating image features with text tokens to construct a coherent prompt for the language model (LLM). The prompt follows this structure: `Caption or summarize the following image. <image_tokens>: <caption>`. To achieve this, we need to combine the image tokens, derived from the vision model, with the text tokens that represent the captions.

Text tokens are obtained by performing a simple lookup of the vocabulary embeddings, while image tokens are projections of the actual image features, as discussed earlier. We can access the LLM’s embeddings via its `get_input_embeddings()` function, which allows us to concatenate these embeddings effectively.

Here’s how we concatenate the vision and text embeddings:

    image_outputs = self.project_image_features(images)
    caption_embeddings = self.language_model.get_input_embeddings()(
        tokenized_captions.input_ids
    ).detach()
    device = images.device
    embeddings = torch.cat(
        [
            self.prepend_embeddings.to(device).expand(len(images), -1, -1),
            image_outputs,
            self.postpend_embeddings.to(device).expand(len(images), -1, -1),
            caption_embeddings,
        ],
        dim=1,
    )

In this code: - `self.project_image_features` refers to the vision model’s output after passing through the projection layers. - `caption_embeddings` are the embeddings of the tokenized captions, detached to prevent gradient updates during backpropagation. - The embeddings are concatenated in a specific order: a prepended segment (`prepend_embeddings`), the image outputs, an appended segment (`postpend_embeddings`), and finally, the `caption_embeddings`.

The inclusion of `prepend_embeddings` and `postpend_embeddings` is due to the specific chat template used for the LLM. The template we adopt is as follows: `"<|im_start|>user**\n**Caption or summarize the following image.<|im_end|>**\n**<|im_start|>assistant"`. The image tokens are inserted before the `<|im_end|>` token, necessitating the separation into prepended and appended embeddings. This is accomplished through the following code:

    input_ids = tokenizer(prepend_text, return_tensors="pt").input_ids
    eos_token_index = (
        input_ids[0] == tokenizer.eos_token_id
    ).nonzero(as_tuple=True)[0].item()
    text_embeddings = self.language_model.get_input_embeddings()(
        tokenizer(prepend_text, return_tensors="pt").input_ids
    ).detach()
    self.prepend_embeddings = text_embeddings[:, :eos_token_index]
    self.postpend_embeddings = text_embeddings[:, eos_token_index:]

This process ensures that the prompt structure is consistent and properly integrated with the image tokens, enabling the LLM to generate meaningful captions based on the visual input.

Loss Calculation
------------------------------------------------------------------------------------------------------------

Calculating the loss for our vision-language model (VLM) follows a straightforward approach, akin to traditional language models, but with particular attention to which parts of the sequence contribute to the loss. Since VLMs essentially function as language models with an additional vision module, the key challenge lies in ensuring that loss is computed only for relevant tokens—specifically, the tokens where the caption is generated. Tokens associated with image embeddings and other non-caption segments do not provide meaningful information about the relationship between language and images, and therefore, should be excluded from loss computation.

In the model’s initialization (`__init__`), we precompute two important tensors:

    self.attention_mask = torch.ones(1, text_embeddings.shape[1] + image_tokens)
    self.labels = torch.full((1, self.attention_mask.shape[1]), LABEL_MASK)

Here: - `text_embeddings` refers to the embeddings generated from the prompt (as discussed in the previous section). - `image_tokens` represents the number of tokens generated by the vision model, which remains constant for a given vision model. - `LABEL_MASK` is set to `-100`, a special value used by [PyTorch’s cross-entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) function to ignore specific tokens during loss calculation.

The final loss is computed by extending the labels and attention mask to include the caption tokens. This is achieved through the following code snippet:

    attention_mask = torch.cat(
        [
            self.attention_mask.to(device).expand(len(images), -1), 
            tokenized_captions.attention_mask
        ], 
        dim=1
    )
    labels = torch.cat(
        [
            self.labels.to(device).expand(len(images), -1), 
            tokenized_captions.input_ids.clone()
        ],
        dim=1,
    )
    labels[attention_mask == 0] = LABEL_MASK
    
    return self.language_model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        labels=labels,
    )

In this block:

*   The attention\_mask ensures that only the relevant parts of the sequence (i.e., the caption tokens) are attended to during training.
*   The labels tensor is used to define which tokens will contribute to the loss. Any token corresponding to an image or non-caption segment is masked out using `LABEL_MASK`.
*   Finally, we leverage the `input_embeds` argument of the language model to pass the combined embeddings (text + image tokens) instead of the traditional `input_ids`. This flexibility is crucial in accommodating the concatenated image embeddings.

The loss can then be easily retrieved using output.loss from the value that is in the `return` statement.

Caption Generation
----------------------------------------------------------------------------------------------------------------

The caption generation is similar to what we have encountered so far, and is shown below:

    def generate(self, images: torch.Tensor, generator_kwargs: dict[str, Union[int, float]]):
        image_outputs = self.project_image_features(images)
        device = images.device
        embeddings = torch.cat(
            [
                self.prepend_embeddings.to(device).expand(len(images), -1, -1),
                image_outputs,
                self.postpend_embeddings.to(device).expand(len(images), -1, -1),
            ],
            dim=1,
        )
        attention_mask = self.attention_mask.to(device).expand(len(images), -1)
        return self.language_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            **generator_kwargs
        )

In this function: - image\_outputs is obtained by projecting the image features through the vision model, just as in the training phase. - The embeddings tensor is constructed by concatenating the pre-defined prompt embeddings, image embeddings, and post-prompt embeddings, ensuring a seamless integration of visual and textual information. - The attention\_mask is similarly expanded to account for the entire input sequence, including the image tokens.

We pass these embeddings to the language model’s generate method. The generator\_kwargs allows flexibility in controlling the generation process by specifying parameters such as `num_beams`, `repetition_penalty`, and other decoding strategies. This flexibility is crucial for tuning the quality of generated captions.

By handling the embeddings in this manner, the model can generate captions directly from the combined visual and textual context, ensuring that the output is coherent and contextually relevant to the provided image.

Results
------------------------------------------------------------------------------------------

The results of our caption generation model are impressive, especially considering the relatively small size of the LLM (135 million parameters) and the computational resources used. The model was trained on a T4 GPU for approximately 8 hours, demonstrating that high-quality caption generation is achievable even with constrained hardware.

You can view the full set of results [here](https://wandb.ai/sachinruk/t5_captions/runs/9vx80dgd). Among the various caption generation tutorials I’ve conducted, this approach has yielded the best results so far. The captions generated by this model are not only accurate but also contextually rich, providing meaningful descriptions of the images.

However, there are still some areas for improvement. For instance, there are occasional artifacts in the generated captions, particularly extra tokens that need to be cleaned up. This issue likely stems from the template structure and the handling of embeddings, and it is an area that can be refined in future iterations of the model.

Below is an example of the model’s output: ![](images/H4aNMfO_26269124c317.png)

If you have any questions or suggestions, please feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/sachinabeywardana/).