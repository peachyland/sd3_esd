from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel, SD3Transformer2DModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor, CLIPTextModelWithProjection, T5TokenizerFast, T5EncoderModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

def to_gif(images, path):

    images[0].save(path, save_all=True,
                   append_images=images[1:], loop=0, duration=len(images) * 20)

def figure_to_image(figure):

    figure.set_dpi(300)

    figure.canvas.draw()

    return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())

def image_grid(images, outpath=None, column_titles=None, row_titles=None):

    n_rows = len(images)
    n_cols = len(images[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols, n_rows), squeeze=False)

    for row, _images in enumerate(images):

        for column, image in enumerate(_images):
            ax = axs[row][column]
            ax.imshow(image)
            if column_titles and row == 0:
                ax.set_title(textwrap.fill(
                    column_titles[column], width=12), fontsize='x-small')
            if row_titles and column == 0:
                ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        image = figure_to_image(plt.gcf())
        plt.close()
        return image

def get_module(module, module_name):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 0:
        return module
    else:
        module = getattr(module, module_name[0])
        return get_module(module, module_name[1:])

def set_module(module, module_name, new_module):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)

def freeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = False

def unfreeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = True

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class StableDiffuser(torch.nn.Module):

    def __init__(self,
                scheduler='LMS'
        ):

        super().__init__()

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        
        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet")
        
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="feature_extractor")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")

        if scheduler == 'LMS':
            self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif scheduler == 'DDIM':
            self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        elif scheduler == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")    

        self.eval()

    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(self,
             iteration,
             latents,
             text_embeddings,
             guidance_scale=7.5
             ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration])

        # predict the noise residual
        noise_prediction = self.unet(
            latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
            (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=False,
                  pred_x0=False,
                  trace_args=None,                  
                  show_progress=True,
                  **kwargs):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:

                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration, 
                latents, 
                text_embeddings,
                **kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[iteration], latents)

            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    # @torch.no_grad()
    # def __call__(self,
    #              prompts,
    #              img_size=512,
    #              n_steps=50,
    #              n_imgs=1,
    #              end_iteration=None,
    #              generator=None,
    #              **kwargs
    #              ):

    #     assert 0 <= n_steps <= 1000

    #     if not isinstance(prompts, list):

    #         prompts = [prompts]

    #     self.set_scheduler_timesteps(n_steps)

    #     latents = self.get_initial_latents(n_imgs, img_size, len(prompts), generator=generator)

    #     text_embeddings = self.get_text_embeddings(prompts,n_imgs=n_imgs)

    #     end_iteration = end_iteration or n_steps

    #     latents_steps, trace_steps = self.diffusion(
    #         latents,
    #         text_embeddings,
    #         end_iteration=end_iteration,
    #         **kwargs
    #     )

    #     latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
    #     images_steps = [self.to_image(latents) for latents in latents_steps]

    #     for i in range(len(images_steps)):
    #         self.safety_checker = self.safety_checker.float()
    #         safety_checker_input = self.feature_extractor(images_steps[i], return_tensors="pt").to(latents_steps[0].device)
    #         image, has_nsfw_concept = self.safety_checker(
    #             images=latents_steps[i].float().cpu().numpy(), clip_input=safety_checker_input.pixel_values.float()
    #         )

    #         images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]

    #     images_steps = list(zip(*images_steps))

    #     if trace_steps:

    #         return images_steps, trace_steps

    #     return images_steps

class StableDiffuser3(torch.nn.Module):

    def __init__(self,
                scheduler='LMS'
        ):

        super().__init__()

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
        
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="tokenizer_2")
        self.tokenizer_3 = T5TokenizerFast.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="tokenizer_3")
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder_2")
        self.text_encoder_3 = T5EncoderModel.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder_3")
        
        # The UNet model for generating the latents.
        self.transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer")
        
        # self.feature_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler")

        self.eval()

    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.transformer.config.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.transformer.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        # latents = noise * self.scheduler.init_noise_sigma

        return noise

    def _get_t5_prompt_embeds(
        self,
        prompt = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device = None,
        dtype = None,
    ):
        device = device or self.transformer.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        device = None,
        clip_skip = None,
        clip_model_index: int = 0,
    ):
        device = device or self.transformer.device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt,
        prompt_2,
        prompt_3,
        device = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt = None,
        negative_prompt_2 = None,
        negative_prompt_3 = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        clip_skip = None,
        max_sequence_length: int = 256,
        lora_scale = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self.transformer.device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def get_text_embeddings(self, prompts, n_imgs):

        # text_tokens = self.text_tokenize(prompts)

        # text_embeddings = self.text_encode(text_tokens)

        # unconditional_tokens = self.text_tokenize([""] * len(prompts))

        # unconditional_embeddings = self.text_encode(unconditional_tokens)

        # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        unconditional_prompt = [""] * len(prompts)

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, negative_prompt=unconditional_prompt)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds

    def predict_noise(self,
             iteration,
             latents,
             text_embeddings,
             pooled_text_embeddings,
             guidance_scale=7.5
             ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        # latents = self.scheduler.scale_model_input(latents, self.scheduler.timesteps[iteration])

        # predict the noise residual
        t = self.scheduler.timesteps[iteration]
        timestep = t.expand(latent_model_input.shape[0])

        noise_prediction = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings,
                    pooled_projections=pooled_text_embeddings,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
            (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  pooled_text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=False,
                  pred_x0=False,
                  trace_args=None,                  
                  show_progress=True,
                  **kwargs):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:

                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration, 
                latents, 
                text_embeddings,
                pooled_text_embeddings,
                **kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[iteration], latents)

            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    # @torch.no_grad()
    # def __call__(self,
    #              prompts,
    #              img_size=512,
    #              n_steps=50,
    #              n_imgs=1,
    #              end_iteration=None,
    #              generator=None,
    #              **kwargs
    #              ):

    #     assert 0 <= n_steps <= 1000

    #     if not isinstance(prompts, list):

    #         prompts = [prompts]

    #     self.set_scheduler_timesteps(n_steps)

    #     latents = self.get_initial_latents(n_imgs, img_size, len(prompts), generator=generator)

    #     text_embeddings = self.get_text_embeddings(prompts,n_imgs=n_imgs)

    #     end_iteration = end_iteration or n_steps

    #     latents_steps, trace_steps = self.diffusion(
    #         latents,
    #         text_embeddings,
    #         end_iteration=end_iteration,
    #         **kwargs
    #     )

    #     latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
    #     images_steps = [self.to_image(latents) for latents in latents_steps]

    #     for i in range(len(images_steps)):
    #         self.safety_checker = self.safety_checker.float()
    #         safety_checker_input = self.feature_extractor(images_steps[i], return_tensors="pt").to(latents_steps[0].device)
    #         image, has_nsfw_concept = self.safety_checker(
    #             images=latents_steps[i].float().cpu().numpy(), clip_input=safety_checker_input.pixel_values.float()
    #         )

    #         images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]

    #     images_steps = list(zip(*images_steps))

    #     if trace_steps:

    #         return images_steps, trace_steps

    #     return images_steps
   
class FineTunedModel(torch.nn.Module):

    def __init__(self,
                 model,
                 train_method,
                 ):

        super().__init__()

        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}

        freeze(self.model)

        for module_name, module in model.named_modules():
            # print(module_name)
            # print(module.__class__.__name__)
            if 'transformer' not in module_name:
                continue
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'xattn':
                    if 'attn' not in module_name:
                        continue
                elif train_method == 'xattn-strict':
                    if 'attn' not in module_name or 'to_q' not in module_name or 'to_k' not in module_name:
                        continue
                elif train_method == 'noxattn':
                    if 'attn' in module_name:
                        continue 
                elif train_method == 'full':
                    pass
                else:
                    raise NotImplementedError(
                        f"train_method: {train_method} is not implemented."
                    )
                print(module_name)
                ft_module = copy.deepcopy(module)
                    
                self.orig_modules[module_name] = module
                self.ft_modules[module_name] = ft_module

                unfreeze(ft_module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())

        
    @classmethod
    def from_checkpoint(cls, model, checkpoint, train_method):

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        modules = [f"{key}$" for key in list(checkpoint.keys())]

        ftm = FineTunedModel(model, train_method=train_method)
        ftm.load_state_dict(checkpoint)

        return ftm

        
    def __enter__(self):

        for key, ft_module in self.ft_modules.items():
            set_module(self.model, key, ft_module)

    def __exit__(self, exc_type, exc_value, tb):

        for key, module in self.orig_modules.items():
            set_module(self.model, key, module)

    def parameters(self):

        parameters = []

        for ft_module in self.ft_modules.values():

            parameters.extend(list(ft_module.parameters()))

        return parameters

    def state_dict(self):

        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}

        return state_dict

    def load_state_dict(self, state_dict):

        for key, sd in state_dict.items():
            
            self.ft_modules[key].load_state_dict(sd)