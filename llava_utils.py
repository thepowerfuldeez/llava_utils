"""
This small module is used to have easier interface to LLaVA official repository:
https://github.com/haotian-liu/LLaVA/
"""


import json
from typing import List
from threading import Thread

import torch
import numpy as np
from transformers import TextIteratorStreamer
from PIL import Image

from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)


def prepare_prompt(text: str, template_name: str = "mistral_instruct"):
    """template name is important, as it formats your text prompt in a format that selected model supports!
    Allowed values (most popular): mistral_instruct, chatml_direct
    llava-v1.6-34b uses chatml_direct
    llava-v1.6-7b uses mistral_instruct"""
    new_state = conv_templates[template_name].copy()
    if '<image>' not in text:
        text = text + '\n<image>'
    new_state.append_message(new_state.roles[0], text)
    new_state.append_message(new_state.roles[1], None)
    prompt = new_state.get_prompt()
    return prompt


def prepare_image_args(model, image_processor, images: List[Image], prompt: str):
    """images are list of PIL.Image instances"""
    if len(images) > 0:
        if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
            raise ValueError("Number of images does not match number of <image> tokens in prompt")
    
        image_sizes = [image.size for image in images]
        model_image_size = model.get_vision_tower().config.image_size
        # we add one extra patch (original image)
        num_image_patches = [int(np.ceil(h / model_image_size)) * int(np.ceil(w / model_image_size)) + 1
                             for (w, h) in image_sizes]
        images = process_images(images, image_processor, model.config)
    
        if type(images) is list:
            images = [image.to(model.device, dtype=torch.float16) for image in images]
        else:
            images = images.to(model.device, dtype=torch.float16)
    
        num_image_tokens = prompt.count(DEFAULT_IMAGE_TOKEN) * model.get_vision_tower().num_patches * sum(num_image_patches)
    else:
        images = None
        image_sizes = None
    image_args = {"images": images, "image_sizes": image_sizes}
    return image_args, num_image_tokens
    

@torch.inference_mode()
def generate(model, tokenizer, prompt: str, image_args: dict, num_image_tokens: int, print_output: bool = True, **params):
    """
    Example usage:
    
    text = "Describe this image, use no more than 20 words. <image>"
    prompt = prepare_prompt(text, template_name="mistral_instruct")
    image_args, num_image_tokens = prepare_image_args(model, images, prompt)
    out = generate(model, prompt, image_args, num_image_tokens, temperature=0.2, top_p=0.7, print_output=False)
    """
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, 'max_position_embeddings', 8192)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # steamer is responsible for tokenizer.decode to convert output ids into text
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    generation_kwargs = dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        **image_args
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        if print_output:
            print(new_text, end="")
        generated_text += new_text
        if (generated_text and stop_str) and generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]
    return generated_text
