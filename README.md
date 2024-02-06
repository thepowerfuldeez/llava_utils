# LLaVA utils

Official [LLaVA repository](https://github.com/haotian-liu/LLaVA/) is great, but it lacks some features to make it easily accessible for quick research and prototyping.

In addition, latest LLaVA-1.6 doesn't support yet (as on 06.02.2024) straight inference code to run improved models.


This repository presents one-file utils file that helps to access official LLaVA repository from any source, without the need to setup gradio demo and run http model worker.

In addition, it provides correct image token number calculation, used for input prompt. 

## Quick start

### Install required libraries
After you install torch, run

```
pip install accelerate>=0.25.0 scikit-learn>=1.3.2 transformers>=4.37.2 bitsandbytes
```

```
pip install flash-attn --no-build-isolation
```

Install LLaVA as a library, ignore fixed outdated requirements
```
git clone https://github.com/haotian-liu/LLaVA/ && cd LLaVA/ && pip install -e . --no-dependencies
```

### Init model
```
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from llava_utils import prepare_prompt, prepare_image_args, generate

model_path = "liuhaotian/llava-v1.6-mistral-7b"
device = 'cuda'

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device,
    use_flash_attn=True,
    # you can also add load_4bit=True to reduce gpu memory usage, it's also x2 faster on RTX4090
)
```

### Run inference
```
images = [Image.open("LLaVA/images/llava_logo.png").resize((336, 336))]

# If you don't include <image>, it will be added automatically
text = "Describe this image, use no more than 20 words. <image>"
prompt = prepare_prompt(text, template_name="mistral_instruct")
image_args, num_image_tokens = prepare_image_args(model, image_processor, images, prompt)
print(f"Using {num_image_tokens=}")
out = generate(model, tokenizer, prompt, image_args, num_image_tokens, temperature=0.2, top_p=0.7)
```

```
> Using num_image_tokens=1152
> A cartoonish red creature with flames on its body, wearing glasses and standing on black feet. 
```


### To-do:
* Prepare simple interface for evaluating LLaVA-like models
* Support multi-turn dialogue using either huggingface apply_chat_formatting or LLaVA state class

