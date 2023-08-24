import argparse
import time
import warnings

import torch
import torch.nn as nn
import transformers
from basaran.model import StreamModel
from conversation import get_default_conv_template
from py3nvml import py3nvml
from rich.live import Live
from rich.spinner import Spinner

from utils import clean_text, seed_everything

warnings.simplefilter("ignore", UserWarning)


def get_max_memory():
    py3nvml.nvmlInit()

    max_memory = {}
    for idx in range(py3nvml.nvmlDeviceGetCount()):
        handle = py3nvml.nvmlDeviceGetHandleByIndex(idx)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        p = 0.7 if idx == 0 else 1
        max_memory[idx] = f"{int((info.total >> 30) * p)}GiB"

    return max_memory

class StreamModelImproved(StreamModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

class StreamModelImproved(StreamModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        for resp in super().__call__(*args, **kwargs):
            if "###" in resp["text"]:
                break
            if "</s>" in resp["text"]:
                break

            yield clean_text(resp["text"])


def load_model(
    name: str = "vicuna-13b",
    cache_dir: str = "/opt/models",
):
    AVAILABLE_MODELS = {
        "alpaca-lora-7b": f"{cache_dir}/models--deepsquare--alpaca-lora-7b",
        "alpaca-lora-13b": f"{cache_dir}/models--deepsquare--alpaca-lora-13b",
        "vicuna-13b": f"{cache_dir}/vicuna-13b-1.1",
        "oasst-pythia-13b": f"{cache_dir}/oasst-sft-4-pythia-12b-epoch-3.5",
        "oasst-llama-30b": f"{cache_dir}/oasst-sft-6-llama-30b",
    }

    if name not in AVAILABLE_MODELS:
        raise AssertionError(
            f"Invalid model name: {name}. "
            + f"Available: {', '.join(AVAILABLE_MODELS.keys())}."
        )
    model_path = AVAILABLE_MODELS[name]

    print(f"*** Loading {name} ***")

    start = time.time()

    if "vicuna" in name or "llama" in name or "alpaca" in name:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    max_memory = get_max_memory()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        cache_dir=cache_dir,
    )

    if len(max_memory) > 1:
        model = nn.DataParallel(model).module

    model.eval()

    print(f"*** Loading done! (in {time.time() - start:.2f} secs) ***\n")

    return (
        StreamModelImproved(model, tokenizer),
        lambda: get_default_conv_template(name).copy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", default="vicuna-13b")
    parser.add_argument("-i", "--instructions", action="extend", nargs="+")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    model, new_conv = load_model(args.model_name)

    conv = new_conv()
    model_params = {"max_tokens": args.max_tokens, "temperature": args.temperature}

    for instruction in args.instructions:
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        for d in conv.to_openai_api_messages():
            for key, value in d.items():
                print(f"\t{key}: {value}")
            print()
        if args.stream:
            spinner = Spinner("dots")
            with Live(spinner, refresh_per_second=120):
                text = ""
                for out in model(conv.get_prompt(), **model_params):
                    text += out
                    spinner.update(text=text.replace("</s>", "").replace("<s>", ""))

                conv.messages[-1][1] = text
        else:
            start = time.time()

            text = ""
            for out in model(conv.get_prompt(), **model_params):
                if out["text"] == "</s>":
                    break
                text += out["text"]
            end = time.time()

            print(
                f"### Instruction:\n\n{instruction}\n\n"
                + f"### Response (in {end - start:.2f} secs):\n\n{text}\n"
            )
        print("")
