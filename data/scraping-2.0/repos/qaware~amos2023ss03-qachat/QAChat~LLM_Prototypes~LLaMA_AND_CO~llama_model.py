# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Amela Pucic
# SPDX-FileCopyrightText: 2023 Hafidz Arifin
# SPDX-FileCopyrightText: 2023 Jesse Palarus

from enum import Enum

from huggingface_hub import hf_hub_download
from langchain import LlamaCpp


class LLaMAModel(Enum):
    LLaMA65B = ("CRD716/ggml-LLaMa-65B-quantized", "ggml-LLaMa-65B-q5_1.bin")
    LLaMA30B = ("Drararara/llama-30B-ggml", "ggml-model-q4_0.bin")
    LLaMA13B = ("Drararara/llama-13B-ggml", "ggml-model-q4_0.bin")
    LLaMA7B = ("Drararara/llama-7B-ggml", "ggml-model-q4_0.bin")
    Alpaca7B = ("Pi3141/alpaca-native-7B-ggml", " ggml-model-q8_0.bin")
    Alpaca13B = ("Pi3141/alpaca-native-13B-ggml", "ggml-model-q8_0.bin")
    Alpaca13BGPT4 = ("Pi3141/gpt4-x-alpaca-native-13B-ggml", "ggml-model-q8_0.bin")
    AlpacaLora30B = (
        "Pi3141/alpaca-lora-30B-ggml",
        "gpt4-alpaca-lora-30B.ggml.q5_1.bin",
    )
    AlpacaLora30BGPT4 = (
        "TheBloke/gpt4-alpaca-lora-30B-4bit-GGML",
        "gpt4-alpaca-lora-30B.ggml.q5_1.bin",
    )
    AlpacaLora65B = ("TheBloke/alpaca-lora-65B-GGML", "alpaca-lora-65B.ggml.q5_1.bin")
    Vicuna7B = ("Pi3141/vicuna-7b-v1.1-ggml", "ggml-model-q8_0.bin")
    Vicuna13B = ("Pi3141/vicuna-13b-v1.1-ggml", "ggml-model-q8_0.bin")
    Vicuna13BGPT4 = (
        "TheBloke/gpt4-x-vicuna-13B-GGML",
        "gpt4-x-vicuna-13B.ggmlv3.q5_1.bin",
    )
    StableVicuna13B = (
        "TheBloke/stable-vicuna-13B-GGML",
        "stable-vicuna-13B.ggml.q5_1.bin",
    )
    OASST30B = (
        "TheBloke/h2ogpt-oasst1-512-30B-GGML",
        " h2ogptq-oasst1-512-30B.ggml.q5_1.bin",
    )


def get_model(model_typ: LLaMAModel, n_ctx=512, max_tokens=256*3):
    path = hf_hub_download(repo_id="TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GGML", filename="wizardlm-13b-v1.1-superhot-8k.ggmlv3.q5_0.bin")

    return LlamaCpp(
        model_path=path,
        verbose=False,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    model = get_model(LLaMAModel.Vicuna13BGPT4)
    print("Answer:" +
          model.generate(["### USER: Was ist 2+2?\n### ASSISTANT:  "])
          .generations[0][0]
          .text
          )
