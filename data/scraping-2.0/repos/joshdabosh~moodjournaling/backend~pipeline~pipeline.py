import io, os
import openai
from PIL import Image

from stability_sdk import client
from stability_sdk.client import generation

openai.api_key = "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"

STABILITY_API_KEY = "sk-tFClvqHpTQZSoBs84X6p1mAZDHWoZTekVwjCA51IBB6ZpYoV"
# STABILITY_API_KEY = "sk-u1Tugp4RvL2i0FenDJbjWXmjY1Ncfi0dDQo2xtjNRwurSFjR"

system_msg = 'You are trying to generate scrapbook photos from a short journal entry. Generate a Stable-Diffusion prompt which is composed of phrases which describe the journal entry. Also include a couple phrases that are somewhat related to the sentiments expressed in the journal entry. When you encounter topics about humans, replace them with analogies in nature. Each phrase should be separated by a comma and a single space. Do not include anything that is outside of the prompt you are responding with. The journal entry is as follows: '


def run_pipeline(user_msg):
    # return memoryview(b"abcdef")
    # return "\x89504e470d0a1a0a00000"
    # return memoryview(b"\xee\xb3\xf6\xfb\xad\xbe\xc0\x08\x9c\xc3\xcd\xad\x1e\xd3aU\x17|\x8d_\xc3\x162\xec\xc1(\xa2\xf3\xab*\xf3\x12\x04.S\xef\x9b\xff\xac\xe8\xfc\xc6\x8c\x04\xb79\xcbj\x82\x11\r\xab*\x8e\x8e\xe71\xa5\xaa\x05\xd9\x1a\x96\x9eCf\x05\x01z{\x0c\x88Y\xda\x02-{\x92\xc3\x96GX\xf8\xd5{\x93,\xb8\xacK\x1a\x1c\xa61\xa2\xd5G\x8b\x08\xdd\xd8\x842\xa3\xf8\x9b\xfb\xe1)R\xc4\xd0\xb2\xbe\xc9\x0f,\x1ep\xc4\xdc\xe6\x1d\xaaE\xcfF[\xe2\xac\x88\xb0\x93y\xc84\x17dL\'\xc3\xa0\xdb\xc5\xf4\x8bgutYVR2\xfd\xe7p\x8b")
    # user_msg = 'Today I did well on my exam, but I also have one tomorrow. I feel a bit relieved, while also anxious.'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    resp_prompt = response["choices"][0]["message"]["content"].replace("Stable-Diffusion prompt: ", "")

    stability_prompt = "nature, no humans, photo, paper texture, "

    stability_prompt += resp_prompt

    print(stability_prompt)

    stability_api = client.StabilityInference(
        host="grpc.stability.ai:443",
        key=STABILITY_API_KEY,
        verbose=True
    )

    answers = stability_api.generate(
        prompt=stability_prompt,
        steps=50, # defaults to 30 if not specified
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                # print(artifact.binary)
                # f = open("image1", "w")
                # f.write(io.BytesIO(artifact.binary))
                # f.close()
                return memoryview(artifact.binary)
