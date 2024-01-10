import os
from images import processor_image

import modal

stub = modal.Stub("aero-processor")

system_prompt = """
Write some Python code assuming these libraries are installed
("pandas", "numpy", "matplotlib", "seaborn", "moviepy", "Pillow",
"BeautifulSoup", "requests", "opencv", pytesseract). Assume no other libraries are installed.

This is where the code will fit in a large file:

filePath = "..."
[CODE TO GENERATE]

Your goal is only return [CODE TO GENERATE].
The file variable will exist prior to the generated code.

Do not use functions. All code generated should in the top scope.
Print the output if it's text or write the outputs as individual files in the directory /tmp/output.

Return only the [CODE TO GENERATE] without any markdown before and after, such as ```python or ```.
"""

main_prompt = """
Generate code to satisfy this requirement: {requirement}
Return only the [CODE TO GENERATE] without any markdown before and after, such as ```python or ```.
"""

MESSAGES = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": 'Generate code to satisfy this requirement: Take a "file" object that\'s an PNG. Convert it to a JPG.',
    },
    {
        "role": "assistant",
        "content": """
from PIL import Image

im = Image.open(filePath)
rgb_im = im.convert('RGB')
rgb_im.save('/tmp/output/converted.jpg')
""",
    },
]


@stub.function(image=processor_image)
def processor(file_info, instruction, openai_api_key, executor_url):
    import magic
    import openai
    import requests
    import json

    openai.api_key = openai_api_key

    [input_file_name, input_file_contents] = file_info

    # Figure out file type to annotate to the AI
    input_file_path = os.path.join("/tmp", os.path.basename(input_file_name))
    with open(input_file_path, "wb") as f:
        f.write(input_file_contents)
    input_file_type = magic.Magic(mime=True).from_file(input_file_path)

    input_description = f"Take the `filePath` string for a {input_file_type} file."

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=MESSAGES
        + [
            {
                "role": "user",
                "content": main_prompt.format(
                    requirement=input_description + instruction
                ),
            }
        ],
        temperature=1,
    )

    code = completion.choices[0].message.content
    print("Generated code: ")
    print("-----------------")
    print(code)
    print("-----------------")

    # TODO: Do 2nd generation

    # POST to the executor endpoint
    with open(input_file_path, "rb") as f:
        data = {
            "options": json.dumps(
                {"filename": input_file_name, "code1": code, "code2": ""}
            )
        }
        files = {"file": f}
        response = requests.post(executor_url, data=data, files=files)

        # Response will be a zip file that we'll return to the CLI
        return ("output.zip", response.content)
