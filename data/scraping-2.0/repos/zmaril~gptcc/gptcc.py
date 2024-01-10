import sys
import os
import subprocess
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import openai
import platform

load_dotenv()

# Load the OpenAI API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_generate_assembly(code):
    host_arch = platform.machine()

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates assembly code for C code. You will receive a prompt with a C program. You must convert the program to assembly and return the assembly output in NASM syntax. Do not include any other text. The host and target architecture is {host_arch}. Use Mach-O 64-bit."},
        {"role": "user", "content": f"Generate assembly code for this C code:\n{code}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
    )

    assembly_code = response.choices[0].message.content.strip()
    return assembly_code

def assemble_and_link(assembly_code):
    with NamedTemporaryFile(mode="w", delete=False, suffix=".asm") as asm_file:
        asm_file.write(assembly_code)
        asm_file.flush()
        print("Generated assembly code:\n", assembly_code)

        obj_file = asm_file.name.replace(".asm", ".o")
        binary_file = asm_file.name.replace(".asm", "")

        # Assemble using NASM
        subprocess.run(["nasm", "-f", "macho64", asm_file.name, "-o", obj_file], check=True)

        # Link using clang
        subprocess.run(["clang", obj_file, "-o", binary_file], check=True)

        return binary_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gptcc <input_c_file>")
        sys.exit(1)

    input_c_file = sys.argv[1]

    with open(input_c_file, "r") as f:
        c_code = f.read()
        assembly_code = gpt_generate_assembly(c_code)
        binary_file = assemble_and_link(assembly_code)

    print(f"Generated binary saved at {binary_file}")
