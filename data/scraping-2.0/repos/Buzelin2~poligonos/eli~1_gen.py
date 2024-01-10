from transformers import AutoTokenizer
import transformers
import torch
from langchain.prompts import PromptTemplate

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

template = """
{text}

write another paragraph to continue the story that above:
"""

prompt = PromptTemplate(input_variables=["text"], template=template)

for i in range(11):  # Loop para iterar sobre os arquivos
    input_file_name = f'eli{i}.txt'
    output_file_name = f'eli{i}_gerado_gen.txt'
    
    with open(input_file_name, 'r') as poem:
        text = poem.read()

    poem_prompt = prompt.format(text=text)

    sequences = pipeline(
        poem_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    with open(output_file_name, 'w') as output_file:
        for seq in sequences:
            output_file.write(seq['generated_text'])
            print(f"Response for {input_file_name} saved to {output_file_name}")
