import os
import math
import dotenv
import torch
import csv
import datetime
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# Configurations
dotenv.load_dotenv('../.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
model_id = '../Llama-2-7b-chat-hf'


# Configuration settings
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model and tokenizer
model_config = AutoConfig.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, quantization_config=bnb_config, use_auth_token=HF_ACCESS_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_ACCESS_TOKEN)
model.eval()

#load a hf text generation pipeline with the llama2 model
pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)


# model persona  template

template = """

Do not write any emojis.
The AI's Persona Description: 
<s>[INST] <<SYS>>
i like to party. my major is business. i am in college. i love the beach. i work part time at a pizza restaurant.
i am a business major but have a part time job
i am trying to get my ba in finance
no still in school work at pizza hut part time
i really hope they have a frat party again soon
i used to party a lot
it is fun i cant get enough
i am an undergrad in college
i love going to the beach
<</SYS>>


Do not write any emojis. Only respond with spoken text. Do not include terms like *smiling* *nods* *excitedly*

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %} 
"""

neg_template = """
Do not write any emojis.
The AI's Persona Description: 
<s>[INST] <<SYS>>
i dislike social gatherings and parties. my major is computer science. i am a graduate. i prefer staying indoors and find solace in solitude. i work full time at a tech company.
i am a computer science major and have a full time job
i have already earned my master's in computer science
i work full time and no longer involved in part-time jobs
i rarely attend social events or parties anymore
i value peace and quiet, and prefer spending time alone or in small gatherings
i have graduated and am currently working in the tech industry
i am not very fond of outdoor activities like going to the beach
<</SYS>>


Do not write any emojis. Only respond with spoken text. Do not include terms like *smiling* *nods* *excitedly*

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %} 
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

# Initialize the conversation chain -langchain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)

# Function to tokenize the persona
def tokenize_persona(template, tokenizer):
    # Extracting persona from the template
    start_idx = template.find("The AI") + len("The AI")
    end_idx = template.find("<</SYS>>")
    persona_text = template[start_idx:end_idx].strip()
    return tokenizer.encode(persona_text, return_tensors="pt")

# tokenize output of model
def tokenize_output(output,tokenizer):
    return tokenizer.encode(output, return_tensors="pt")


#combine the persona and input tokens
def combine_inputs(persona_tokens,input_tokens ):
    return torch.cat((persona_tokens, input_tokens), dim=-1)

def calculate_log_likelihood(input_tokens, output_tokens, model, tokenizer, temperature=1.0, normalize=False, debug=False):
    max_length = max(input_tokens.size(1), output_tokens.size(1))
    input_tokens = F.pad(input_tokens, pad=(0, max_length - input_tokens.size(1)), value=tokenizer.pad_token_id)
    output_tokens = F.pad(output_tokens, pad=(0, max_length - output_tokens.size(1)), value=tokenizer.pad_token_id)

    with torch.no_grad():
        outputs = model(input_ids=input_tokens, labels=output_tokens)

    logits = outputs.logits
    log_probs = F.log_softmax(logits / temperature, dim=-1)

    actual_log_probs = log_probs.gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)
    mask = (output_tokens != tokenizer.pad_token_id)
    actual_log_probs = actual_log_probs * mask

    log_likelihood = actual_log_probs.sum().item()

    if normalize:
        sequence_lengths = mask.sum(dim=-1).float()
        log_likelihood /= sequence_lengths.sum().item()

    return log_likelihood



def calculate_similarity_score(persona_tokens, output_tokens, model, tokenizer, debug=False):
    max_length = max(persona_tokens.size(1), output_tokens.size(1))

    persona_tokens = F.pad(persona_tokens, pad=(0, max_length - persona_tokens.size(1)), value=tokenizer.pad_token_id)
    output_tokens = F.pad(output_tokens, pad=(0, max_length - output_tokens.size(1)), value=tokenizer.pad_token_id)

    # Calculating cosine similarity
    persona_tokens = persona_tokens.cpu().numpy()
    output_tokens = output_tokens.cpu().numpy()
    similarity_score = cosine_similarity(persona_tokens, output_tokens)
    
    return similarity_score



def calculate_perplexity(input_tensor, output_tensor, model):
    max_length = max(input_tensor.size(1), output_tensor.size(1))
    
    if input_tensor.size(1) < max_length:
        padding_size = max_length - input_tensor.size(1)
        input_tensor = F.pad(input_tensor, pad=(0, padding_size), value=tokenizer.pad_token_id)
    
    if output_tensor.size(1) < max_length:
        padding_size = max_length - output_tensor.size(1)
        output_tensor = F.pad(output_tensor, pad=(0, padding_size), value=tokenizer.pad_token_id)

    with torch.no_grad():
        loss = model(input_ids=input_tensor, labels=output_tensor).loss
    
    return math.exp(loss)


def save_to_csv(data, filename="log_likelihood.csv"):
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H")
    filename = f"scores/loglikelihood_{current_datetime}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["user_msg", "ll" , "nll", "response"])
        writer.writerow(data)


def predict(message: str, history: list = None):
    output = conversation.predict(input=message)
    
    persona_tokens = tokenize_persona(template, tokenizer)
    persona_neg_tokens = tokenize_persona(neg_template, tokenizer)
    output_tokens = tokenize_output(output, tokenizer)  
    
    log_likelihood = calculate_log_likelihood(persona_tokens, output_tokens, model, tokenizer)
    neg_log_likelihood = calculate_log_likelihood(persona_neg_tokens, output_tokens, model, tokenizer)
    
    print('log likelihood', log_likelihood)
    print('neg_log likelihood', neg_log_likelihood)
    save_to_csv(["user_msg", message, "ll", log_likelihood , "nll", neg_log_likelihood, "response", output ])
    
    return output


interface = gr.ChatInterface(
    fn=predict
)
interface.launch(
    share=True
)
