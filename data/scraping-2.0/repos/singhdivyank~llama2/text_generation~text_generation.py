import os

# for UI
import gradio
# specify tensor dtypes
import torch

# framework to connect with LLM
from langchain import (
    PromptTemplate, 
    LLMChain
)
from langchain.llms import HuggingFacePipeline
# use pre-trained models
from transformers import (
    pipeline, 
    AutoTokenizer, 
    # for quantization
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoConfig
)

# huggingface API token
access_token = os.getenv('HF_KEY')

# Llama model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# memory quantization
BNB_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME, use_auth_token=access_token)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=access_token)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code = True,
    # specify model configurations
    config = MODEL_CONFIG,
    # save memory by quantization
    quantization_config = BNB_CONFIG,
    use_auth_token = access_token
)

def generate_text(company_name: str, product_name: str, product_description: str, instructions: str, target_audience: str) -> str:
    """
    function for text generation using Llama2 and Langchain

    Args:
        company_name (str): name of company
        product_name (str): name of product
        product_description (str): product description
        instructions (str): some instructions from the user
        target_audience (str): audience the answer is intended for
    
    Returns:
        content_generated (str): answer from LLM
    """

    content_generated = 'could not generate an answer'
    # template for the prompt
    template = """
    SYSTEM: You are a text-generation assistant for {company_name}. The generated text is intended for {target_audience}. \
        You are supposed to generate compelling and informative content for a product named '{product_name}'. \
        The product description is as follows: {product_description} 
    USER: {user_template}
    ASSISTANT:  
    """
    # define the prompt template for LangChain
    prompt_template = PromptTemplate(template=template, 
                                     input_variables=["company_name", "target_audience", 
                                                      "product_name", "product_description", 
                                                      "user_template"])

    try:
        # pipeline for text-generation
        hf_pipeline = pipeline(
            "text-generation",
            model = MODEL,
            tokenizer = TOKENIZER,
            # max tokens and temperature parameters
            model_kwargs = {'max_tokens': 512, 'temperature': 0.5}
        )
        print("Initialised pipeline")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("Initialised LLM")
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)
        print("Generating response ...")
        # run the LLM chain and specify all input variables
        content_generated = llm_chain.run({
            'company_name': company_name, 
            'target_audience': target_audience, 
            'product_name': product_name, 
            'product_description': product_description, 
            'user_template': instructions
        })
        print("Answer: ", content_generated)
    except Exception as error:
        print(f"error while generating response :: Exception :: {str(error)}")
    
    return content_generated

def gradio_interface(inputs: list=[gradio.Textbox(label = "company name", placeholder = "Enter name of company here ...", show_label = True, visible = True), 
                                   gradio.Textbox(label = "product name", placeholder = "Enter name of product here ...", show_label = True, visible = True), 
                                   gradio.Textbox(label = "product description", lines = 3, placeholder = "Enter product description here ...", show_label = True, visible = True),
                                   gradio.Textbox(label = "instructions", lines = 6, placeholder = "Enter the instructions here ...", show_label = True, visible = True),
                                   gradio.Textbox(label = "Target audience", placeholder = "Enter target audience details here ...", show_label = True, visible = True)
                                   ], 
                     outputs=gradio.Textbox(label = "Response", lines = 6, placeholder = "Generated text from Llama2....")) -> None:
    """
    function to render UI using Gradio

    Args:
        inputs (list): input components
        outputs (gradio.Textbox): ouput components
    """

    demo = gradio.Interface(fn=generate_text, inputs=inputs, outputs=outputs)
    demo.launch(share=False)

if __name__ == '__main__':
    gradio_interface()
