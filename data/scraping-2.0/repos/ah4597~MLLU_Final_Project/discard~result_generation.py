from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_gptj6B(device, prompts):
    # baseline for the results
    """Get the pretrained GPT-J-6B model from hugging face libaries and froze the parameters then generate
       the result texts
    Args:
        device: 'cuda' if gpu is avalilable, otherwise 'cpu'
        prompts: the transformed prompts stored in a string
    Returns:
        complete_texts: the result for current prompt
    """

    # cpu/ gpu check
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
    if torch.cuda.is_available():
        model =  AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).cuda()
    else:
        model =  AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # freeze the model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # prompting the model
    input_ids = tokenizer((prompts), return_tensors="pt").input_ids
    torch.manual_seed(0)
    complete_texts = []
    for p in prompts:
        gen = model.generate(input_ids) # TODO: more parameters to be added
        complete_text = tokenizer.batch_decode(gen)
        complete_texts.append(complete_text)

    return complete_texts



def run_gpt3(prompts):
    """Start GPT-3 model and add prompt to it
    Args:
        prompts: the transformed prompts stored in a string
    Returns:
        completion: the result for prompts
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    engines = openai.Engine.list()
    completions = []
    for p in prompts:
        completion = openai.Completion.create(engine="davinci", prompt=prompts).choices[0].text
        completions.append(completion)
    return completions
