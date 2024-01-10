from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings


def create_hf_pipeline(opt, env):
    hf_key = env['HUGGINGFACEHUB_API_TOKEN']

    model = AutoModelForCausalLM.from_pretrained(opt.model_name, trust_remote_code=True, load_in_4bit = opt.hf_load_in_4bit,token = hf_key, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name, trust_remote_code = True, token = hf_key)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=opt.hf_max_new_tokens, temperature=opt.temperature)
    llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceInstructEmbeddings()

    return llm, embeddings

