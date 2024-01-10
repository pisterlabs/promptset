import json
from jsonschema import validate
import torch
from langchain import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from langchain import PromptTemplate, LLMChain
from constants import assessment_template, template, question_template
from data_model import initialize_object_from_schema, question_schema, schema


def convert_to_json_and_validate(data, schema):
    json_data = json.dumps(data)

    try:
        validate(instance=data, schema=schema)
        print("JSON data is valid.")
    except Exception as e:
        print(f"JSON data is invalid: {e}")

    return json_data


def categorize_risk(data):
    total_questions = len(data.get('questions', []))
    one_third = total_questions / 3
    two_thirds = 2 * one_third

    count = 0
    for question in data.get('questions', []):
        answer = question.get('answer', '').strip().lower()
        if answer == 'yes':
            count += 1

    if count < one_third:
        return 'Low'
    elif count < two_thirds:
        return 'Average'
    else:
        return 'High'


def load_llm_model(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    use_quantization=True
):

    quantization_config = None

    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder="/tmp",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer


def setup_llm_pipeline(model, tokenizer):
    pp = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=3000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pp)


def unload_llm_model(model):
    del model
    torch.cuda.empty_cache()



def analyse_document(llm, initial_object, base_questions, specific_questions, context_p):  # noqa: E501 

    print(initial_object)

    for key, question in tqdm(base_questions.items()):

        prompt = PromptTemplate(template=template, input_variables=["question", "context"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = llm_chain.run({"question": question, "context": context_p})
        initial_object["general_info"][key] = response

        if key == "treatment_plan":
            prompt = PromptTemplate(template=assessment_template, input_variables=["context"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run({"context": response})

            initial_object["general_info"]["meta"]["treatment_assessment"] = response

    for question in tqdm(specific_questions):
        question_item = initialize_object_from_schema(question_schema)

        prompt = PromptTemplate(template=question_template, input_variables=["question","context"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = llm_chain.run({"question": question, "context": context_p})

        question_item["question"] = question
        question_item["justification"] = response.split("$$SEPARATOR$$")[1].replace("Answer:", "")
        question_item["answer"] = response.split("$$SEPARATOR$$")[0].replace("Answer:", "").replace("\n", "")

        initial_object["questions"].append(question_item)

    return initial_object

