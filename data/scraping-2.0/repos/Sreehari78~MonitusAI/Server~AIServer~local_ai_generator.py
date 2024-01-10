# ai_generator.py
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch




def generate_responses(input_text, faiss_vectorizer):
    generatedresponses = []
    print("Input Text:", input_text)
    def generate_response_for_medicine(input_text):
        model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
        model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=False,
        revision="main",
        )
        
        print("Input Text:", input_text)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        output = retrieve_info(input_text)
        print(output)
        prompt_template = """
        You are an intelligent chatbot that predicts adverse drug reactions. Given a prescription and patient details, predict possible adverse reactions.
        Prescription and Patient Information:
        {input}

        List of Adverse Drug Reactions in Similar Scenarios:
        {output}

        Output Format (in under 30 words):
        Drug Name only||Short description of possible interactions and allergies in a sentence||List of up to 10 adverse drug reactions||Risk level as H for high, M for Medium, and L for Low only (e.g., Aspirin||No interactions||Headache||M)
        The entire output should be a single sentence with no line breaks or extra spaces.
        """

        input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
        reply = model.generate(
            inputs=input_ids,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=512,
            repetition_penalty=1.1,

        )

        result = tokenizer.decode(reply[0])
        generatedresponses.append(result)
        print("\n\nGenerated Response: " + result + "\n")

    # Function for similarity search
    def retrieve_info(query):
        similar_response = faiss_vectorizer.similarity_search(query, k=5)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array

    for medicine in input_text:
        generate_response_for_medicine(medicine)
    generate_response_for_medicine(input_text)
    return generatedresponses

input_text = ["50 year old man prescribed 500mg paracetamol for 3 days"]
from vectorizer import load_faiss_vectorizer
faiss_vectorizer = load_faiss_vectorizer()
generate_responses(input_text, faiss_vectorizer)