# from langchain.chains import LLMChain
# from langchain.llms import HuggingFaceHub
# from langchain.prompts import PromptTemplate

# question = "Who won the FIFA World Cup in the year 1994? "

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])


# # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
# repo_id = "google/flan-t5-xxl"


# llm = HuggingFaceHub(
#     repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
# )
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# print(llm_chain.run(question))


# pipe = pipeline("text-generation", model="sarvamai/OpenHathi-7B-Hi-v0.1-Base", torch_dtype=torch.bfloat16,
#                 device_map="auto", token='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')

# Usage
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    'sarvamai/OpenHathi-7B-Hi-v0.1-Base')
model = LlamaForCausalLM.from_pretrained(
    'sarvamai/OpenHathi-7B-Hi-v0.1-Base', torch_dtype=torch.bfloat16)


ll = ['अतिरिक्\u200dत उर्जा स्रोत विभाग', 'अल्पसंख्यक कल्याण एवं वक्फ', 'अवस्थापना एवं औद्योगिक विकास', 'आई.टी. एवं इलैक्ट्रानिक्स', 'आबकारी विभाग', 'आयुष', 'आवास एवं शहरी नियोजन', 'उच्\u200dच शिक्षा विभाग', 'उद्यान तथा खाद्य प्रसंस्करण', 'ऊर्जा विभाग', 'कृषि विपणन एवं विदेश व्\u200dयापार', 'कृषि विभाग', 'कृषि शिक्षा एवं अनुसंधान', 'खादी एवं ग्रामोद्योग', 'खाद्य एवं रसद विभाग', 'खाद्य सुरक्षा एवं औषधि प्रशासन', 'खेलकूद विभाग',
      'गृह एवं गोपन (पुलिस)', 'ग्रामीण अभियन्त्रण सेवा विभाग', 'ग्राम्\u200dय विकास विभाग', 'चिकित्सा शिक्षा', 'चिकित्सा स्वास्थ्य एवं परिवार कल्याण', 'चीनी उद्योग एवं गन्ना विकास', 'दिव्यांगजन सशक्तिकरण विभाग', 'दुग्\u200dध विकास विभाग', 'धर्मार्थ कार्य', 'नगर विकास तथा नगरीय रोजगार एवं गरीबी उन्मूलन', 'नमामि गंगे तथा ग्रामीण जलपूर्ति विभाग', 'नागरिक उड्डयन', 'नियोजन विभाग', 'परती भूमि विकास', 'परिवहन विभाग', 'पर्यटन विभाग', 'पर्यावरण,वन एवं जलवायु परिवर्तन विभाग', 'पशुधन विभाग', 'पिछड़ा वर्ग कल्\u200dयाण विभाग', 'पंचायती राज विभाग', 'प्राविधिक शिक्षा', 'बाल विकास पुष्टाहार विभाग', 'बेसिक शिक्षा विभाग', 'भारत सरकार से सम्बंधित विभाग', 'भूतत्\u200dव एवं खनिकर्म विभाग', 'मत्\u200dस्\u200dय विभाग', 'महिला कल्\u200dयाण', 'माध्\u200dयमिक शिक्षा विभाग', 'युवा कल्\u200dयाण विभाग', 'राजस्व एवं आपदा विभाग', 'राज्य कर विभाग', 'रेशम विकास', 'लोक निर्माण विभाग', 'वित्त', 'व्यवसायिक शिक्षा', 'श्रम विभाग', 'समाज कल्\u200dयाण विभाग', 'सहकारिता विभाग', 'सिंचाई, जल संसाधन', 'सूक्ष्म लघु एवं मध्यम उद्यम', 'संस्कृति', 'स्टाम्प एवं रजिस्ट्रेशन', 'हथकरघा एवं वस्त्रोद्योग उद्योग', 'होमगार्ड']


prompt = f"You are an expert label assigner. Your task is to assign one of the given labels for the given questions.\nQuestion: Mere yaha generator ki wajah se pollution ho raha hai.\nLabels:{ll} "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_new_tokens=50)
a = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)[0]
print(a)
