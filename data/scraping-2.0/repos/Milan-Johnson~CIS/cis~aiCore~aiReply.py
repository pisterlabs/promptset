from langchain import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large") #large x1
#tokenizer = T5Tokenizer.from_pretrained("t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
import chromadb
from aiCore.chromaDbBase import collection

def genrateAnswer(question):
  results = collection.query(
      query_texts=question,
      n_results=1
  )
  context = str(results["documents"])
  # Template
  template="Context = {context} answer the following question based on the context. {question}."

  multiple_input_prompt = PromptTemplate(
      input_variables=["context", "question"],
      template=template
  )
  #formatting
  formatted = multiple_input_prompt.format(context=context, question=question)
  #output
  print(formatted)
  my_text = formatted
  inputs = tokenizer(my_text, return_tensors="pt")
  outputs = model.generate(**inputs, \
                          min_length=20, \
                          max_new_tokens=512, \
                          length_penalty = 2, \
                          num_beams=16, \
                          temperature=0.9, \
                          no_repeat_ngram_size=2, \
                          #num_return_sequences 2,\
                          early_stopping=True)

  output_text_Flan_t5 = tokenizer.batch_decode(outputs, \
                                              skip_special_tokens=True)
  print (output_text_Flan_t5)
  return output_text_Flan_t5

# question = "college libriary ?"
# print(genrateAnswer(question))