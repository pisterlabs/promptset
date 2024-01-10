from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM():
  def __init__(self, model_type="chat"):
    self.model_type = model_type
    if self.model_type=="meditron":
      self.tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", token="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")

      lang_encoder_path = "epfl-llm/meditron-7b"

      self.model = AutoModelForCausalLM.from_pretrained(
              lang_encoder_path,
              local_files_only=False,
              trust_remote_code=True,
              load_in_4bit=True,
              token="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"
          )
      
    else : 
      self.model=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

  def predict(self, prompt):
    if self.model_type=="meditron":
      inputs = self.tokenizer(f"<question>: {query} <answer>", return_tensors="pt")
      output = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True, length_penalty=1.0)
      response = self.tokenizer.decode(output[0], skip_special_tokens=True)
      index_of_answer = response.find("<answer>")
      result = response[index_of_answer + len("<answer>"):] if index_of_answer != -1 else response
      index_of_closing_tag = result.find("answer")
      result_before_answer = result[:index_of_closing_tag] if index_of_closing_tag != -1 else result
      return result_before_answer
    else : 
      return self.model.predict(prompt)