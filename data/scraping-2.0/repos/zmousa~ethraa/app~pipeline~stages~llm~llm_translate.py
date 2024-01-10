from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM


class LLMTranslateStage:
    tokenizer = AutoTokenizer.from_pretrained("PontifexMaximus/ArabicTranslator")
    model = AutoModelForSeq2SeqLM.from_pretrained("PontifexMaximus/ArabicTranslator")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    def translate(self, input_text: str) -> str:
        translated_text = self.local_llm(input_text)
        print(f"translated_text: {translated_text}")
        return translated_text
