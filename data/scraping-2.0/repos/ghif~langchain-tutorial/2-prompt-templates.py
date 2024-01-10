from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

ROLE = "profesor"
COUNTRY = "Singapura"

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())

"""
Language Model
"""
chat_llm = ChatOpenAI(temperature=0.0)


# template = """
# Apa persyaratan untuk menjadi seorang {role} di {country}?

# Jawaban harus dalam Bahasa Indonesia.
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Assign variable to the template
# final_prompt = prompt.format(
#     role=ROLE, 
#     country=COUNTRY
# )

next_month = "Desember"
template = """
Bulan ini merupakan bulan Agustus, bulan depan merupakan bulan {month}. 
Apakah pertanyaan tersebut benar? Jika salah, jelaskan letak kesalahannya.
"""

prompt = ChatPromptTemplate.from_template(template)
final_prompt = prompt.format(
    month=next_month
)

response = chat_llm.predict(final_prompt)
print(f"Final prompt: {final_prompt}")
print(f"Response: {response}")