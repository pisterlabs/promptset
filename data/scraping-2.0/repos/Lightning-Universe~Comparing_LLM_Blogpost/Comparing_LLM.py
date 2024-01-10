# %%
import torch 
from pathlib import Path
from Llama.model_lit_llama import  LitLlamaPipeline
from Alpaca.model_lit_llama_alpaca import  AlpacaPipeline
from HF_models.models_HF import CustomPipeline
from langchain.llms import OpenAI
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  
import os

model_ids_hf = [
        "EleutherAI/gpt-j-6B" ,
        "google/ul2",
        "bigscience/bloomz-7b1",
        "facebook/opt-iml-max-30b" ,
        "google/flan-t5-xxl" ,
        "bigscience/bloomz-7b1",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/pythia-12b-deduped",
        "dolly-v2-12",
        "nomic-ai/gpt4all-j",
        "Cerebras-GPT-13B",
        "t5-11b",
        "ul2",
        "OPT",
]

# If you decide to  use GPT provide a  your key:
os.environ["OPENAI_API_KEY"] = ''
model_id = "OPT"
print(model_id)

if model_id == "Llama":
    from Llama.covert_llama_weights import meta_weights_for_nano_model
    # meta_weights_for_nano_model(
    #     output_dir = Path("checkpoints/lit-llama"),
    #     ckpt_dir = Path(your_llama_checkpoint_path),
    #     tokenizer_path = Path(your_tokenizer_path),
    #     model_size=model_size,
    # )
    llm = LitLlamaPipeline()
elif model_id == "GPT3":
    llm = OpenAI(temperature=.75, model_name="text-davinci-003")
elif model_id == "GPT3.5":
    llm = OpenAI(temperature=.75, model_name= "gpt-3.5-turbo")
elif model_id == "Alpaca":
    llm =  AlpacaPipeline()
elif model_id in model_ids_hf:
    llm = CustomPipeline(model_id)
else:
    raise Exception("Not a valid model")


template =  """
We have provided context information below.

Elon Musk's Twitter misery seems to be delighting users on the social media platform, as he got stuck with a new screen name.
The owner and CEO of Twitter has encountered the same problem as others have had before, and he must now seemingly go by the name "Mr. Tweet" for the foreseeable future.
Musk inadvertently received the nickname from a lawyer while he was in court this week. He shared his misfortune with his millions of followers, and didn't receive much sympathy in return.
Mr. Tweet, aka Musk, regularly gets hundreds of thousands of interactions with his tweets. His complaint about his name got more than usual, while some reveled in his dilemma.
It's not the first time a celebrity has found themselves stuck with a Twitter name they didn't want. In November, singer Doja Cat called on Musk for help after she got stuck with an unusual name.
"I don't wanna be Christmas forever [Elon Musk] please help I've made a mistake," she wrote. Musk replied telling her they're working on it, but he also acknowledged it was "pretty funny though."
The irony that the owner and CEO of Twitter couldn't change his own name wasn't lost on many of his followers.
"Have you tried calling the help desk?" Twitter user @TheChiefNerd replied.
Musk's new screen name wasn't picked at random, though, as some explained how the joke came about.
On January 23, long before Musk renamed himself, Silicon Valley journalist Teddy Schleifer explained: "The lawyer who is cross-examining Elon Musk accidentally just called him 'Mr. Tweet' instead of 'Mr. Musk.' Elon says 'Mr. Tweet' is all good. 'That's probably an accurate description,'" Schleifer wrote. Musk even clicked like on that tweet at the time.
Musk was appearing in court during the Tesla shareholder trial. Investors are suing him, alleging that he committed securities fraud via a tweet in 2018.
"Mr. Tweet in the house..." wrote Fox News contributor Joe Concha. Journalist Johnna Crider also approved of the new pseudonym. "I personally think Mr. Tweet is better—has more personality as a nickname," she commented.
The popular creator @iamchillpill joked that Musk would have to speak into a mirror to find help. "Mr. Tweet please, let me be Elon again," they wrote.
Online news outlet The Chainsaw didn't see the funny side though, and brought the name back to the ongoing litigation. "Hey Mr. Tweet, how's the Tesla trial going?" it wrote.
Newsweek reached out to representatives of Musk and Tesla for comment on the ongoing trial.

Using only this information, please answer the question: {text}

Answer:
"""

prompt_template = PromptTemplate(input_variables=[ "text"], template=template)
answer_chain = LLMChain(llm=llm , prompt=prompt_template)

questions = ["what’s Elon's new Twitter username?",
    "why is it funny that he cannot change it?",
    "make a joke about this",
    "How did this get started?"
    ]

for question in questions:
    answer_chain = LLMChain(llm=llm , prompt=prompt_template)
    answer = answer_chain.run(question)
    print(f"\nThe question is: {question }")
    print(f"\n {answer.strip()}")





