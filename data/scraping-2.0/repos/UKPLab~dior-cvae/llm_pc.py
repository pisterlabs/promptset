import torch
from langchain import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import LLMChain
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
import tqdm


dataset_name = 'personachat'
#model_name = "llama-2-hf"
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_version = "/7B-Chat"
#tokenizer = AutoTokenizer.from_pretrained("/storage/ukp/shared/shared_model_weights/models--"+model_name+"/"+model_version)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#model_name = "alpaca"
#model_version = "/7B"
#tokenizer = AutoTokenizer.from_pretrained("/storage/ukp/shared/shared_model_weights/models--"+model_name+"/"+model_version)

#model = AutoModelForCausalLM.from_pretrained(
#        "/storage/ukp/shared/shared_model_weights/models--" + model_name + "/" + model_version,
#        torch_dtype=torch.float16,
#        load_in_8bit=False,
#        device_map="auto",
#    )

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
    )

pipe_50tokens = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500, # you can configure how many tokens you want to generate
)
llm_50tokens = HuggingFacePipeline(pipeline=pipe_50tokens)


template = """
I will give you a personal character description followed by the token [CLS] and a dialogue history between two people separated by [SEP] token and preceded by the [CLS] token, you need to generate the most possible next utterance for the dialogue history according to the character description and the dialogue history. Please DO NOT write sentences for both turns: you must play the role described by the character description and must only generate 1 possible responses corresponding to the role and to the dialogue history.
I will give you some examples first.

Dialogue history:
 drive a ford pickup truck . [SEP] i am very conservative . [SEP] my family lives down the street from me . [SEP] i go to church every sunday . [SEP] i have three guns and love hunting . [CLS] how are you ? being an old man , i am slowing down these days [SEP] hi , my dad is old as well , they live close to me and i see them often [SEP] that is a great thing honor your dad with your presence [SEP] sure , i pick him up for church every sunday with my ford pickup [SEP] sounds wonderful my wheelchair can go very fast on various terrains .
Response:
i guess that means you do not go hunting often ? i love hunting , i own 3 guns .

Dialogue history:
i met my best friend in kindergarten . [SEP] i am of the jewish faith . [SEP] i grew up in north dakota . [SEP] i got a job working in advertising last year . [SEP] i have traveled and studied abroad . [CLS] hey , are you a student , i traveled a lot , i even studied abroad [SEP] no , i work full time at a nursing home . i am a nurses aide . [SEP] nice , i just got a advertising job myself . do you like your job ? [SEP] nice . yes i do . caring for people is the joy of my life . [SEP] nice my best friend is a nurse , i knew him since kindergarten [SEP] very cool . do you have pets ?", "target": "no i do not , do you ?
Response:
no i do not , do you ?

Dialogue history:
i love watching movies and tv . [SEP] i have a husband who i love very much . [SEP] i do not like exercise or physical activity . [SEP] my favorite food is ice cream . [SEP] i am a homebody . [CLS] hey there . how are you ? [SEP] good , do you like animals ?
Response:
yeah , i like cats . i have one .

Dialogue history:
my mom is a secretary . [SEP] i have one brother . [SEP] i am a bodybuilder . [SEP] i like to eat a lot of meat . [CLS] hi there , how are you tonight ? [SEP] great , i just finished a huge steak how are you ? [SEP] i am good , drinking some scotch . [SEP] i am major meat eater to build muscles [SEP] oh , i like meat ok . [SEP] scotch goes good with meat [SEP] scotch goes good with anything lol . [SEP] i have a brother who likes scotch [SEP] i am an only child . [SEP] i have just the one brother and 3 sisters [SEP] do you have any pets ? i am scared of dogs . [SEP] i have a cat and a dog you would not be scared of my dog [SEP] what do you do for a living ? [SEP] i am a secretary like my mom hard living and you ? [SEP] i am an author . i write short stories .
Response:
do you want to write a story about a major meat eating bodybuilder ?

Dialogue history:
i am taking college and hoping to be a teacher . [SEP] i am a huge ed sheeran fan . [SEP] i run to relieve stress . [SEP] i am learning jujitsu but am still new to it . [CLS] hi trying to find some die hard ed sheeran fans . [SEP] sorry , i do not like music , i like reading mystery books . [SEP] so do i , but i like the older mysteries like agatha christie . [SEP] growing up on a farm , you learn to read a lot more . [SEP] i can imagine , in fact i hope to teach english after school . [SEP] nice , my job is being a truck driver , i have seven kids to feed . [SEP] wow , you need a hobby to get away , like jujitsu or running . [SEP] i like running , but i have no time .
Response:
well i sure can understand that . running does relieve stress though .
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = """Dialogue history: {question}
                    Response:
                    """
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
question_classification_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

class slashn_OutputParser(BaseOutputParser):
    def parse(self, text: str):
        # """Parse the output of an LLM call.
        # Get the generated text until \n\n
        # """
        # return text.strip().split("\n\n")[0]
        # return text.strip().split('\n')[:-2]
        return text
        
question_classification_chain = LLMChain(llm=llm_50tokens, 
                                         prompt=question_classification_prompt,
                                         output_parser=slashn_OutputParser(),
                                        )
                                        
                                        
with open('./' + dataset_name + '/processed/test.src') as f:
    test_src = f.readlines()
for i in range(len(test_src)):
    test_src[i] = test_src[i].strip()

test_hyp = []
    
for s in tqdm.tqdm(test_src):
    output = question_classification_chain.run(question=test_src[i])
    test_hyp.append(output)

with open('./' + dataset_name + '_' + model_name + '.txt', 'w') as f:
    for h in test_hyp:
        f.write(h + '\n')


    

