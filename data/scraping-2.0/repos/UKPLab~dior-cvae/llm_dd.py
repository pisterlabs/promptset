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


dataset_name = 'dailydialog'
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
I will give you a dialogue history between two people separated by [SEP] token, you need to generate the most possible next utterance for the dialogue history. Please DO NOT write sentences for both turns: you must choose the role you need to play and must only generate 1 possible responses corresponding to the role according to the dialogue history.
I will give you some examples first.

Dialogue history:
Say , Jim , how about going for a few beers after dinner ? [SEP] You know that is tempting but is really not good for our fitness . [SEP] What do you mean ? It will help us to relax . [SEP] Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ? [SEP] I guess you are right.But what shall we do ? I don't feel like sitting at home . [SEP] I suggest a walk over to the gym where we can play singsong and meet some of our friends . [SEP] That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them . [SEP] Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too . [SEP] Good.Let ' s go now .
Response:
All right .

Dialogue history:
Hey Lydia , what are you reading ? [SEP] I ' m looking at my horoscope for this month ! My outlook is very positive . It says that I should take a vacation to someplace exotic , and that I will have a passionate summer fling ! [SEP] What are you talking about ? Let me see that ... What are horoscopes ? [SEP] It ' s a prediction of your month , based on your zodiac sign . You have a different sign for the month and date you were born in . I was born on April 15th , so I ' m an Aries . When were you born ? [SEP] January 5th . [SEP] Let ' s see . . . you ' re a Capricorn . It says that you will be feeling stress at work , but you could see new , exciting developments in your love life . Looks like we ' ll both have interesting summers ! [SEP] That ' s bogus . I don't feel any stress at work , and my love life is practically nonexistent . This zodiac stuff is all a bunch of nonsense . [SEP] No , it ' s not , your astrology sign can tell you a lot about your personality . See ? It says that an Aries is energetic and loves to socialize . [SEP] Well , you certainly match those criteria , but they ' re so broad they could apply to anyone . What does it say about me ?
Response:
A Capricorn is serious-minded and practical . She likes to do things in conventional ways . That sounds just like you !

Dialogue history:
Hi , Becky , what's up ? [SEP] Not much , except that my mother-in-law is driving me up the wall . [SEP] What's the problem ?
Response:
She loves to nit-pick and criticizes everything that I do . I can never do anything right when she's around .

Dialogue history:
What are your personal weaknesses ? [SEP] I ' m afraid I ' m a poor talker . I ' m not comfortable talking with the people whom I have just met for the first time . That is not very good for business , so I have been studying public speaking . [SEP] Are you more of a leader or a follower ? [SEP] I don ' t try to lead people . I ' d rather cooperate with everybody , and get the job done by working together .
Response:
Do you think you can make yourself easily understood in English ?

Dialogue history:
I really need to start eating healthier . [SEP] I have to start eating better too . [SEP] What kind of food do you usually eat ? [SEP] I try my best to eat only fruits , vegetables , and chicken .
Response:
Is that really all that you eat ?
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
                                        
                                        
with open('./data/' + dataset_name + '/processed/test.src') as f:
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


    

