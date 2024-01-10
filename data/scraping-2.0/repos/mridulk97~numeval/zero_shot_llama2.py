import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
# pipeline = transformers.pipeline(
#     "text-generation", 
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
#     max_length=1000,
#     eos_token_id=tokenizer.eos_token_id)

device='cuda'

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
"text-generation", #task
model=model,
tokenizer=tokenizer,
torch_dtype=torch.bfloat16,
trust_remote_code=True,
device_map="auto",
max_length=2000,
# do_sample=True,
# top_k=10,
num_return_sequences=1,
eos_token_id=tokenizer.eos_token_id
)


llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs= {'temperature':0})


train_path = '/fastscratch/mridul/numeval/Train_Headline_Generation.json'
dev_path = '/fastscratch/mridul/numeval/Dev_Headline_Generation.json'

with open(train_path, 'r') as f:
    training_data = json.load(f)

with open(dev_path, 'r') as f:
    dev_data = json.load(f)


template2 = """

<s> [INST] <<SYS>> Write a headline of the following news article delimited by triple backquotes. 
The headline should contain numerical information from the news article. Always report 1000 numbers in K. for example: 10000 as 10K
Limit the headline to 5 words.<</SYS>>.
Here are 2 examples: 
News article 1 : {news1}
Headline : {headline1}

News article 2 : {news2}
Headline : {headline2}

[/INST]

Generate headline for the following news article. Limit the headline to 5 words : ```{text}```
Headline :
"""

news1 = """
(Oct 7, 2014 12:40 PM CDT) As of Jan. 1, Walmart will no longer offer 30,000 of its employees health insurance. 
Bloomberg notes that's about 2% of its workforce. The move comes as a reaction to the company's rising health 
care costs as far more of its employees and their families enrolled in its health care plans than it had expected 
following the ObamaCare rollout. The AP reports those costs will surge $500 million this fiscal year, $170 million 
more than had been estimated. Those affected are employees who average fewer than 30 hours of work per week; 
the Wall Street Journal explains they were grandfathered in when Walmart in 2012 stopped offering insurance to 
new hires who didn't exceed the 30-hour threshold. A benefits expert says Walmart is actually late to the game 
in terms of cutting insurance to some part-time workers; Target, the Home Depot, and others have already done so. 
Meanwhile, Walmart's full-time workers will see their premiums rise in 2015. Premiums for the basic plan, which 40% of 
its workforce is on, will increase 19% to $21.90 per pay period come Jan. 1.
"""

headline1 = "30K Walmart Part-Timers to Lose Health Insurance"

news2 = """
(Oct 29, 2013 8:15 AM CDT) Dax Shepard and Kristen Bell got married at the Beverly Hills courthouse, in a 
ceremony about as different from Kim Kardashian's last wedding extravaganza as it is possible to be. 
As Shepard revealed last night on Jimmy Kimmel Live, the whole thing\u2014including the fuel it took 
to get to the courthouse\u2014cost $142. It was just Kristen and I at this lonely courthouse, he said, 
so friends showed up afterward with a cake reading, in icing, The World's Worst Wedding. How many people 
can say they threw the world's worst wedding? Shepard asked.
"""
headline2 = "Dax Shepard: Wedding to Kristen Bell Cost $142"


template1 = """ <s> [INST] <<SYS>> Write a headline of the following news article delimited by triple backquotes. 
The headline should contain numerical information from the news article. Always report 1000 numbers in K. for example: 10000 as 10K
Limit the headline to 10 words<</SYS>>.
```{text}```
[/INST]
HEADLINE:
""" 

local_llm = HuggingFacePipeline(pipeline=pipeline)

prompt = PromptTemplate(template=template1, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


all_generated = []
for row in tqdm(dev_data):
    text = row['news']
    all_generated.append(llm_chain.run(text))

all_generated_strip = []
for row in all_generated:
    
    # breakpoint()
    if row[0]=='"':
        new_text = row[1:-1]
    elif row[0]=='`':
        new_text = row[1:-1]
    elif row[0]=='``':
        new_text = row[2:-2]
    elif row[0:3]=='```':
        new_text = row[3:-3]
    elif row[0:3]=='"``':
        new_text = row[3:-3]
    else:
        new_text = row
    print(new_text)
    all_generated_strip.append(new_text.strip())

print(len(all_generated_strip))
    

with open('/fastscratch/mridul/numeval/models/zero_zhot_llama/two_shot_preds_7b.txt', 'w') as f:
      for line in all_generated_strip:
          f.write(f"{line}\n")

# breakpoint()
print('DOne')