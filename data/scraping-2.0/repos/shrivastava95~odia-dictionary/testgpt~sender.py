import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm
import html
from time import sleep

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY



source_folder = f'parsed_texts'
output_destination = f'testgpt'

# for text_path in tqdm(os.listdir(f'{source_folder}')):
text_path = ''
input_path = f"{source_folder}/{text_path}"
output_path_check = f'{text_path}'
output_path = f'{output_destination}/{text_path}'
try:
    if output_path_check not in os.listdir(output_destination):
        # text_path = 'page6_0.png.pdf.txt'
        # with open(input_path, 'rb') as input_text:
        #     input_text_decoded = html.unescape(input_text.read().decode('utf8'))
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 0.8,
            max_tokens = 1600,
            messages = [
                {
                    "role": "system", 
                    "content": "You will help me parse raw noisy OCR output into neat tables. The output should contain the proper Odia characters instead of escape sequences."
                },
                {
                    "role": "user", 
                    "content": """The table should have three columns:
1. Word in English
2. Part of speech it belongs to
3. Odiya Translation of the word

I will give you two OCR outputs. Some entries hold erroneous values. For any given position, it may be such that only one of the outputs, or both, or none have the desired value. Your job is to merge the outputs in such a way that the errors are reduced.
Here is the raw output 1. Format it into the table. For example, here are the first two rows of the table:

a | Adjective&Article   | ଅରୋଟ୍‌
accordingly |	Adverb  |  ଆଦିଙ୍କ୍‌ ଲେକେ"""
                },
                {
                    "role": "assistant",
                    "content": "Sure! Please provide me with the raw output 1."
                },
                {
                    "role": "user",
                    "content": """basket  Noun.  କୁଲ୍‌  ଲା 
basket  made  with  bark 
Noun.  Som 
basket  small  size  Noun.  ଟାଟ୍‌ 
basket  used  for  storing  cow 
dung  Noun.  ପେଣ୍ଢା  ଟାଟ୍‌ 
bat  bird  Noun.  ଗାବିଲି  ପିଟେ 
bath  room  Noun.  ଏର୍‌  ପୁନ୍ଦାଙ୍କ୍‌  ଗାତ୍‌ 
bathe  —  Verb.  WQ  Qal 
bay  weaver  bird  Noun.  ଜାକ୍‌  କା 
gal 
be  Verb.  ଆଦାନାଦ୍‌,ମାନ୍‌ 
be  angry  Verb.  ` କୋପାମ୍‌ 
be  careful  Adverb.  ଚେତୁର୍‌  ମାନ୍‌ 
be  cleaved(split)  Verb.  ତାଲା 
AI  ତାନାଦ୍‌ 
be  cooled  Adjective.  GR  ତା 
be  employed  Adjective.  AIGA 
ମାଡ଼ାନାଦ୍‌ 
be  extinct  Verb.  ପିଭାନାଦ୍‌ 
be  struck  Verb.  କୋୟେ  ଇର୍କିଭା,ଗିଭା 
be  there  Verb.  ମାନ୍ଦାନାଦ୍‌ 
be  times  Adverb.  ଟୀକ୍‌  ସାମାୟାମ୍‌ 
ତେ 
beak  Noun.  ମୁସ୍‌  କିଡ଼୍‌ 
beans  `  Noun.  Age  କାୟା 
bear  Noun.  NIG  ଜ୍‌ 
bear  back  ।%/.  ପେର୍କେ  ଜାର୍ଗାନାଙ୍କ୍‌ 
ଇଦାନାଦ୍‌ 
bear  down  Verb.  ଅଡ଼ି  ଆନ୍ଦାନାଦ୍‌ 
bear  on  Verb.  ସାମ୍ାନ୍ଦାମ୍‌  ତାସାନାଦ୍‌ 
bear  through  Verb.  ତାର୍ରି  ଆନ୍ଦାନାଦ୍‌ 
bear  up  Verb.  ମେଦୁଲ୍‌  କଟେ 
ମାନ୍ଦାନାଦ୍‌ 
bear,sustain  Transitive"""
                },
                {
                    "role": "assistant",
                    "content": "Please provide me with the raw output 2, and I'll format it into a three-column table for you after merging with output 2 while considering the removal of non-common errors in both."
                },
                {
                    "role": "user",
                    "content": """basket  Noun.  କୁଲ୍‌  ଲା 
basket  made  with  bark 
Noun.  SYR 
basket  small  size  Noun.  ଟାଟ୍‌ 
basket  used  for  storing  cow 
dung  Noun.  ପେଣ୍ଡା  ଟାଟ୍‌ 
bat  bird  Noun.  ଗାବିଲି  ପିଟେ 
bath  room  Noun.  IQ  Qale  ଗାତ୍‌ 
bathe  —  Verb.  WQ  Qal 
bay  weaver  bird  Noun.  ଜାକ୍‌  କା 
gal 
be  Verb.  ଆଦାନାଦ୍‌,ମାନ୍‌ 
be  angry  Verb.  ` କୋପାମ୍‌ 
be  careful  Adverb.  ଚେତୁର୍‌  ମାନ୍‌ 
be  cleaved(split)  Verb.  ତାଲା 
AI  ତାନାଦ୍‌ 
be  cooled  Adjective.  GR  ତା 
be  employed  Adjective.  AIGA 
ମାଡ଼ାନାଦ୍‌ 
be  extinct  Verb.  ପିଭାନାଦ୍‌ 
be  struck  Verb.  କୋୟେ  ଭଇର୍କିଭା,ଗିଭା 
be  there  Verb.  ମାନ୍ଦାନାଦ୍‌ 
be  times  Adverb.  AR  ସାମାୟାମ୍‌ 
ତେ 
beak  —  Noun.  YQ  କିଡ଼୍‌ 
beans  `  Noun.  AQS  କାୟା 
bear  =  Noun.  ଏଡ଼୍‌  ଜ୍‌ 
bear  back  ।%/.  ପେର୍କେ  ଜାର୍ଗାନାଙ୍କ୍‌ 
ଇଦାନାଦ୍‌ 
bear  down  Verb.  ଅଡ଼ି  ଆନ୍ଦାନାଦ୍‌ 
bear  on  Verb.  ସାମ୍ାନ୍ଦାମ୍‌  ତାସାନାଦ୍‌ 
bear  through  Verb.  ତାର୍ରି  ଆନ୍ଦାନାଦ୍‌ 
bear  up  Verb.  ମେଦୁଲ୍‌  କଟେ 
ମାନ୍ଦାନାଦ୍‌ 
bear,sustain  Transitive"""
                }
            ]
        )

        gpt_text = html.unescape(completion.choices[0].message['content'])
        with open(f'testgpt/test_output.txt', 'w+b') as f:
            f.write(gpt_text.encode())
        # sleep(5)
        # break
except Exception as e:
    print(f'error occured during creation of {output_path}:')
    print(e)
    # continue
# break