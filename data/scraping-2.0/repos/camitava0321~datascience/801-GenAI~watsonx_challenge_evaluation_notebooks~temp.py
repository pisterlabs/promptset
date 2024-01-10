import os

from dotenv import load_dotenv

from genai.model import Credentials, Model
from genai.schemas import GenerateParams


GENAI_KEY="pak-dlgq9J79SHjqkFeeD0r2FvneplVqkOfLVzIjy7bCy6Y"
GENAI_API="https://bam-api.res.ibm.com/v1/"
#GENAI_API="https://workbench-api.res.ibm.com/v1/"
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

#creds = Credentials(api_key, api_endpoint=api_url)
creds = Credentials(GENAI_KEY, GENAI_API)
#%%
# Using Python "with" context
print("\n------------- Example (Greetings)-------------\n")

# Instantiate the GENAI Proxy Object
params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=10,
    min_new_tokens=1,
    stream=False,
    temperature=0.7,
    top_k=50,
    top_p=1,
)

# model object
model = Model("google/flan-ul2", params=params, credentials=creds)
greetings = [
    "Hello! How are you?",
    "I am fine and you?"
    ]

# Call generate function
responses = model.generate_as_completed(greetings)
for response in responses:
    print(f"Generated text: {response.generated_text}")

#%%
import time
print("\n------------- Example (Model Talk)-------------\n")

bob_params = GenerateParams(decoding_method="sample", max_new_tokens=40, temperature=2)
alice_params = GenerateParams(decoding_method="sample", max_new_tokens=25, temperature=0)
bob = Model("bigscience/bloom", params=bob_params, credentials=creds)
alice = Model("google/flan-ul2", params=alice_params, credentials=creds)

sentence = "Do you know how plants communicate?"

print(f"[Alice] --> {sentence}")
count=0
while (count<10):
    bob_response = bob.generate([sentence])
    # from first batch get first result generated text
    bob_gen = bob_response[0].generated_text
    print(f"[Bob] --> {bob_gen}")

    alice_response = alice.generate([bob_gen])
    # from first batch get first result generated text
    alice_gen = alice_response[0].generated_text
    print(f"[Alice] --> {alice_gen}")

    sentence = alice_gen
    count=count+1
    time.sleep(0.5)

#%%
from genai.schemas import TokenParams
print("\n------------- Example (Tokenize)-------------\n")

flan_t5 = Model("google/flan-t5-xxl", params=TokenParams, credentials=creds)
sentence = "Hello! How are you?"
tokenized_response = flan_t5.tokenize([sentence], return_tokens=True)

print(f"Token counts: {tokenized_response[0].token_count}")
print(f"Tokenized response: {tokenized_response[0].tokens}")

#%%
print("\n------------- Example (Async Greetings)-------------\n")

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=20,
    min_new_tokens=10,
    temperature=0.7,
)
flan_ul2 = Model("google/flan-ul2", params=params, credentials=creds)

greeting = "Hello! How are you?"
lots_of_greetings = [greeting] * 20
num_of_greetings = len(lots_of_greetings)
num_said_greetings = 0

# yields batch of results that are produced asynchronously and in parallel
for result in flan_ul2.generate_async(lots_of_greetings):
    num_said_greetings += 1
    print("[Progress {:.2f}]".format(num_said_greetings/num_of_greetings*100.0))
    print("\t {}".format(result.generated_text))

#%%
from genai.extensions.langchain import LangChainInterface


print("\n------------- Example (LangChain)-------------\n")

params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

langchain_model = LangChainInterface(model="google/flan-ul2", params=params, credentials=creds)
print(langchain_model("Answer this question: What is life?"))    


#%%
from genai.schemas import ModelType


params = GenerateParams(temperature=0, max_new_tokens=3)
model = Model(ModelType.UL2, params=params, credentials=creds)
input_sentence = "Write a tagline for an alumni association: Together we"

response = model.generate([input_sentence])
res_sentence = response[0].generated_text

print(input_sentence) # Write a tagline for an alumni association: Together we
print(res_sentence) # can do
print("\n------------- Example (Tokenize)-------------\n")

model = Model(ModelType.FLAN_T5, params=TokenParams, credentials=creds)
input_sentence = "Write a tagline for an alumni association: Together we"
tokenized_response = model.tokenize([input_sentence], return_tokens=True)
print(tokenized_response[0].token_count) # 13
print(tokenized_response[0].tokens) # ['▁Write', '▁', 'a', '▁tag', ...]


#%% - AMC Playground
def printResponse(response):
    items=len(response)
    for res in response:
        print(res.generated_text)
        print("\n")
        print("generated_token_count : "+str(res.generated_token_count)+
              " input_token_count : "+str(res.input_token_count)+
              " stop_reason : "+res.stop_reason)

input_sentence = "Generate a 5 sentence advertisement message for a hotel with the following \
    features:\nHotel - Marriott\noffer includes - President Suites, 22% discount, \
    private meeting lounge and swimming pool, private terrace and garden, \
    complimentary breakfast and dinner, banquet facility at 35% discount, \
    valet parking\ntone - informative\nresponse requested - click the link\n \
    end date - July 15\nWe bring out an all new discount offer with features like: 22% \
    deduction, special facilities for Meeting lounge and Swimming Poilpool the package \
    included president suites is of course exclusive to men offering your friend a poolside \
    cocktail? Let's plan one right now it only need my quick booking for 20 pax max...it \
    includes complementary Breakfast even dessert plus a very interesting theme-faunic \
    party (all set decorators/party organizer)"
params = GenerateParams(decoding_method="sample",
                        temperature=1, 
                        top_p=0,
                        top_k=50,
                        random_seed=111,
                        repetition_penalty=1.2,
                        min_new_tokens=50,
                        max_new_tokens=300,
                        beam_width=1)
model = Model(ModelType.UL2, params=params, credentials=creds)

response = model.generate([input_sentence])
res_sentence = response[0].generated_text
printResponse(response)

#%%

input_sentence = "Financial Highlights\nThanks Arvind. I’ll start with the financial highlights of the fourth quarter. We delivered $16.7 billion in revenue, $3.8 billion of operating pre-tax income and operating earnings per share of $3.60. In our seasonally strongest quarter, we generated $5.2 billion of free cash flow. Our revenue for the quarter was up over six percent at constant currency. While the dollar weakened a bit from 90 days ago, it still impacted our reported revenue by over $1 billion – and 6.3 points of growth. As always, I’ll focus my comments on constant currency. And I’ll remind you that we wrapped on the separation of Kyndryl at the beginning of November. The one-month contribution to our fourth quarter revenue growth was offset by the impact of our divested health business.\nRevenue growth this quarter was again broad based. Software revenue was up eight percent and Consulting up nine percent. These are our growth vectors and represent over 70 percent of our revenue. Infrastructure was up seven percent. Within each of these segments, our growth was pervasive. We also had good growth across our geographies, with mid-single digit growth or better in Americas, EMEA and Asia Pacific. And for the year, we gained share overall. We had strong transactional growth in software and hardware to close the year. At the same time, our recurring revenue, which provides a solid base of revenue and profit, also grew – led by software. I’ll remind you that on an annual basis, about half of our revenue is recurring.\nOver the last year, we’ve seen the results of a more focused hybrid cloud and AI strategy. Our approach to hybrid cloud is platform centric. As we land the platform, we get a multiplier effect across Software, Consulting and Infrastructure. For the year, our hybrid cloud revenue was over $22 IBM 4Q22 Earnings Prepared Remarks billion – up 17 percent from 2021.\nLooking at our profit metrics for the quarter, we expanded operating pretax margin by 170 basis points. This reflects a strong portfolio mix and improving Software and Consulting margins. These same dynamics drove a 60-basis point increase in operating gross margin. Our expense was down year to year, driven by currency dynamics. Within our base expense, the work we’re doing to digitally transform our operations provides flexibility to continue to invest in innovation and in talent. Our operating tax rate was 14 percent, which is flat versus last year. And our operating earnings per share of $3.60 was up over seven percent. Turning to free cash flow, we generated $5.2 billion in the quarter and $9.3 billion for the year. Our full-year free cash flow is up $2.8 billion from 2021. As we talked about all year, we have a few drivers of our free cash flow growth. First, I’ll remind you 2021’s cash flow results included Kyndryl-related activity – including the impact of spin charges and capex. Second, we had working capital improvements driven by efficiencies in our collections and mainframe cycle dynamics. Despite strong collections, the combination of revenue performance above our model and the timing of the transactions in the quarter led to higher-than-expected working capital at the end of the year. This impacted our free cash flow performance versus expectations. Our year-to-year free cash flow growth also includes a modest tailwind from cash tax payments and lower payments for structural actions, partially offset by increased capex investment for today’s IBM.\nIn terms of cash uses for the year, we invested $2.3 billion dollars to acquire eight companies across software and consulting, mitigated by over a billion dollars in proceeds from divested businesses, and we returned nearly six billion dollars to shareholders in the form of dividends. From a  IBM 4Q22 Earnings Prepared Remarks balance sheet perspective, we ended the year in a strong liquidity position with cash of $8.8 billion. This is up over a billion dollars year to year. And our debt balance is down nearly a billion dollars. Our balance sheet remains strong, and I’d say the same for our retirement-related plans. At year end, our worldwide tax-qualified plans are funded at 114 percent, with the U.S. at 125 percent. Both are up year to year. You’ll recall back in September, we took another step to reduce the risk profile of our plans. We transferred a portion of our U.S. qualified defined benefit plan obligations to insurers, without changing the benefits payable to plan participants. This resulted in a significant non-cash charge in our GAAP results in the third quarter, and we’ll see a benefit in our nonoperating charges going forward. You can see the benefit of this and other pension assumptions to the 2023 retirement-related costs in our supplemental charts."

params = GenerateParams(decoding_method="greedy",
                        random_seed=111,
                        repetition_penalty=2,
                        min_new_tokens=50,
                        max_new_tokens=300,
                        beam_width=1)
model = Model(ModelType.UL2, params=params, credentials=creds)

response = model.generate([input_sentence])
res_sentence = response[0].generated_text
printResponse(response) # can do

#%%
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams, ModelType
from genai.credentials import Credentials

print("\n------------- Example (LangChain)-------------\n")

params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

langchain_model = LangChainInterface(model=ModelType.FLAN_UL2, params=params, credentials=creds)
print(langchain_model("Answer this question: What is life?"))

#%% - Prompt Templating - Watsonx Prompt Output
from genai.prompt_pattern import PromptPattern
from genai.schemas import GenerateParams, ModelType


params = GenerateParams(temperature=0.5)

model = Model(ModelType.FLAN_UL2, params=params, credentials=creds)


_template = """
{{instruction}}

{{#list}}

Country: {{country}}
Airport: {{airport}}
Capital: {{capital}}

{{/list}}
Country: {{input}}
"""

print("\n------------- Mustaches Prompt Template -------------\n")
pt = PromptPattern.from_watsonx(credentials=creds, 
            name="list-qa-airport-3", template=_template)
print(f"\nPrompt: {pt}")


print("\n------------- Rendered Prompt -------------\n")
inputs = ["Spain", "Finland", "Iraq", "India", "Bangladesh"]
data = {
    "list": [
        {"country": "Canada", "capital": "Ottawa", "airport": "YOW"},
        {"country": "Germany", "capital": "Berlin", "airport": "BER"},
        {"country": "USA", "capital": "Washington", "airport": "DCA"},
    ]
}

rendered_prompts = pt.render(inputs=inputs, data=data)
for pt in rendered_prompts:
    print(pt)
    print("- - - - ")

#%%

from genai.options import Options
from genai.prompt_pattern import PromptPattern
from genai.schemas import GenerateParams, ModelType


params = GenerateParams(temperature=0.5)

model = Model(ModelType.FLAN_UL2, params=params, credentials=creds)


_template = """
{{instruction}}

{{#list}}

Country: {{country}}
Airport: {{airport}}
Capital: {{capital}}

{{/list}}
Country: {{input}}
"""

print("\n------------- Mustaches Prompt Template -------------\n")
pt = PromptPattern.from_watsonx(credentials=creds, name="list-qa-airport-3", template=_template)
print(f"\nPrompt: {pt}")


print("\n------------- Response -------------\n")
options = Options(
    watsonx_template=pt,
    watsonx_data={
        "instruction": "blaahh",
        "list": [
            {"country": "Canada", "capital": "Ottawa", "airport": "YOW"},
            {"country": "Germany", "capital": "Berlin", "airport": "BER"},
            {"country": "USA", "capital": "Washington", "airport": "DCA"},
        ],
    },
)

inputs = ["Spain", "Finland", "Iraq", "India", "Bangladesh"]
for resp in model.generate_async(inputs * 10, options=options, hide_progressbar=True):
    print(f"\nCountry: {resp.input_text} \n{resp.generated_text}")


for resp in model.tokenize_async(inputs, options=options, return_tokens=True):
    print(resp)

#%%

from genai.prompt_pattern import PromptPattern
from genai.schemas import TokenParams
from genai.services.prompt_template_manager import PromptTemplateManager


print("\n------------- Example Mustaches Prompt [ CREATE ] -------------\n")

params = TokenParams(return_tokens=True)

_template = """
{{ instruction }}
{{#examples}}

{{input}}
{{output}}

{{/examples}}
{{input}}
"""

print("\n------------- Example Mustaches Prompt [ SAVE ] -------------\n")
test_pt = PromptPattern.from_watsonx(credentials=creds, name="test", template=_template)
print("\nSaved template information:\n", test_pt)


print("\n------------- Example Mustaches Prompt [ LIST ] -------------\n")
pts = PromptTemplateManager.load_all_templates(credentials=creds)
for r in pts.results:
    print(f"{r.name}, {r.id}")


print("\n------------- Example Mustaches Prompt [ GET ONE ] -------------\n")
template = PromptPattern.from_watsonx(credentials=creds, name="aa", template=_template)
print(template.watsonx)


print("\n------------- Example Mustaches Prompt [ UPDATE ] -------------\n")
template = PromptPattern.from_watsonx(credentials=creds, name="aa", template="{{a}}{{b}}")
print(template.watsonx)


print("\n------------- Example Mustaches Prompt [ DELETE ] -------------\n")
_id = test_pt.delete()
print(f"\n Deleted prompt with id : {_id}")
    
#%%

    
    
    
    