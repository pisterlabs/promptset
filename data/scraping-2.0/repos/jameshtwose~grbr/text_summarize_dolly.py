# %%
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import requests

# %%
generate_text = pipeline(
    model="databricks/dolly-v2-3b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    return_full_text=True,
)

# %%
# template for an instrution with no input
prompt = PromptTemplate(input_variables=["instruction"], template="{instruction}")

# %%
# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}",
)

# %%
hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
# %%
llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
# %%
print(
    llm_chain.predict(
        instruction="Explain to me the difference between nuclear fission and fusion."
    ).lstrip()
)
# %%
context = """George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman,
and Founding Father who served as the first president of the United States from 1789 to 1797."""

print(
    llm_context_chain.predict(
        instruction="When was George Washington president?", context=context
    ).lstrip()
)

# %%
new_context = """
The Economic Times daily newspaper is available online now.
Goodyear's India unit posts near two-fold jump in Q4 profit
on easing costs Goodyear India's standalone profit for Q4 2019
doubled, rising to INR 336.1m ($4.1m) from INR 173.9m year-on-year.
The increase was helped by lower raw material costs and
increased demand for vehicles in India. The decrease in raw
material costs, coupled with cost efficiencies, resulted in
higher profitability in the OE segment, stated Sandeep Mahajan,
chairman and managing director of Goodyear India. Cost of
materials consumed fell 5.4%, while revenue rose 9.1% to
INR 6.53bn for the quarter. Total expenses increased by 5.4%
to INR 6.10bn. Don’t miss out on ET Prime stories! Get your
daily dose of business updates on WhatsApp. click here!
Prime Minister Narendra Modi inaugurated the new Parliament
building on Sunday and said India is leaving behind the
“mentality of slavery.” He also called for making India a 
developed nation by the 100th anniversary of its Independence. 
Baring Private Equity Asia (BPEA) EQT is set to acquire 
Credila Financial Services, the educational loan arm of 
Housing Development Finance Corp (HDFC), for $1.3-1.5 
billion (₹10,000 crore-12,000 crore), trumping private 
equity rivals Carlyle, TA Associates, Blackstone and 
CVC Capital, among others, said people in the know. US-based 
multinationals, British companies and leading French 
industrial groups that have closely held subsidiaries in 
India will face so-called angel tax scrutiny if they 
bring in fresh equity capital. Read More News on Download 
The Economic Times News App to get Daily Market Updates 
& Live Business News. Indian Indices 11 Advance1 Decline 
32 Advance18 Decline 61 Advance39 Decline 35 Advance15 
Decline 20 Advance10 Decline Market Dashboard \n    \t    
NSEBSE NSE BSE Popular In Markets All Mutual Funds Top 
Tax Saving Mutual Funds Better Than Fixed Deposits Low 
Cost High Return Funds Best Hybrid Funds Best Large 
Cap Funds SIP’s starting Rs. 500 Top Performing Mid 
Caps Promising Multi Cap Funds Top Rated Funds Top 
Performing Index Funds Why follow tips? Choose your 
winners rationally in 3 simple steps! Latest from ET 
Most Searched Stocks Trending Now Popular Categories 
Hot on Web In Case you missed it Top Calculators Top 
Searched Companies Most Searched IFSC Codes Popular 
Articles Most Searched Articles Top Definitions Top 
Prime Articles Top Videos Top Story Listing Top Performing 
MF Top Slideshow Top Trending Topics Trending Articles 
Follow us on: Download ET App: subscribe to our newsletter 
Find this comment offensive? Choose your reason below 
and click on the Report button. This will alert our 
moderators to take action Reason for reporting: Your 
Reason has been Reported to the admin. To post this 
comment you must Log In/Connect with: Fill in your 
details: Will be displayed Will not be displayed Will 
be displayed Share this Comment:  Stories you might 
be interested in
"""

print(
    llm_context_chain.predict(
        instruction="Please summarize this text", context=new_context
    ).lstrip()
)
# %%
