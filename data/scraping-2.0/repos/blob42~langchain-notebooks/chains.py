r"""°°°
# Chains

Chaining LLMs with each other or with other experts.

## Getting Started

- Using the simple LLM chain
- Creating sequential chains
- Creating a custom chain

### Why Use Chains ?

- combine multiple components together
- ex: take user input, format with PromptTemplate, pass formatted text to LLM.

## Query an LLM with LLMChain

°°°"""
#|%%--%%| <2XVP2VXIL1|DPRWRo3fl7>

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import pprint as pp

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}"
        )

#|%%--%%| <DPRWRo3fl7|tOpTb9idHh>
r"""°°°
We can now create a simple chain that takes user input format it and pass to LLM
°°°"""
#|%%--%%| <tOpTb9idHh|QXu2N1dEEC>

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt, output_key='company_name')

# run the chain only specifying input variables
print(chain.run("hand crafted handbags"))

# NOTE: we pass data to the run of the entry chain (see sequence under)

#|%%--%%| <QXu2N1dEEC|Kv6bj1l9I3>
r"""°°°
## Combining chains with SequentialChain

Chains that execute their links in predefined order.

- SimpleSequentialChain: simplest form, each step has a single input/output. 
Output of one step is input to next.
- SequentialChain: More advanced, multiple inputs/outputs.


Following tutorial uses SimpleSequentialChain and SequentialChain, each chains output is input to the next one.
This sequential chain will:
    1. create company name for a product. We just use LLMChain for that
    2. Create a catchphrase for the product. We will use a new LLMChain for the catchphrase, as show below.
°°°"""
#|%%--%%| <Kv6bj1l9I3|BMZLsdY9VP>

second_prompt = PromptTemplate(
        input_variables=["company_name"],
        template="Write a catchphrase for the following company: {company_name}",
        )
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key='catchphrase')

#|%%--%%| <BMZLsdY9VP|epQHxmeWCP>
r"""°°°
We now combine the two chains to create company name and catch phrase.
°°°"""
#|%%--%%| <epQHxmeWCP|SHwDHjVCxb>

from langchain.chains import SimpleSequentialChain, SequentialChain

#|%%--%%| <SHwDHjVCxb|lKgp9HR0VX>

full_chain = SimpleSequentialChain(
        chains=[chain, chain_two], verbose=True,
        )

print(full_chain.run("hand crafted handbags"))

#|%%--%%| <lKgp9HR0VX|RiYcYwJhdC>
r"""°°°
---

In the third prompt we create an small advertisement with the title and the product description
°°°"""
#|%%--%%| <RiYcYwJhdC|RhnqOumOtX>

ad_template = """Create a small advertisement destined for reddit. 
The advertisement is for a company with the following details:

name: {company_name}
product: {product}
catchphrase: {catchphrase}

advertisement:
"""
ad_prompt = PromptTemplate(
        input_variables=["product", "company_name", "catchphrase"],
        template=ad_template,
        )

#|%%--%%| <RhnqOumOtX|MsQnieyxgL>

#Connet the three chains together

ad_chain = LLMChain(llm=llm, prompt=ad_prompt, output_key='advertisement')

#|%%--%%| <MsQnieyxgL|4PYfwOxTlq>

final_chain = SequentialChain(
        chains=[chain, chain_two, ad_chain],
        input_variables=['product'],
        output_variables=['advertisement'],
        verbose=True
        )

ad = final_chain.run('Professional Cat Cuddler')
#|%%--%%| <4PYfwOxTlq|2akm8eB1EV>

print(ad)

#|%%--%%| <2akm8eB1EV|1iT7gBMABZ>
r"""°°°
## Creating a custom chain

Example: create a custom chain that concats output of 2 LLMChain

Steps:
    1. Subclass Chain class
    2. Fill out `input_keys` and `output_keys`
    3. add the `_call` method that shows how to execute chain
°°°"""
#|%%--%%| <1iT7gBMABZ|OUXv7kGtDH>

from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List

class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains
        all_inputs_vars = set(self.chain_1.input_keys).union(
                        set(self.chain_2.input_keys))
        return list(all_inputs_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str,str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}

#|%%--%%| <OUXv7kGtDH|MUOMbKovF6>
r"""°°°
Running the custom chain
°°°"""
#|%%--%%| <MUOMbKovF6|kBfPU3rB6L>
prompt_1  = PromptTemplate(
        input_variables=['product'],
        template='what is a good name for a company that makes {product}?'
        )
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
        input_variables=['product'],
        template='what is a good slogan for a company that makes {product} ?'
        )
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain  = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)

concat_output = concat_chain.run('leather handbags')
print(f'Concatenated output:\n{concat_output}')


#|%%--%%| <kBfPU3rB6L|9CdH3GtsmW>




