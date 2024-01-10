from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import LLMChain
import logging
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.callbacks import get_openai_callback
import polars as pl


def get_model(source: str, kwargs):
    assert "temperature" in kwargs, "temperature must be specified"
    ## Langchain wants temp explicitly stated, so here we go
    temperature = kwargs["temperature"]
    del kwargs["temperature"]

    if source.lower() == "chatgpt":
        logging.info("Initializing OpenAI chatGPT LLM")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",
            temperature=temperature,
            model_kwargs=kwargs,
        )
    elif source.lower() == "claude":
        logging.info("Initializing Anthropic Claude LLM")
        llm = ChatAnthropic(
            model="claude-instant-1.1", temperature=temperature, model_kwargs=kwargs
        )
    return llm

system_instruction = (
    "You are an academic with experience in molecular biology. "
    "You always answer in a factual unbiased way."
)
def get_extraction_prompt():
    extraction_context = (
        "Does the paper delimited with triple quotes below contain any per residue information? "
        "Per-residue information might look like the following: "
        "His96, meaning residue position 99 is a histidine. "
        "For example in text it might look like this: "
        "A hydrophilic pocket formed by the residues Thr199, Thr200, "
        "and His96, as mentioned above, and a hydrophobic one defined by "
        "Val121 which is known to represent the binding pocket for the enzyme substrate CO2.\n"
        "Summarise the information in a table which should look like this:\n"
        "Residue Position|Amino Acid Residue|Notes|\n"
        "Thr199|Threonine|Hydrogen bond with sulphonamide oxygen\n"
        "Thr200|Threonine|Hydrophilic pocket for the nitro group\n"
        "His96|Histidine|Hydrophilic pocket for the nitro group\n"
        "Val121|Valine|Lost hydrophobic interaction with the aromatic ring due to steric hindrance from the N-substitution\n\n"

        "Here is another example sentence:\n"
        "This 100-Å-long tunnel starts at the active site residue Lys219 of urease, "
        "exits HpUreB near Asp336 of the switch II region, passes through HpUreD "
        "between the two layers of β sheets, enters HpUreF near Ala233, and reaches "
        "the dimerization interface of HpUreF (Fig. 4A).\n"

        "Residue Position|Amino Acid Residue|Notes\n"
        "219|Lysine|Active site residue\n"
        "336|Aspartate|Residue in switch II region\n"
        "233|Ala|N/A\n"
        "'''\n"
        "{paper}\n"
        "'''\n"
        "Residue information table:\n"
        )

    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    human_prompt = HumanMessagePromptTemplate.from_template(extraction_context)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return chat_prompt


def get_extraction_chain(llm, verbose=False) -> LLMChain:
    prompt = get_extraction_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return chain

def find_residue_information(chain, paper):
    with get_openai_callback() as cb:
            summary = chain.run(
                paper=paper
            )
            print(cb)
            print(summary)
            return summary

extra_args={}
extraction_chain = get_extraction_chain(
        get_model(
            "chatgpt",
            {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
            | extra_args,
        ),
        verbose=True,
    )

paper = """The VapBC system, which belongs to the type II toxin–antitoxin (TA) system, is the most abundant and widely studied system in Mycobacterium tuberculosis. The VapB antitoxin suppresses the activity of the VapC toxin through a stable protein–protein complex. However, under environmental stress, the balance between toxin and antitoxin is disrupted, leading to the release of free toxin and bacteriostatic state. This study introduces the Rv0229c, a putative VapC51 toxin, and aims to provide a better understanding of its discovered function. The structure of the Rv0229c shows a typical PIN-domain protein, exhibiting an β1-α1-α2-β2-α3-α4-β3-α5-α6-β4-α7-β5 topology. The structure-based sequence alignment showed four electronegative residues in the active site of Rv0229c, which is composed of Asp8, Glu42, Asp95, and Asp113. By comparing the active site with existing VapC proteins, we have demonstrated the justification for naming it VapC51 at the molecular level. In an in vitro ribonuclease activity assay, Rv0229c showed ribonuclease activity dependent on the concentration of metal ions such as Mg2+ and Mn2+. In addition, magnesium was found to have a greater effect on VapC51 activity than manganese. Through these structural and experimental studies, we provide evidence for the functional role of Rv0229c as a VapC51 toxin. Overall, this study aims to enhance our understanding of the VapBC system in M. tuberculosis."""


examples = pl.read_csv("examples.csv")
print(examples)

examples = examples.with_columns(residue_table=pl.col("input_text").apply(lambda x: find_residue_information(extraction_chain, x)))

examples.write_csv("examples_with_residue_table.csv")


exit()





