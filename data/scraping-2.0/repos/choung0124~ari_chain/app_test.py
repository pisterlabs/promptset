import streamlit as st
from transformers import logging
from langchain.llms import TextGen
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
import streamlit as st
import streamlit as st
from pyvis.network import Network
from CustomLibrary.Custom_Agent import CustomLLMChain, CustomLLMChainAdditionalEntities
from CustomLibrary.Custom_Prompts import (
    Entity_type_Template_add,
    Entity_type_Template_Alpaca, 
    Entity_Extraction_Template_alpaca, 
    Entity_type_Template_airo, 
    Entity_Extraction_Template_airo, 
    Entity_Extraction_Template,  
    Entity_type_Template, 
    Additional_Entity_Extraction_Template,
    Final_Answer_Template,
    Additional_Entity_Extraction_Template_Vicuna,
    Entity_type_Template_add_Vicuna,
    Entity_type_Template_add_Alpaca,
    Entity_Extraction_Template_alpaca,
    Additional_Entity_Extraction_Template_Alpaca
)
from CustomLibrary.App_Utils import(
    get_umls_info, 
    extract_entities, 
    get_names_list, 
    get_names_list, 
    get_entity_types, 
    get_additional_entity_umls_dict,
    create_and_display_network
)
from CustomLibrary.Graph_Visualize import parse_relationships_pyvis, parse_relationships_graph_qa
from CustomLibrary.Graph_Class import KnowledgeGraphRetrieval
from CustomLibrary.Pharos_Graph_QA import PharosGraphQA
from CustomLibrary.OpenTargets_Graph_QA import OpenTargetsGraphQA
from CustomLibrary.Predicted_QA import PredictedGrqphQA
#Could there be a synergistic drug-drug interaction between lamotrigine and rivastigmine for lewy body dementia?
# Set logging verbosity
logging.set_verbosity(logging.CRITICAL)
@st.cache_data()
def initialize_models():
    model_url = "https://foods-believe-happened-f.trycloudflare.com/"
    local_model_url = "http://127.0.0.1:5000/"
    llm = TextGen(model_url=model_url, max_new_tokens=2048)
    local_llm = TextGen(model_url=local_model_url, max_new_tokens=2048)
    Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template, input_variables=["input"])
    #Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_alpaca, input_variables=["input"])
    entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
    #entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=local_llm, output_key="output",)
    return llm, entity_extraction_chain

@st.cache_data()
def initialize_knowledge_graph():
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "NeO4J"
    return uri, username, password

st.set_page_config(layout="wide")
st.title("Multi-Hop Question Answering")

# Define the progress bar
progress_bar = st.empty()

# Define the callback function to update the progress bar
def progress_callback(progress):
    progress_bar.progress(progress)

###########################################################################################################################################################################################################################################

additional_entity_extraction_prompt = PromptTemplate(template=Additional_Entity_Extraction_Template, input_variables=["input", "entities"])
#additional_entity_extraction_prompt = PromptTemplate(template=Additional_Entity_Extraction_Template_Alpaca, input_variables=["input", "entities"])
llm, entity_extraction_chain = initialize_models()
#llm, local_llm, entity_extraction_chain = initialize_models()
uri, username, password = initialize_knowledge_graph()
additional_entity_extraction_chain = CustomLLMChainAdditionalEntities(prompt=additional_entity_extraction_prompt, llm=llm, output_key="output",)

Entity_type_prompt = PromptTemplate(template=Entity_type_Template, input_variables=["input"])
#Entity_type_prompt = PromptTemplate(template=Entity_type_Template_Alpaca, input_variables=["input"])
Entity_type_prompt_add = PromptTemplate(template=Entity_type_Template_add, input_variables=["input"])
#Entity_type_prompt_add = PromptTemplate(template=Entity_type_Template_add_Alpaca, input_variables=["input"])
Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=llm)
#Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=local_llm)
Entity_type_chain_add = LLMChain(prompt=Entity_type_prompt_add, llm=llm)
#Entity_type_chain_add = LLMChain(prompt=Entity_type_prompt_add, llm=local_llm)

question = st.text_input("Enter your question")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Checking drug interaction..."):
        # Entity extraction
        entities, additional_entities = extract_entities(question, entity_extraction_chain, additional_entity_extraction_chain)

        entities_umls_ids = get_umls_info(entities)

        names_list = get_names_list(entities_umls_ids)
        print(names_list)

        entity_types = get_entity_types(Entity_type_chain, names_list)

###########################################################################################################################################################################################################################################

        if additional_entities:
            additional_entity_umls_dict = get_additional_entity_umls_dict(additional_entities, Entity_type_chain_add)
            print(additional_entity_umls_dict)

            keys_to_remove = []
            for key, value in additional_entity_umls_dict.items():
                # Check if any value is empty
                if any(v is None or v == '' for v in value.values()):
                    # Add the key to the list of keys to remove
                    keys_to_remove.append(key)

            # Remove the keys outside the loop
            for key in keys_to_remove:
                del additional_entity_umls_dict[key]

            print(additional_entity_umls_dict)

            knowledge_graph = PredictedGrqphQA(uri, username, password, llm, entity_types, question, additional_entity_types=additional_entity_umls_dict)
        else:
            knowledge_graph = PredictedGrqphQA(uri, username, password, llm, entity_types, question)

        previous_answer = """Based on the additional context and potential mechanisms involved in a synergistic drug-drug interaction between lamotrigine and rivastigmine for Lewy body dementia, there are several possible pathways and molecular processes that could be implicated. Here are some examples:
ABCB1 and UGT1A4 involvement: As you mentioned, these efflux transporters play important roles in the elimination of both drugs from the brain. If one drug alters their expression or activity, it can indirectly impact the other's pharmacokinetics. This could lead to an increase in the concentrations of both drugs in the central nervous system, potentially leading to a synergistic effect on therapeutic outcomes but also side effects.
Complex network of interactions: The intricate web of relationships among various neurotransmitter systems, receptor, and metabolic enzymes suggests that even subtle changes in one pathway can have wide-ranging consequences on others. This complexity may make it difficult to predict exactly how these two drugs will interact but provides many potential mechanisms for synergy or antagonism.
Neurotransmitter system interplay: Lamotrigine's modulation of glutamate while rivastigmine targets cholinesterase could lead to direct interactions between the two drugs at the level of neurotransmission. For example, lamotrigine might enhance the effectiveness of rivastigmine's cholinergic activity through modulating glutamate receptors, and vice versa.
Shared target proteins: While unlikely given their different mechanisms of action, it is possible that both drugs may target the same or similar molecular targets in the brain. If so, their combined effects may be additive or opposing depending on the specific molecular interactions involved.
SCN2A, SCN8A, and other sodium channel involvement: Lamotrigine interacts with multiple sodium channels including SCN2A and SCN8A, which are implicated in mood disorders and cognition. These channels could play a role in the drug's therapeutic actions but also potential side effects like seizures. Rivastigmine has been shown to bind to ACHE, which degrades acetylcholine, an important neurotransmitter that is relevant for memory function. The interplay between these different systems may influence the overall effectiveness of the combination treatment.
GBA-BCHE-ACHE pathway: GBA is involved in the degradation of beta-amyloid, while BCHE cleaves acetylcholinesterase (AChE). Lamotrigine could target GBA to enhance the clearance of beta-amyloid, complementing rivastigmine's AChE-degrading activity and potentially improving cholinergic neurotransmission. This complex network of interactions could lead to synergistic effects on cognitive function but also potential side effects due to altered amyloid or choline levels.
These are just a few examples of the potential mechanisms that could underlie a synergistic drug-drug interaction between lamotrigine and rivastigmine in Lewy body dementia based on the additional context provided. Further research is needed to fully elucidate all the intricate relationships involved and optimize the use of these two medications together for maximum therapeutic benefit with minimal side effects."""

        final_context = None
        final_rels = set()
        # Query the knowledge graph
        for response in knowledge_graph._call(names_list, 
                                            question,
                                            previous_answer=previous_answer,
                                            generate_an_answer=True, 
                                            progress_callback=progress_callback):
        
            final_context = response["result"]
            all_rels = response['all_rels']
            final_rels.update(all_rels)
            names_to_print = response['names_to_print']

            nodes, edges = parse_relationships_pyvis(all_rels)

            # Create a new container for each similar entity
            ckg_container = st.container()
            with ckg_container:
                with st.chat_message("assistant"):
                    st.write(f"Predicted QA Based on Semantic Similarity:{names_to_print}")
                with st.chat_message("assistant"):
                    create_and_display_network(nodes, edges, '#fff6fe', "CKG", names_list[0], names_list[1])
                with st.chat_message("assistant"):
                    st.write("CKG_Answer:")
                with st.chat_message("assistant"):
                    st.write(final_context)

    nodes, edges = parse_relationships_graph_qa(final_rels)
    index_creator = GraphIndexCreator(llm=llm)
    graph = index_creator.from_text(edges)
    chain = GraphQAChain.from_llm(llm=llm, graph=graph, verbose=True)

    followup_question = st.chat_input("Enter your followup question")
    if followup_question:
        with st.chat_message("user"):
            st.write(followup_question)
        answer = chain.run(followup_question)
        with st.chat_message("assistant"):
            st.write(answer)