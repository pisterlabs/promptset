import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

import pinecone
from langchain.vectorstores import Pinecone
import os
from langchain.vectorstores import Vectara

local = False


client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

original_system_message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about the contents of the PhD thesis. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about the PhD thesis. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)

thesis_summary_system_message = SystemMessage(
    content=(
       """You are a helpful chatbot who is tasked with answering questions about the contents of the PhD thesis. 
            Unless otherwise explicitly stated, it is probably fair to assume that questions are about the PhD thesis. 
            If there is any ambiguity, you probably assume they are about that.
            This is a summary of the thesis:
                
Chapter summary chapter 1:      

Chapter 1 of the PhD thesis explores the viability and desirability of an analytical perspective on knowledge in the Norwegian AEC industry. The chapter begins by highlighting the significant impact of the AEC industry on the Norwegian economy, energy consumption, and climate footprint. It governs key societal resources such as land use, infrastructure, housing, and public space, making the built environment crucial for daily life.

The chapter delves into the increasing complexity of the AEC industry, driven by various factors such as changes in operating parameters, organization, and societal expectations. This complexity contributes to project failure and a decline in productivity. The lack of formalization in the industry's principles and methods is identified as a key issue hindering the industry's ability to address root problems.

To address these challenges, the chapter proposes adding an epistemological perspective to the production of solutions in the AEC industry. This involves formalizing existing knowledge and understanding the underlying mechanisms and principles that enable knowledge integration. The chapter emphasizes the need for a comprehensive formalization of the industry's knowledge integration principles to develop a common language and understanding across its heterogeneous elements.

The research approach outlined in the chapter involves adding an epistemic perspective to the production of solutions in the AEC industry. The aim is to develop a reflective understanding of the knowledge that shapes the physical structure of society. The chapter discusses the importance of formalization and the development of methods to uncover the underlying mechanisms of industry practices. It also highlights the value of developing a vocabulary and analytical platform for analyzing knowledge in the AEC industry, bridging epistemological descriptions with industry applications.

The chapter situates the research on Poincaré's knowledge curve, positioning it in the first to third phases of knowledge development. This involves trial-and-error exploration, formalization of the method, and the development of a logic of discovery. The research offers new observations of industry practices, the development of a vocabulary for empirical knowledge analysis, and the potential for tools and applications for knowledge management in the AEC industry.

Overall, the chapter emphasizes the need for an analytical perspective on knowledge in the AEC industry and highlights the potential benefits of formalization and understanding the underlying mechanisms of industry practices.

Chapter summary chapter 2:

Chapter 2 of the PhD thesis focuses on establishing a theoretical vocabulary to describe the universal epistemic characteristics of the Architecture, Engineering, and Construction (AEC) industry. The chapter aims to develop a vocabulary that can be used to study the unique features of the industry and adapt existing empirical methods. The chapter follows an analytical procedure consisting of three steps: articulating and constraining the theoretical problem, designing a theoretical response to the problem, and classifying the design within a philosophical context.

The first part of the chapter focuses on establishing the theoretical foundation for formalization in the AEC industry. The industry lacks systematic formalization of its enabling logics and knowledge content, as well as a functional epistemic vocabulary. The research aims to address this gap by developing a theoretical foundation for the formalization of shared mechanisms in the industry. The chapter also emphasizes the importance of acknowledging and handling epistemic heterogeneity as a fundamental constraint on formalization.

The second part of the chapter responds to the constraints by designing a theoretical framework. It frames the industry phenomenologically and analyzes the common denominators of the AEC industry production process. The section concludes with reflections on the scientific characteristic and status of the theoretical approach.

The third part of the chapter focuses on classifying the theoretical platform within a philosophical context. It reflects on the underlying naturalism and pragmatic attitude towards phenomena in the research. The pragmatic attitude is critical to understanding the theoretical vocabulary and the perspective, limitations, and output of the research.

The chapter also establishes several premises. Firstly, it highlights the need for a theoretical foundation for the formalization of shared mechanisms in the AEC industry. Secondly, it emphasizes the empirical analysis of building information as a primary data source for understanding industry knowledge. Building information contains knowledge, but often in indirect ways, and there is a need to make the indirect knowledge aspects explicit. Thirdly, the chapter acknowledges heterogeneity as an epistemic constraint in the AEC industry. The industry consists of different kinds of knowledge, and this heterogeneity must be accounted for in the study of industry knowledge. Finally, the chapter acknowledges the problematic limits of knowledge in the industry, including the difference between knowing that and knowing how, the presence of lies in the industry, and the practical limits of knowing due to time constraints.

Overall, the chapter provides a theoretical vocabulary and framework for studying the epistemic characteristics of the AEC industry. It emphasizes the need to address the heterogeneity and practical constraints of industry knowledge and provides a foundation for further research in the field. The chapter also introduces the concept of operationalization, which involves the practical integration of knowledge in the production process and the operationalization of future actions through the affordances of the physical building. The chapter also discusses the pragmatic nature of scientific inquiry and the importance of maintaining an open and self-correcting approach to knowledge discovery.

Chapter summary chapter 3:

Chapter 3 of the PhD thesis focuses on the methodological approach used to operationalize an analytical perspective on the data from the Architecture, Engineering, and Construction (AEC) industry. The chapter begins by discussing the ambition of bridging the gap between theoretical concepts and empirical data in order to analyze the indirect knowledge content of building industry information. The methodical approach is seen as a reflection of a learning process, with the end goal being the generalization of conceptual insights into a technique or set of rules.

The chapter outlines the components of the method, including a description of the main methodical problem, the research design chosen, a timeline of the process, the formalization of the technique, a summary of the technique, and the limits of the technique. The main methodical problem is identified as the challenge of making the non-salient knowledge aspect of information in the AEC industry salient and open to analysis. Knowledge in the industry is not manifested in traditional epistemological statement forms, but rather in the quality of actions. The chapter argues that epistemic unity in the industry manifests as knowledge integration in practice, and this integration can be established through the mediation of controversies during the action sequence.

The research design is described as a bottom-up learning process, where the technique is generated through trial and error and the analysis of industry data. The research papers attached in the appendices are summarized, including their context, data sources, and research problems. The papers cover topics such as the modal descriptions of buildings in the Norwegian building code, changes in the technical regulations for garages, and the epistemic differences and similarities between key actors in a shopping center development project.

The research process is visualized as a genealogy of technique, showing the timeline of events and the development of the technique over time. The elements of the technique are classified and sorted based on their origins in the research process. The chapter concludes by highlighting the importance of formalizing the technique and making it available to the research community for further application and development.

The technique section of the chapter provides a description and systematization of the techniques from the research papers, presenting them as a particular set of rules for dealing with empirical data. The aim is to present a generalization of the technical aspect of the research that is accessible to other researchers without having to read the papers. The technique is divided into four parts: data sources, analysis, modeling, and a step-by-step procedure. The data sources section discusses the collection and sampling aspect of the technique, focusing on building information as the data of interest. The analysis section explains how the data is processed and accentuates the adverbial aspect of information. The modeling section discusses the use of diagrammatic representations to compare and analyze the data. Finally, the step-by-step procedure summarizes the technique and its application.

The chapter also addresses the limitations of the technique, including its scope, reliability, and validity. The scope is limited to the specific aspects of the AEC industry that were studied, and further research is needed to expand the technique to a more comprehensive methodology. The reliability of the technique is dependent on the researcher's judgment, but statistical methods can be used to improve confidence and measure uncertainty. The validity of the technique is rooted in its ability to provide a new perspective on industry information, but it is important to recognize that this perspective is a representation and not a true or false statement.

Overall, the chapter provides a detailed overview of the methodological approach used in the research, highlighting the importance of bridging the gap between theory and empirical data in the AEC industry. The technique developed in the research papers is presented as a tool for analyzing and understanding the knowledge integration in the industry, with the aim of further development and application in future research.

Chapter summary chapter 4:

Chapter 4 of the PhD thesis delves into the logic of discovery and the need to bridge the gap between analysis and application in the Architecture, Engineering, and Construction (AEC) industry. The chapter argues that a logical procedure is necessary to connect philosophical inquiry with industry applications, in order to make the research findings valuable to industry practitioners.

The chapter begins by highlighting the disparity in abstraction between philosophical inquiry and industry applications in the AEC industry. It stresses the importance of a logical procedure that can link epistemological observations to actionable principles that can be tested in practice. The chapter asserts that the research technique outlined in the method section must be interpreted and contextualized to extend its value beyond pure research.

To address this gap, the chapter introduces the concept of the logic of discovery as a means to bridge the divide between analysis and application. The logic of discovery aims to transform the deductive reading of data produced by the research technique into something that is testable and actionable in the AEC industry. The chapter also introduces the concept of levels within the logic of discovery, representing different layers of knowledge or insight.

The first level of the logic of discovery involves diagrammatic representations of new totalities or fresh interpretations of industry information. These representations provide insights into the constraints and requirements of the AEC industry. The second level entails identifying patterns in the data and recognizing salient features that demand explanation. The third level involves completing the observations through abductive inference, resulting in the formulation of testable action principles.

The chapter provides concrete examples for each level of the logic of discovery. These examples include diagrammatic representations of the Norwegian building code and a shopping center development, comparisons between different regulatory descriptions, and timelines illustrating the historical development of building codes.

In conclusion, the chapter discusses the implications and potential applications of the logic of discovery. It suggests that the logic of discovery can be utilized to generate new hypotheses and action principles that can be tested in practice. The chapter emphasizes the importance of validation criteria and proposes methods for validating the findings of the logic of discovery.

Overall, the chapter offers a comprehensive exploration of the logic of discovery and its potential applications in the AEC industry. It underscores the significance of bridging the gap between analysis and application to ensure that research findings are valuable to industry practitioners.

Chapter summary chapter 5:

Chapter 5 of the PhD thesis focuses on the prospects for a new perspective on knowledge formalization in the Architecture, Engineering, and Construction (AEC) industry. The chapter begins by recapping the knowledge gap addressed in the dissertation and summarizing the research approach developed to address this gap. The author then outlines new frontiers of knowledge formalization in the AEC industry that emerged during the inquiry. The concrete recommendations for further research, method development, theoretical generalization, and knowledge management applications are found in the conclusions chapter.

The chapter highlights the lack of epistemological description and formalization in the AEC industry, specifically in terms of the enabling interface between different professional actors and the universally enabling knowledge. The author distinguishes this knowledge gap from the form and content of each professional agency's contribution during the building process. The gap focuses on the formalization of the epistemological glue that exists in the practical, but often tacit, action principles that enable different agencies to work together to produce buildings.

The chapter discusses the reasons why a unifying perspective on knowledge has not been developed in the AEC industry, including the industry's focus on new solutions rather than formalizing existing knowledge, the practical nature of the industry prioritizing getting things done over reflection, new responsibility structures increasing specialization and fragmentation of knowledge, the absence of an overall perspective on knowledge exchange, the shift in leadership from architects to consultants, and the lack of attention from philosophers.

The chapter also emphasizes the need for a theoretical vocabulary that can capture the knowledge component of the AEC industry and enable the analysis of enabling actions. It highlights the need for a methodical way of connecting analytical descriptions of knowledge to industry applications to provide concrete value. The chapter proposes a logical procedure, called the logic of discovery, that generates new potential action principles based on new knowledge descriptions and can be tested and validated in practice.

Furthermore, the chapter discusses the limitations of the research, including the researcher's judgment in the results, the partial representation of reality, and the need for further statistical methods to improve confidence in the judgment process. It emphasizes the importance of the adverbial qualities of information in determining validity.

Overall, the chapter provides a detailed analysis of the knowledge gap in the AEC industry and proposes a research approach and technique to address this gap. It highlights the need for a unifying perspective on knowledge, a theoretical vocabulary, and a methodical way of connecting analytical descriptions to industry applications. The chapter sets the stage for further research and development in the field of knowledge formalization in the AEC industry. The new context provided in section 5.2.3 expands on the logic of discovery as a formalized feedback loop between analytical observations and industry applications, and section 5.3 explores new frontiers and deficiencies in knowledge formalization in the AEC industry. The chapter concludes by discussing the potential consequences of leaving these deficiencies unattended and the value of systematic formalization and digitalization in the industry.

Chapter summary chapter 6:

Chapter 6 of the PhD thesis titled "Conclusions" discusses the proof of the thesis and the key findings of the research. The chapter begins by highlighting the lack of a satisfactory epistemological description in the contemporary building industry, particularly at the interfaces between different professional disciplines. The aim of the research was to explore the plausibility and actionability of this claim and to address the root problems faced by the AEC industry in Norway and globally.

The author confirms the lack of analytical descriptions of interfaces between disciplines and professional interests in the AEC industry through empirical observations. Three key empirical observations are highlighted: the distortion caused by the Norwegian building code in terms of technological descriptions and the neglect of aesthetic and social aspects, the fear of unknown consequences of new technologies in building codes, and the lack of a single actor overseeing the conceptual development in a project organization.

These observations support the thesis and demonstrate the need for a theoretical and methodological platform to describe and address the knowledge interfaces in the industry. The research has produced compelling evidence that this can be done using the argument, vocabulary, method, and logic outlined in the dissertation. The findings and insights generated by applying this framework are valuable to the industry itself, not just to philosophers and specialist researchers.

The author organizes the evidence into three categories: new observations and analysis of industry data, the vocabulary, techniques, and logic developed in the research, and direct applications in the form of testable action principles and knowledge management tools. These categories validate the viability and value of studying the AEC industry from an applied, epistemological perspective.

The chapter concludes by summarizing the proof of the thesis and suggesting future directions for research. These include expanding the theoretical framework, applying it to more industry sources, conducting comparative studies, developing new analytical parameters, validating and testing hypotheses, comparing the framework with other approaches, and automating the analytical procedure.

The chapter also presents a synthesis of the research findings in relation to Møystad's theory of cognitive reciprocity between the brain and built environments. The author suggests a model of the production of built environments as cognition, which involves the discovery of insights, inscription into the collective horizon, decision-making, and changes in memory structure.

In conclusion, the research has provided evidence of the lack of a satisfactory epistemological description in the AEC industry and has proposed a theoretical and methodological framework to address this issue. The findings have practical implications for knowledge management in the industry and suggest future research directions. The research has demonstrated the plausibility and actionability of the claim and has generated valuable insights and tools for the industry practitioner. The chapter also highlights the potential for further research and the expansion of the theoretical framework to other contexts and approaches. Overall, the research contributes to a better understanding of the interfaces between disciplines and professional interests in the AEC industry and provides a foundation for improving knowledge integration and management in building projects.
"""
    )
)   

@st.cache_resource(ttl="1h")
def configure_retriever(vectorstore_choice='Pinecone'):
    if vectorstore_choice == 'Pinecone':
        if local:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
                environment = os.getenv("PINECONE_ENV_FINN")  # next to api key in console
            )
            embeddings = OpenAIEmbeddings() 
        else:
            pinecone.init(
                api_key=st.secrets["PINECONE_API_FINN"], 
                environment = st.secrets["PINECONE_ENV_FINN"]
            )
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

        index_name = "jorgen-phd-thesis"
        vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        return retriever
    
    elif vectorstore_choice == 'Vectara':
        if local:
            vectara_customer_id= '' #st.secrets["vectara_customer_id"]
            vectara_corpus_id= '' #st.secrets["vectara_corpus_id"]
            vectara_api_key= '' #st.secrets["vectara_api_key"]
        else:
            vectara_customer_id=st.secrets["vectara_customer_id"]
            vectara_corpus_id=st.secrets["vectara_corpus_id"]
            vectara_api_key=st.secrets["vectara_api_key"]

        vectorstore = Vectara(
                vectara_customer_id=vectara_customer_id,
                vectara_corpus_id=vectara_corpus_id,
                vectara_api_key=vectara_api_key
        )
        retriever = vectorstore.as_retriever(n_sentence_context=200)

        return retriever

def reload_llm(model_choice="gpt-4", temperature=0, vectorstore_choice="Pinecone", system_message_choice="Original"):
    if local:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, )
    else:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, openai_api_key=st.secrets["openai_api_key"])

    if system_message_choice == "Original":
        message = original_system_message
    elif system_message_choice == "New (extensive summary)":
        message = thesis_summary_system_message

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )

    tool = create_retriever_tool(
        configure_retriever(vectorstore_choice),
        "search_phd_thesis",
        "Searches and returns text from PhD thesis. This tool should be used to get additional details about the contents of the PhD Thesis.",
    )
    tools = [tool]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    memory = AgentTokenBufferMemory(llm=llm)
    print ("Reloaded LLM")
    return agent_executor, memory, llm


# Using "with" notation
with st.sidebar:
    with st.form('my_form'):
        model_choice = st.radio(
            "Model",
            ("gpt-4", "gpt-3.5-turbo-16k")
        )
        temperature = st.slider('Temperature', 0.0, 1.0, 0.0, 0.01)
        vectorstore_choice = st.radio(
            "Vectorstore",
            ("Pinecone", "Vectara")
        )
        system_message_choice = st.radio(
            "System message",
            ("Original", "New (extensive summary)")
        )
        submitted = st.form_submit_button('Reload LLM')
    if submitted: 
        reload_llm(model_choice=model_choice, temperature=temperature)
        print(model_choice, temperature)
    

"# Chat🦜🔗"


starter_message = "Ask me the PhD thesis!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

agent_executor, memory, llm = reload_llm()

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        agent_executor, memory, llm = reload_llm(model_choice=model_choice, temperature=temperature, vectorstore_choice=vectorstore_choice, system_message_choice=system_message_choice)
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
        print(llm)
