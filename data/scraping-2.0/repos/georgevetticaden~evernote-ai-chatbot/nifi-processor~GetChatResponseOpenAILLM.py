# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import json
import re

from langchain import PromptTemplate, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.properties import ExpressionLanguageScope
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


class GetChatResponseOpenAILLM(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        dependencies = ['langchain', 'openai', 'pinecone-client','tiktoken']
        version = '0.0.1-SNAPSHOT'
        description = 'Performs a similarity search in Pinecone based on the query/question that is asked and returns list of similar text docs with metadata.'
        tags = ['Pinecone','OpenAI', 'AI', 'Vector Similarity Search', 'Vector Database']


    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.openai_api_key = PropertyDescriptor(
            name="OpenAI API Key",
            description="The API key to connect to OpeanAI services",
            required = True,
            sensitive = True
        )

        self.openai_llm_model = PropertyDescriptor(
            name="OpenAI LLM Model",
            description="The OpenAI LLM model to answer the user question/query",
            required = True,
            default_value = "gpt-3.5-turbo",
            allowable_values = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']
        )

        self.openai_embedding_model = PropertyDescriptor(
            name="OpenAI Embedding Model",
            description="The OpenAI embedding model to use to convert query/question to a text embedding which is then used to search for similar docs.",
            required = True,
            default_value = "text-embedding-ada-002",
            allowable_values = ['text-embedding-ada-002', 'text-davinci-001', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
        )

        self.openai_llm_temperature= PropertyDescriptor(
            name="LLM temperature",
            default_value="0",
            description="The temperature controls how much randomness is in the output. O means no randomness while 1 means high randomness. Valid values from 0-1",
            required=True,
        )

        self.question = PropertyDescriptor(
            name="question",
            default_value="0",
            description="The question/chat that the LLM needs to answer/respond to.",
            required=True,
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES
        )

        self.chat_history = PropertyDescriptor(
            name="chat_history",
            default_value="0",
            description="The previous chat history so the LLM has more context",
            required=True,
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES
        )


        self.pinecone_api_key = PropertyDescriptor(
            name="Pinecone API Key",
            description="The API key to connect to the Pinecone to get relevant documents for the question",
            required = True,
            sensitive = True
        )
        self.pinecone_environment_name = PropertyDescriptor(
            name="Pinecone Env Name",
            description="The environment that the Pinecone index is located.",
            required = True,
            sensitive = True
        )

        self.pinecone_index_name = PropertyDescriptor(
            name="Index Name",
            description="The Pinecone index to store the embeddings.",
            required = True,
        )

        self.pinecone_namespace = PropertyDescriptor(
            name="Namepace",
            description="The  namespace within the index to store the vector embeddings.",
            required = True,
        )

        self.search_results_size= PropertyDescriptor(
            name="Number of Similar Documents to Return",
            default_value="10",
            description="The number of similar documents to return from the similarity searech",
            required=True,
        )

        self.user= PropertyDescriptor(
            name="User Name",
            description="The name of the user asking the questions.",
            required = True,
        )

        self.descriptors = [self.openai_api_key, self.openai_llm_model, self.openai_embedding_model, self.openai_llm_temperature, self.question, self.chat_history, self.pinecone_api_key, self.pinecone_environment_name, self.pinecone_index_name, self.pinecone_namespace, self.user, self.search_results_size]

    def onScheduled(self, context):
        self.logger.info("Initializing OpenAI and Pinecone Services.")

        # Get the properties from the processor needed to configure the OpenAI Embedding Service
        openai_api_key = context.getProperty(self.openai_api_key.name).getValue()
        embeddings_model_name = context.getProperty(self.openai_embedding_model.name).getValue()
        openai_embedding_service = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embeddings_model_name)

        # Initialize Pinecone and get the index we will be upserting into.
        pinecone_api_key =  context.getProperty(self.pinecone_api_key.name).getValue()
        pinecone_environment = context.getProperty(self.pinecone_environment_name.name).getValue()
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        pinecone_index = context.getProperty(self.pinecone_index_name.name).getValue()
        namespace = context.getProperty(self.pinecone_namespace.name).getValue()

        pinecone_vector_store = Pinecone.from_existing_index(index_name=pinecone_index, embedding=openai_embedding_service, namespace=namespace)

        temperature = context.getProperty(self.openai_llm_temperature.name).getValue()
        llm_model_name = context.getProperty(self.openai_llm_model.name).getValue()


        _template = """Given the following extracted parts of a long document and a question,
        create a final answer with references ("SOURCES") unless identified below.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        But if you are asked something similar to what your purpose is as an AI Assistant, then answer with the following:
        I'm a helpful assistant for {username} answering his questions based on the notes within his Evernote.
        Also, ALWAYS return a "SOURCES" part in your answer.

        QUESTION: Which state/country's law governs the interpretation of the contract?
        =========
        Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
        Source: 28-pl
        Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
        Source: 30-pl
        Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
        Source: 4-pl
        =========
        FINAL ANSWER: This Agreement is governed by English law.
        SOURCES: 28-pl

        QUESTION: What did the president say about Michael Jackson?
        =========
        Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
        Source: 0-pl
        Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
        Source: 24-pl
        Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
        Source: 5-pl
        Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
        Source: 34-pl
        =========
        FINAL ANSWER: The president did not mention Michael Jackson.
        SOURCES:

        QUESTION: {question}
        =========
        {summaries}
        =========
        FINAL ANSWER:"""

        QA_PROMPT = PromptTemplate.from_template(_template)


        # Create the chain
        chat_llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model=llm_model_name)
        question_generator = LLMChain(llm=chat_llm, prompt=CONDENSE_QUESTION_PROMPT,  verbose=True)
        doc_chain = load_qa_with_sources_chain(chat_llm, chain_type="stuff", verbose=False, prompt=QA_PROMPT)

        self.qa_chain = ConversationalRetrievalChain(
            retriever=pinecone_vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            verbose=False
        )


    def transform(self, context, flowFile):

        # Get the user asking the question
        user = context.getProperty(self.user.name).getValue()

        # Get the question we are asking the LLM
        question = context.getProperty(self.question.name).evaluateAttributeExpressions(flowFile).getValue()

        # Get the chat history and escape it before passing to Question/Answer LLM Chain
        chat_history = context.getProperty(self.chat_history.name).evaluateAttributeExpressions(flowFile).getValue()
        escaped_chat_history = chat_history.replace('\n', '-')

        self.logger.info("Escaped Chat History is: "  + escaped_chat_history)

       # array_of_tuples_chat_history = ast.literal_eval('[' + escaped_chat_history + ']')

        regex_pattern = r'\((.*?)\)'
        array_of_tuples_chat_history = re.findall(regex_pattern, chat_history)

        # Split each tuple string into individual values
        array_of_tuples_chat_history = [tuple_str.split('", "') for tuple_str in array_of_tuples_chat_history]

        # Create tuples by stripping the quotes and whitespace from each value
        array_of_tuples_chat_history = [(value[1:-1], value2[1:-1]) for value, value2 in array_of_tuples_chat_history]

        self.logger.info("********* Inside transform of GetChatResponseOpenAILLM with question: "+question)

        # Use the Langchain Question Answer Chain to look up relevant documents in pinecone and then ask LLM the question with the contextual data
        result = self.qa_chain({"username": user, "question": question, "chat_history": array_of_tuples_chat_history})

        answer = result["answer"]
        answer = answer.replace('\n', '').replace("'", "\\'")

        self.logger.info("LLM answerrrstxz that is escaped for question[" + question + "] is: " + answer)

        return FlowFileTransformResult(relationship="success", contents=answer)


    def getPropertyDescriptors(self):
        return self.descriptors
