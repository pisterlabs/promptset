# utilities for building tearsheets 
# syntax for metadata filters: https://github.com/langchain-ai/langchain/discussions/10537

import os
import glob
import openai

# langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, StrOutputParser, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma

# authentication
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# where Tearsheets are saved
TEARSHEET_DIR = 'data/tearsheets/tearsheet_{client}.html'


def create_or_load_vectorstore(path, documents=[],
    embedding_function=OpenAIEmbeddings(), override=False):
    '''
    Create or load vectorstore in the specified `path`. If the path exists,
    and `override` is False, the vectorstore is returned. If `path` does
    nor exist OR `override` is True, then the vectorstore is (re)created
    given the list `documents`. The provided `embedding_function` applies
    in either case.
    '''

    if os.path.exists(path) and override==False:  # use existing
        vectordb = Chroma(persist_directory=path,
            embedding_function=embedding_function)
    else:
        # clean out existing data
        for f in glob.glob(os.path.join(path, '**', '*.*'), recursive=True):
            os.remove(f)
        vectordb = Chroma.from_documents(documents, embedding_function,
            persist_directory=path)

    return vectordb


def qa_metadata_filter(q, vectordb, filter, top_k=10,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    '''
    Perform Q&A given a question `q`, vectorstore `vectordb`, and language
    model `llm`. The `top_k` most relevant documents meeting the requirements
    in `filter` are considered.
    '''

    # embed filter in retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k,
                       "filter": filter,})

    """
    # run qa chain with retriever
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    result = qa_chain({"query": q})

    return result['result']
    """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # compare to default prompt: https://smith.langchain.com/hub/rlm/rag-prompt
    template = """Use the following pieces of context to answer the
    question at the end.  Always assume any private, protected, or real-time
    information you would normally not have access to is in the context. If you
    still cannot answer the question from the context, just say you don't know.
    Don't try to make up an answer. Keep the answer as concise as possible
    unless otherwise indicated.
    {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(q)

def llm_chat(msgs=None, human_msg=None, system_msg=None,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    '''
    Chat with specified LLM. Upload list of chat messages (`msgs`) or separately
    provide human message (`human_msg`) and optional system message (`system_msg`).
    '''

    if human_msg:
        msgs = []
        if system_msg:
            msgs.append(SystemMessage(content=system_msg))
        msgs.append(HumanMessage(content=human_msg))

    return llm(msgs).content


def test_response_relevance(answer):
    '''
    Utility function to check the relevance to an answer provided by an LLM.
    '''

    system_msg = '''
        Examine the following answer to a question provided by an AI. Return \
        "No" only if the answer is along the lines of "information not available", \
        "I don\'t know", "I cannot answer". Otherwise return "Yes".
        '''

    return llm_chat(human_msg=answer, system_msg=system_msg)


def create_filter(client_name='all', doc_types='all', logical_operator='$and'):
     '''
     Create a filter for a single client name and/or a list of one or more
     document types.  No filter is applied to a field when the input is 'all'.
     If both filters are present, they are combined with a logical AND or OR
     via `logical_operator`.
     '''

     client_filter, doc_filter = None, None

     if isinstance(client_name, str) and client_name != 'all':
         client_filter = {'client_name': {'$eq': client_name}}

     if doc_types != 'all': # doc filter
         if isinstance(doc_types, str):  # convert to list
             doc_types = [doc_types]

         doc_filter = {'doc_type': {'$in': doc_types}}
    
     values = [f for f in [client_filter, doc_filter] if f is not None]
     if len(values) > 1:
         filter_ = {logical_operator: values}  # combine with $and or $or
     else:
         filter_ = values[0]  # return single dictionary

     return filter_


def load_persona_html():
    '''
    Load HTML documents for synthetic personas all into one dataset.
    '''
    urls = glob.glob('data/text/*html')
    docs = []
    for url in urls:
        doc = UnstructuredHTMLLoader(url).load()
        # infer client name and add to metadata
        root = url.split('/')[-1]
        toks = root.split('_')
        client_name = toks[:-1]
        doc_type = toks[-1][:-5]
        # manually edit metadata
        doc[0].metadata['client_name'] = ' '.join(client_name)
        doc[0].metadata['doc_type'] = doc_type
        docs.extend(doc)

    return docs


def tearsheet_bio(client, vectordb,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    '''
    Build tearsheet bio for a given `client`.

    TODO: decide which details currently in the bio are better placed
    in a summary table.
    '''

    output1 = tearsheet_bio_1(client, vectordb, llm)  # separate q&a
    output2 = tearsheet_bio_2(client, output1, llm)  # consolidate
    output3 = tearsheet_bio_3(output2, llm)  # polish text
    return output3


def tearsheet_table(client, vectordb,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    '''
    Build summary table for a given `client`.
    '''

    output1 = tearsheet_table_1(client, vectordb, llm)  # separate q&a
    return output1


def tearsheet_table_1(client, vectordb, llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    all_docs = create_filter(client, 'all')

    #birthday
    #new client date

    # todo: change keys from number str to text descriptions for easier retrieeval

    multi_doc_prompt_dict = {
      'title':
          {'q': 'what is the current job title of {client}? Answer with just the job title.',
           'f': create_filter(client, 'linkedin'),
           },

      'employer':
          {'q': 'what is the employer of {client}? Answer with just the employer name.',
           'f': create_filter(client, 'linkedin'),
           },


     'location':
         {'q': 'what is the location of {client}? Answer with just the city and state (e.g., city, state).',
          'f': all_docs,
          },

     'net worth': {'q': 'what is the individual and family net worth of {client}? Answer with a pipe-delimited list, e.g., individual net worth | family net worth',
           'f': all_docs,
           },

     'prior positions':
         {'q': '''What prior positions were held by {client}? Answer with a pipe-delimited list, e.g.,\
                Position @ Company 1 | Position2 @ Company 2
               ''',
          'f': all_docs,
          },

     'education':
         {'q': '''What education credentials does {client} have? Answer with a pipe-delimited list, e.g.,\
                Degree 1 (School 1) | Degree 2 (School 2)
               ''',
          'f': all_docs,
          },

     'current boards':
         {'q': '''What boards or committees does the {client} currently serve on? Answer with a pipe-delimited list, e.g.,\
               Board 1 | Board 2
               ''',
          'f': create_filter(client, ['linkedin', 'relsci', 'pitchbook']),
          },

     'previous boards':
         {'q': '''What boards did {client} previously serve on?  Answer with a pipe-delimited list, e.g.,\
               Board 1 | Board 2
               ''',
          'f': create_filter(client, ['linkedin', 'relsci', 'pitchbook']),
          },

     'pitchbook deals': {'q': '''Itemize any deals where the {client} was a lead partner. Answer with a pipe-delimited list, e.g.,\
               Deal 1 | Deal 2
              ''',
           'f': create_filter(client, 'pitchbook'),
           },

     'stock transactions': {'q': '''Itemize the the equity transactions in the last 36 months for {client}. Answer with a pipe-delimited list list, e.g., \
           Stock sold: amount | Options exercised: amount | New equity grants: amount
           ''',
           'f': create_filter(client, 'equilar'),
           },

     'recent news': {'q': '''Itemize news articles about {client}, including title and date. Answer with a pipe-delimited list, e.g.,\
           Title 1 (date 1) | Title 2 (date 2)
           ''',
           'f': create_filter(client, 'google'),  # doc filter also makes difference here
           },
        }

    # answer each question separately
    for key in multi_doc_prompt_dict.keys():
        q = multi_doc_prompt_dict[key]['q'].format(client=client)  # question
        f = multi_doc_prompt_dict[key]['f']  # filter
        response = qa_metadata_filter(q, vectordb, f, llm=llm)  # response
        multi_doc_prompt_dict[key]['a'] = response

        #quality check
        multi_doc_prompt_dict[key]['check'] = test_response_relevance(f'{q} {response}')

    filtered_dict = {key:value for key, value in multi_doc_prompt_dict.items() if value['check'].lower()=='yes'}  # require check == yes

    return filtered_dict


def tearsheet_bio_1(client, vectordb, llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):
    all_docs = create_filter(client, 'all')

    multi_doc_prompt_dict = {
         'current job title': {'q': 'what is the job title of {client} at their employer?',
               'f': create_filter(client, 'linkedin'),
              },

         'work industry': {'q': 'describe the nature (industry, purpose) of the organization where {client} currently works',
               'f': all_docs,
              },

         'prior work history': {'q': 'where did {client} work prior to the current position?',
              'f': all_docs,
             },

         'board memberships': {'q': 'what boards does the {client} currently serve on? What boards did they previously serve on?',
              'f': create_filter(client, ['linkedin', 'relsci', 'pitchbook']),
             },

         'philantropic activities': {'q': 'describe the philantropic activies of {client}',
               'f': all_docs,
              },

         'deals as lead partner': {'q': 'describe any deals where the {client} was a lead partner',
               'f': create_filter(client, 'pitchbook'),
              },

         'investment bio': {'q': 'summarize the investment bio of {client}. Include the amounts of stock and options sold in the last 36 months.',
               'f': create_filter(client, ['equilar', 'pitchbook']),  # applying doc filter here gets more specific and response
              },

         'stocks sold': {'q': 'what stock did {client} sell and when were the effective dates?',
               'f': create_filter(client, 'equilar'),
              },

         'net worth': {'q': 'what is the net worth of {client} and their family',
               'f': all_docs,
              },

         'education': {'q': 'what education credentials does {client} have',

               'f': all_docs,
              },

         'recent news': {'q': 'summarize recent news articles about {client}',
               'f': create_filter(client, 'google'),  # doc filter also makes difference here
              },
        }

    # answer each question separately
    for key in multi_doc_prompt_dict.keys():
        q = multi_doc_prompt_dict[key]['q'].format(client=client)  # question
        f = multi_doc_prompt_dict[key]['f']  # filter
        response = qa_metadata_filter(q, vectordb, f, llm=llm)  # response
        multi_doc_prompt_dict[key]['a'] = response

        multi_doc_prompt_dict[key]['check'] = test_response_relevance(f'{q} {response}')  # check quality of response

    filtered_dict = {key:value for key, value in multi_doc_prompt_dict.items() if value['check'].lower()=='yes'}  # require check == yes

    return filtered_dict


def tearsheet_bio_2(client, qa_dict,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):

    '''
    Given a client and a dictionary of Q&A responses, build a summary
    paragraph that according to the prompt embedded in this function.

    TODO: generate bulleted list dynamically based on keys in the input qa_dict. this will prevent statement conveying no information.
    '''

    bio_prompt_template = '''
        You are a writer and biographer. You specialize in writing
        accurate life profiles given several input documents.
        Below is information on several topics about a single
        client named {client}.

        The context is arranged in the format "topic: information".

        Using this context, write a biography formatted as prose.
        Use matter of fact statements and avoid phrases like "According to ...".
        Do not add an "in summary" or "in conclusion" paragraph at the end of your response.

        Input context:
        {context}

        Your response here:
    '''

    context = ''
    for key in qa_dict:
        context += f'{key}: {qa_dict[key]["a"]}\n\n'

    formatted_prompt = bio_prompt_template.format(client=client, context=context)

    response = llm.call_as_llm(formatted_prompt)
    return response


def tearsheet_bio_3(proposed_bio,
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)):

    '''
    Reformat the output into a single paragraph and adjust language as
    specified in the input prompt.
    '''

    prompt = f'''Reformat the text below. Preserving the order of details but 
    consolidate similar themes into a paragraphs.

    ###
    {proposed_bio}
    ###

    Your response here:
    '''

    response = llm.call_as_llm(prompt)

    return response#.replace('\n', '<br>')


def generate_tearsheet(client, vectordb, override=True):
    '''
    Given client and vectorstore, generate tearsheet components and write.
    If override is False, any existing document will be served. Otherwise,
    a new document is generated if not present or if override is True.
    '''

    # check if document exists
    html, output_path = read_tearsheet_html(client)

    if html is None or override==True:
        print(f'generate_tearsheet for {client}')
        bio = tearsheet_bio(client, vectordb)
        print(f'generate_tearsheet got bio')
        table = tearsheet_table(client, vectordb)
        print(f'generate_tearsheet got table')
        html, output_path = write_tearsheet_html(client, bio, table)

    return html, output_path


def write_tearsheet_html(client, bio, table):
    '''
    Write tearsheet data to html template
    '''    
    # output path
    output_path = TEARSHEET_DIR.format(client=client.replace(" ", "_"))

    # read template
    with open('data/tearsheets/template.html', 'r') as fin:
        template = fin.read()

    # format template
    html = format_template(template, bio, table, client=client, banker='XXX',
        client_type='Client')

    with open(output_path, 'w') as fout:
        fout.write(html)

    return html, output_path


def read_tearsheet_html(client):
    '''
    Read existing tearsheet document. Returns html and path if exists.
    Otherwise returns None, None.
    '''    
    # output path
    input_path = TEARSHEET_DIR.format(client=client.replace(" ", "_"))

    if os.path.exists(input_path):
        # read template
        with open(input_path, 'r') as fin:
            html = fin.read()

        return html, input_path

    else:
        return None, None


def format_template(template, bio, table, client='client', banker='banker',
    client_type='Client'):
    '''
    Format tearsheet template.
    '''

    employer = table['employer']['a']

    formatted_table = '<table class="center" style="border: 1px solid;">'  # table start

    for key, value in table.items():
        formatted_table += f'''
        <tr>
          <td style="border: 1px solid;"><b>{key.capitalize()}:</b> </td>
          <td style="border: 1px solid;">{value["a"]}</td>
        </tr>
        '''

    formatted_table += '</table>'  # table end

    template = template.format(bio=bio, client=client, banker=banker,
        client_type=client_type, employer=employer, table=formatted_table)

    return template


if __name__ == '__main__':
    import tearsheet_utils as m
    docs = m.load_persona_html()
    vectordb = m.create_or_load_vectorstore('data/chroma', docs, override=False) #override=True)
    # test filter 1: require exact match for multiple fields w/ logical OR
    filter_ = {'$or': [{'client_name': {'$eq': 'Robert King'}},
            {'doc_type': {'$eq': 'linkedin'}}]}
    junk = vectordb.similarity_search('summarize the current employers of all people', k=99, filter=filter_)
    for d in junk: print(d.metadata)

    # test filter 2: match to lists of values with logical AND
    filter_ = {'$and': [{'client_name': {'$in': ['Robert King']}},
            {'doc_type': {'$in': ['linkedin', 'relsci', 'equilar']}}]}
    junk = vectordb.similarity_search('summarize the current employers of all people', k=99, filter=filter_)
    for d in junk: print(d.metadata)

    # test create_filter
    filter1 = m.create_filter('Robert King', 'all')
    filter2 = m.create_filter('Robert King', 'linkedin')
    filter3 = m.create_filter('Robert King', ['linkedin', 'google'])
    filter4 = m.create_filter('all', ['google'])

    # test Q&A for filters
    q1 = 'What is noteworthy about Robert King?'
    q2 = 'where does Robert King currently work?'
    q3 = 'What important roles does Robert King have in the community?'
    q4 = 'Summarize all the recent news articles based on their titles'

    r1 = m.qa_metadata_filter(q1, vectordb, filter1)
    r2 = m.qa_metadata_filter(q2, vectordb, filter2)
    r3 = m.qa_metadata_filter(q3, vectordb, filter3)
    r4 = m.qa_metadata_filter(q4, vectordb, filter4)

    # test tearsheet bio functions separately
    output1 = m.tearsheet_bio_1('Robert King', vectordb)
    output2 = m.tearsheet_bio_2('Robert King', output1)
    output3 = m.tearsheet_bio_3(output2)

    # test tearsheet bio
    bio1 = m.tearsheet_bio('Robert King', vectordb)
    bio2 = m.tearsheet_bio('Velvet Throat', vectordb)
    bio3 = m.tearsheet_bio('Julia Harpman', vectordb)

    # test tearsheet table functions separately
    table1 = m.tearsheet_table('Robert King', vectordb)
    table2 = m.tearsheet_table('Velvet Throat', vectordb)
    table3 = m.tearsheet_table('Julia Harpman', vectordb)

    # write tearsheet
    html, output_path = m.generate_tearsheet('Robert King', vectordb)  # generates bio/table internally
    #html, output_path = m.write_tearsheet_html('Robert King', bio1, table1)
    html, output_path = m.generate_tearsheet('Velvet Throat', vectordb)
    html, output_path = m.generate_tearsheet('Julia Harpman', vectordb)
    #html, output_path = m.write_tearsheet_html('Julia Harpman', bio3, table3)
