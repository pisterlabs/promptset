import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


from autofinviz.utils import preprocess_code, get_globals_dict

class Visualizer():
    def __init__(self, model="gpt-3.5-turbo") -> None:
        model = ChatOpenAI(model_name=model)

        ## Load .py file
        loader = GenericLoader.from_filesystem(
            "notebooks/graph",
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        documents = loader.load()

        ## Split the text in .py file
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
        )
        texts = python_splitter.split_documents(documents)

        ## Embedding
        db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
        retriever = db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 5},
        )

        # RAG template
        prompt_RAG = """
            You are a proficient python developer. Respond with the syntactically correct code for to the question below. Make sure you follow these rules:
            1. Use context to understand the APIs and how to use it & apply.
            2. Do not add license information to the output code.
            3. Do not include colab code in the output.
            4. Ensure all the requirements in the question are met.

            Question:
            {question}

            Context:
            {context}

            Helpful Response :
            """

        prompt_RAG_tempate = PromptTemplate(
            template=prompt_RAG, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_llm(
            llm=model, prompt=prompt_RAG_tempate, retriever=retriever, return_source_documents=True
        )


    def visualize(
        self, 
        questions: list,
        data: pd.DataFrame
    ):
        
        visualizer_results = []
        
        for viz in questions:

            df = data.copy()

            print(viz)

            question = f"""Write a python function that takes df dataframe and plot {viz['visualization_type']} in plotly library.
            USE the X_column {viz["x_axis"]}, Y_columns {viz["y_axis"]}
            SET title as "{viz['title']}".
            ONLY Return executable PYTHON code.
            NAME the function as plot(df)

            TEMPLATE:

            ```
            import ...

            def plot(df: pd.dataframe):

                return fig

            # DO NOT MODIFY ANYTHNIG BELOW
            # Dataframe 'df' is already defined so NO NEED TO CONSTRUCT DATAFRAME "df"
            fig = plot(df)
            
            ```
            THE OUTPUT SHOULD ONLY USE THE PYTHON FORMAT ABOVE.
            MUST execute the plot(df) after defining the plot().
            """
            def generate_code(question):
                results = self.qa_chain({"query": question})
                code = results["result"]
                return code
            
            count = 0

            while count < 3:

                code = generate_code(question)

                try:
                    code = preprocess_code(code)
                    global_dict = get_globals_dict(code, df)
                    exec_vars = {}

                    print(code)
                    exec(code, global_dict, exec_vars)

                    fig = exec_vars.get('fig')
                    # fig.write_image("example/figures/{viz['title']}.png")
                    # fig.show()
                    
                    print(type(fig))
                    if fig != None:
                        visualizer_results.append({"fig": fig, "code": code})
                        break
                    else:
                        print("Function was not executed")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    count += 1
        
        return visualizer_results