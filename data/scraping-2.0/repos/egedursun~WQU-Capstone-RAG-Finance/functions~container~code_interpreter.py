import datetime
import io
import contextlib

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.tools import tool

import pandas as pd
import numpy as np
import arch
import matplotlib.pyplot as plt

import streamlit as st

dotenv.load_dotenv()
config = dotenv.dotenv_values()


@tool("run_code", return_direct=True)
def run_code(query):
    """
    Run code in a sandboxed environment.

    :param query:
    :return:
    """

    current_date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                     streaming=False,
                     model_name="gpt-4",
                     temperature="0.5",
                     max_tokens="2048")

    with st.spinner("Internal Agent [Code Interpreter Agent] is transforming your request..."):

        response = llm(
            [ChatMessage(role="user", content=f"""
                                        The user asked the following query to another GPT agent:

                                        - {query}

                                        - Here is the current date in case you might need it: {current_date_string}

                                        ---

                                        Based on the user's query, you need to run the code in a sandboxed environment.
                                        You should first create a code snippet that will be executed in the sandboxed
                                        environment. The code snippet should be able to return a value, and the value
                                        should be returned as a string, for you to be able to return it to the other 
                                        agent.
                                        
                                        *** Sometimes, the user might send you a piece of data that you can use to
                                        calculate something, for instance, GARCH/ARCH may require inputs from the user
                                        which can be included in the query. However, if its not included, you can
                                        assume default parameters, and calculate accordingly. However, if you assumed
                                        default parameters, please include them in the context information you return.
                                        
                                        The libraries you can use are:
                                        - pandas
                                        - numpy
                                        - arch
                                        - matplotlib.pyplot
                                        
                                        Do not forget to import the libraries you need. Be careful for the 
                                        indentation, and the syntax of the code.
                                        
                                        IMPORTANT NOTE:
                                        - The answer you should return should not be directly the result of the code,
                                        but you should also provide the context explaining what is the result, 
                                        and what is the meaning of the result. For example, if the result is a
                                        Pandas DataFrame, you should explain what is the data inside it, and what
                                        does it represent. 
                                        - You can include this context information in the returned string.
                                        
                                        !!!
                                        * Do NOT share ANYTHING except for directly RUNNABLE/EXECUTABLE code.
                                        * Do NOT even put ```python or ``` at the beginning of the code. 
                                        * Do NOT put ``` at the end of the code as well.
                                        * ALWAYS 'PRINT' THE RESULT OF THE CODE and THE CONTEXT INFORMATION.
                                        !!! 

                                        ---
                                    """)]
        )
        st.success("Internal Agent [Code Interpreter Agent] has transformed your request successfully.")

        with st.spinner("Internal Agent [Code Interpreter Agent] is querying the API..."):

            # Create a string buffer to capture the output
            output_buffer = io.StringIO()

            # Capture all output of the exec() function
            with contextlib.redirect_stdout(output_buffer):
                try:
                    # Execute the code
                    exec(response.content)
                except Exception as e:
                    # If there's an error, capture the error message
                    return f"An error occurred: {e}"

            data = output_buffer.getvalue()
            data = str(data)
        st.success("Internal Agent [Code Interpreter Agent] has queried the API successfully.")

    with st.expander("Reference Data [Code Interpreter API]", expanded=False):
        st.warning("\n\n" + str(data) + "\n\n")
        st.warning("Source: \n\n [1] Sandbox Code Interpreter")

    return data


if __name__ == "__main__":
    # test the interpreter
    query = "Do a simple GARCH calculation"
    print(run_code(query))
