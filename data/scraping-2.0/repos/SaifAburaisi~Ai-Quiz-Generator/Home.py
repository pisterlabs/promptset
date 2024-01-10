import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
from dotenv import load_dotenv
from streamlit.components.v1 import html

load_dotenv()

# Function to navigate to another page
def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)


# Function to generate questions
def generate_questions(topic, num_of_questions):

    response_schemas = [
        ResponseSchema(name="question", description="A multiple choice question generated from input text snippet."),
        ResponseSchema(name="options", description="Possible choices for the multiple choice question as a python list."),
        ResponseSchema(name="answer", description="Correct answer for the question.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        template="""Generate {num} mcq's about {top}, where each question has 4 choices and the answer
                 \n{format_instructions}with commas between each question\n""", input_variables=["num", "top"],
        partial_variables={"format_instructions": format_instructions})

    llm = ChatOpenAI(model="gpt-4")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    result = llm_chain.predict(num=num_of_questions, top=topic)
    print(result)
    json_string = re.search(r'```json\n(.*?)```', result, re.DOTALL).group(1)

    python_list = json.loads(f'[{json_string}]')

    questions_dict = {i + 1: question for i, question in enumerate(python_list)}

    return questions_dict


def main():
    st.set_page_config(page_title="Quiz Generator", page_icon="❔")

    st.title("Quiz Generator")
    if "topic" not in st.session_state:
        st.session_state["topic"] = ""
    if "num_of_questions" not in st.session_state:
        st.session_state["num_of_questions"] = ""
    if "data" not in st.session_state:
        st.session_state["data"] = ""

    topic = st.text_input("Topic", st.session_state["topic"])
    num_of_questions = st.text_input("Number of questions", st.session_state["num_of_questions"])

    if st.button("Start Quiz") and topic != "" and num_of_questions != "":

        st.session_state["topic"] = topic
        st.session_state["num_of_questions"] = num_of_questions

        st.write("Please wait until quiz is generated.")

        questions = generate_questions(topic, num_of_questions)
        st.session_state["data"] = questions

        nav_page("quiz")


if __name__ == "__main__":
    main()
