from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

class QuestionFormulator():
    def __init__(self, model="gpt-3.5-turbo") -> None:
        self.model = ChatOpenAI(model_name=model)

    def formulate_question(self, summary, category, num_goals=5):

        if category == "Market Dataset":
            list_of_graph = ["OHLC Chart", "Moving Average Graph", "RSI Graph","Candlestick Chart", "Moving Average Convergence Divergence",
                "Waterfall Chart", "Funnel Chart", "Time Series graph with a range slider","Multi-Measurment Time Series Graph", ]
        else:
            list_of_graph = ["Multi-Measurment Time Series Graph", "Time Series graph with a range slider"]
    
        # Read the base system prompt from the file
        with open('autofinviz/components/question/prompts/format_instruction.tmpl', 'r') as file:
            format_instruction = file.read()

        # Read the base system prompt from the file
        with open('autofinviz/components/question/prompts/graph_description.tmpl', 'r') as file:
            graph_description = file.read()
        

        user_prompt = f"""

            CHOOSE {num_goals} plots, that fit for dataframe df, among {list_of_graph} as the visulization_type 
            SET a QUESTION as the TITLE of the plot.
            CHOOSE columns name for x axis and y axis.

            FOLLOW this output format instruction:

            {format_instruction}

            'visualization_type' can ONLY be within {list_of_graph}.

        """

        prompt ="""
            Given the summary of dataframe: {{summary}}
            Given description of each graph: {{graph_description}}

            {{user_prompt}}
        """

        prompt = PromptTemplate.from_template(prompt, template_format="jinja2")
        chain = prompt |  self.model | SimpleJsonOutputParser()
        question_result = chain.invoke({"summary": summary, "graph_description": graph_description, "user_prompt": user_prompt})

        return question_result