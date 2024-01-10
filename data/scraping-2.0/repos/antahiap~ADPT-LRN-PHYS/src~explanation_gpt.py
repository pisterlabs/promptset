import ast
from openai_api import OpenAIApi
from openai_multi_client import OpenAIMultiClient
from constants import PAPER_GPT_PATH
from database import keyword_db
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class ExplanationGPT():
    def __init__(self, topic, explanation=None, context=None):
        self.api = OpenAIApi("gpt-3.5-turbo-16k")
        self.topic = topic
        self.explanation = explanation
        self.context = context
        self.keywords = set()
        self.keywords_explanations = {}
        self.html = None
        self.info_path = PAPER_GPT_PATH / f"{self.topic}.json"

    def generate_info(self):
        self.fill_from_db()
        if not self.explanation:
            self.generate_explanation()
        if not self.keywords:
            self.find_keywords()
        self.fill_explanation_from_db()
        self.explain_keywords()
        self.to_html()

    def fill_from_db(self):
        res = keyword_db.select(self.topic)
        if res:
            self.explanation = res[0][2]
            if res[0][3]:
                self.keywords = set(res[0][3])

    def explanation_prompt(self, topic, context):
        return f"{context}\n\nAnswer using markdown. Explain '{topic}' in every details using markdown."

    def set_explanation(self, explanation):
        self.explanation = explanation
        keyword_db.insert_explanation(self.topic, self.explanation)

    def generate_explanation(self):
        logging.info("Generating explanation")
        prompt = self.explanation_prompt(self.topic, self.context)
        self.explanation, _, _ = self.api.call_api(prompt)
        keyword_db.insert_explanation(self.topic, self.explanation)
    
    def _parse_keywords(self, raw_keywords):
        try:
            keywords = ast.literal_eval(raw_keywords)
        except ValueError as e:
            logging.error(f"Error: {e}\nFor raw_keywords: {raw_keywords}")
            keywords = []
        return keywords
    
    def find_keywords(self):
        logging.info("Finding keywords")
        for _ in range(3):
            prompt = f"What are the 10 most important keywords of the text. Answer in a python array: \n{self.explanation}\n"
            raw_keywords, _, _ = self.api.call_api(prompt, model="gpt-3.5-turbo")
            keywords = self._parse_keywords(raw_keywords)
            if keywords:
                self.keywords = set(keywords)
                keyword_db.update_keywords(self.topic, self.keywords)
                break
            
    def setup_explanation(self, keyword, explanation=None):
        explanation_gpt = ExplanationGPT(keyword, explanation=explanation)
        self.keywords_explanations[keyword] = explanation_gpt
            
    def fill_explanation_from_db(self):
        logging.info("Filling explanation from db")
        results = keyword_db.select_multi(self.keywords)
        for res in results:
            self.setup_explanation(res[1], res[2])
            for keyword in self.keywords.copy():
                if keyword.lower() == res[1].lower():
                    self.keywords.remove(keyword)

    def add_explanation(self, result):
        keyword = result.metadata["keyword"]
        explanation = result.response['choices'][0]['message']['content']
        print(f"Result: {keyword}\n{explanation}")
        keyword_db.insert_explanation(keyword, explanation) # TODO Optimize db call
        self.setup_explanation(keyword, explanation)
    
    def explain_keywords(self):
        logging.info("Explaining keywords")
        api = OpenAIMultiClient(concurrency=50, endpoint="chats", data_template={"model": "gpt-3.5-turbo"})
        def make_requests():
            for index, keyword in enumerate(self.keywords):
                prompt = self.explanation_prompt(keyword, self.explanation)
                print(f"Request {index} {keyword}")
                api.request(
                    data={"messages": [{"role": "user", "content": prompt}]},
                    metadata={'id': index, 'keyword': keyword},
                )
        api.run_request_function(make_requests)

        for result in api:
            self.add_explanation(result)
        return self.keywords_explanations
    
    def add_keyword_explanation(self, keyword):
        logging.info("Adding keyword explanation")
        self.setup_explanation(keyword)
    
    def _format_explanation(self, explanation):
        max_length = 300
        extra_character =  "..."
        explanation = explanation.replace("\n", "<br>")
        if len(explanation) < max_length:
            max_length = len(explanation)
            extra_character = ""
        explanation = explanation[:max_length]
        explanation += extra_character
        return explanation

    def to_html(self, tooltip_class="tooltip", tooltip_text_class="tooltiptext"):
        logging.info("Generating html")
        html = self.explanation
        placeholder = lambda keyword: f"__{keyword.upper()}__"
        for keyword, _ in self.keywords_explanations.items(): # 2 Step to avoid replacing keywords inside of a tooltip
            html = html.replace(keyword, placeholder(keyword))
        for keyword, explanation_gpt in self.keywords_explanations.items():
            explanation = self._format_explanation(explanation_gpt.explanation)
            html = html.replace(placeholder(keyword), f'<span class="{tooltip_class}">{keyword}<span class="{tooltip_text_class}">{explanation}</span></span>')
        self.html = html


if __name__ == "__main__":
    with open("data/paper.txt", "r") as f:
        content = f.read()
    print(len(content))
    paper = ExplanationGPT("this paper", context=content)
    paper.explanation = "# Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering\n\nThe paper titled \"Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering\" introduces a benchmarking framework called LLM-KG-Bench for evaluating and monitoring the performance of Large Language Models (LLMs) in the field of knowledge graph engineering (KGE). The framework includes three challenges that focus on syntax and error correction, facts extraction, and dataset generation. The paper also discusses the limitations of LLMs in knowledge graph generation and highlights the need for prompt engineering and model performance tracking.\n\n## Keywords\nLarge Language Model, Knowledge Graph Engineering, Large Language Model Benchmark\n\n## Introduction\nThe introduction of the paper provides an overview of the potential of Large Language Models (LLMs) in changing the way we interact with data and technology. It mentions models like GPT-3 and GPT-4 that have shown proficient capabilities in solving textual assignments and have led to the emergence of prompt engineering. However, with the fast evolution and growing landscape of LLMs, it becomes challenging to keep track of their capabilities and choose the best model and prompt for a specific task. The paper highlights the under-explored area of LLM assessment in the context of knowledge graph engineering (KGE) and proposes the LLM-KG-Bench framework as a solution.\n\n## Related Work\nThe section on related work discusses the utilization of LLMs in the semantic web domain, where their capability to handle RDF-related syntaxes such as JSON-LD, Turtle, and SPARQL is mentioned. It refers to previous works that have explored the amalgamation of LLMs and knowledge graphs (KGs) as well as the relevance of this combination in the Knowledge Base Construction from Pre-trained Language Models (LM-KBC) Challenge. The study [5], which assesses ChatGPT's use in knowledge graph engineering, is also mentioned as providing insights into LLMs' potential and limitations.\n\n## The LLM-KG-Bench Framework\nThis section provides details about the LLM-KG-Bench framework, which is designed for benchmarking LLMs in the context of knowledge graph engineering. The framework focuses on automated evaluation procedures and supports configurable task sizing based on the relevance of the LLM's context size for KGE tasks. The framework is organized around benchmark tasks and LLM model connectors, with code for execution organization and result persistence. Benchmark tasks handle the evaluation of LLMs for specific tasks, and LLM model connectors encapsulate the connection to a specific LLM. The framework provides result visualization using seaborn and supports the addition of new benchmark tasks and LLM model connectors.\n\n## Initial Evaluation of the Framework with first Tasks\nTo test the LLM-KG-Bench framework, the paper presents an evaluation of three highly ranked LLMs using a set of benchmark tasks. The paper provides the details of the tasks performed and the specific evaluations conducted for each task. Task a involves fixing errors in Turtle files, Task b focuses on creating a knowledge graph from factsheet plaintext, and Task c involves synthetic dataset generation. The evaluation results are presented in the form of F1 scores and mean error values for different metrics associated with each task.\n\n## Conclusion and Future Work\nThe paper concludes by highlighting the need for measuring the knowledge graph engineering capabilities of LLMs and the benefits of the LLM-KG-Bench framework in addressing this need. It mentions the potential for extending the framework to enable dialogs between benchmark tasks and LLMs and the possibility of evaluating LLMs' capabilities to fix their answers with feedback. The paper also acknowledges the support received from grants and provides references to related online resources for further exploration.\n\n## References\nThe paper includes a list of references for further reading and exploration on the topic of large language models and knowledge graph engineering.\n\n## Online Resources\nThe paper provides links to the LLM-KG-Bench repository and the experimental data for further exploration and access to the framework and results."
    paper.keywords = set(['Developing', 'Scalable Benchmark', 'Assessing', 'Large Language Models', 'Knowledge Graph Engineering', 'LLM-KG-Bench', 'evaluating', 'performance', 'syntax', 'error correction', 'facts extraction', 'dataset generation', 'limitations', 'prompt engineering', 'model performance tracking', 'GPT-3', 'GPT-4', 'prompt engineering', 'knowledge graph generation', 'LLMs', 'assessing', 'framework', 'related work', 'semantic web domain', 'RDF-related syntaxes', 'JSON-LD', 'Turtle', 'SPARQL', 'amalgamation', 'knowledge graphs', 'Knowledge Base Construction from Pre-trained Language Models (LM-KBC) Challenge', 'ChatGPT', 'potential', 'limitations', 'LLM-KG-Bench framework', 'automated evaluation procedures', 'configurable task sizing', 'benchmark tasks', 'LLM model connectors', 'result visualization', 'seaborn', 'initial evaluation', 'highly ranked LLMs', 'benchmark tasks', 'F1 scores', 'mean error values', 'metrics', 'conclusion', 'future work', 'dialogs', 'feedback', 'references', 'online resources'])
    paper.fill_explanation_from_db()
    paper.to_html()
    print(paper)
