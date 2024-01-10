import os

from metaphor_python import Metaphor
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import json
import uuid

from .base_tool import BaseTool


class ResearchAgentTool(BaseTool):
    def __init__(self, metaphor_api_key):
        super().__init__(
            name="Research Agent",
            model="gpt-3.5-turbo-16k",
            temperature=1.0,
            uploads=None,
            inputs=[
                {
                    "input_label": "Research goal",
                    "example": "Study the impact of climate change on agriculture",
                    "button_label": "Research",
                    "help_label": "The AI Research Agent tool helps in conducting detailed research on a given topic, collecting facts and data, and producing concise results.",
                }
            ],
        )
        self.metaphor_api_key = metaphor_api_key

    # TODO: Change each step to a separate function
    def execute(self, chat, research_goal):
        metaphor = Metaphor(api_key=self.metaphor_api_key)
        search_response = self._search(chat, research_goal, metaphor)
        results = self._scrape_and_summarize(
            chat, research_goal, metaphor, search_response
        )
        output = self._combine_results(results)
        self._save_to_file(research_goal, output)
        return output

        # 1. Tool for search

    def _search(self, chat, research_goal, metaphor):
        search_response = metaphor.search(research_goal, use_autoprompt=True)
        results = [
            {
                "title": result.title,
                "url": result.url,
            }
            for result in search_response.results
        ]

        # 2. Tool for scraping and summarizing

    def _scrape_and_summarize(self, chat, research_goal, metaphor, results):
        all_content = []
        for result in results:
            url = result["url"]
            objective = research_goal
            find_similar_response = metaphor.find_similar(url)
            ids = [result.id for result in find_similar_response.results]
            contents_response = metaphor.get_contents(ids)
            content = "\n".join(
                [document.extract for document in contents_response.contents]
            )

            if len(content) > 10000:
                output = self._summary(objective, content, chat)
            else:
                output = content

            all_content.append(
                {"title": result["title"], "url": url, "content": output}
            )
            return all_content

        # Combine the results into a displayable format

    def _combine_results(self, all_content):
        result_display = ""
        for content in all_content:
            result_display += f"Title: {content['title']}\nURL: {content['url']}\nContent: {content['content']}\n\n"

        return result_display

    def _summary(self, objective, content):
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
        )
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the following text for {objective}:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"]
        )

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True,
        )

        output = summary_chain.run(input_documents=docs, objective=objective)

        return output

    def _save_to_file(self, research_topic, output):
        directory = "research_completions"
        os.makedirs(directory, exist_ok=True)

        date_str = datetime.today().strftime("%Y-%m-%d")
        filename = os.path.join(directory, f"{date_str}.json")

        unique_id = str(uuid.uuid4())  # Generate a unique ID

        data = {"id": unique_id, "research_topic": research_topic, "output": output}

        with open(filename, "a") as file:
            json.dump(data, file)
            file.write("\n")

        print(f"Saved data with ID {unique_id} to {filename}")
