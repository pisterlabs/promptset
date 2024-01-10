from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
from datetime import date

class BlogGenerator:
    """
    A class for generating blog content using ChatOpenAI and summarization chains.

    Args:
        syntax_file_path (str): The file path to the syntax template.
    """
    def __init__(self, syntax_file_path="syntax.json"):
        self.syntax_file_path = syntax_file_path
        self.load_syntax()

    def load_syntax(self):
        """
        Load the syntax template from the specified file.
        """
        try:
            with open(self.syntax_file_path, "r") as f:
                self.syntax = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Syntax file not found at path: {self.syntax_file_path}")

    def generate_blog_from_url(self, url, generate_prompt_template=None, summarize_prompt_template=None):
        """
        Generate a blog from a given URL.

        Args:
            url (str): The URL of the article.
            generate_prompt_template (PromptTemplate, optional): The template for generating content.
            summarize_prompt_template (PromptTemplate, optional): The template for summarizing content.

        Returns:
            dict: The generated blog content in JSON format.
        """
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

            # Use default prompts if none are provided
            if generate_prompt_template is None:
                generate_prompt_template = PromptTemplate(
                    template="""- you are a SEO specialist and you need to write a blog post on the following notes: {topic}
                        - you can add details from your own knowledge
                        - use your knowledge of SEO to write a blog post that will rank high on google
                        - blog should follow the following syntax: {syntax}
                        - strictly follow the format of the blog post, you can also add any markdown syntax you want to make the blog post look better
                        - content under each heading should also be very detailed and long and SEO optimized
                        - todays date is {date}""",
                    input_variables=["topic", "syntax", "date"]
                )

            if summarize_prompt_template is None:
                summarize_prompt_template = PromptTemplate(
                    template="""you are a SEO specialist and you need to summarize the article don't miss any important details or dates, summary should be very detailed
                                dont miss any quotables and names of people, places, companies, etc.""",
                    input_variables=[]
                )

            summary_chain = load_summarize_chain(llm, chain_type="stuff")

            chain = (
                summary_chain
                | (lambda dictionary: {"topic": dictionary["output_text"], "syntax": self.syntax, "date": date.today().strftime("%B %d, %Y")})
                | generate_prompt_template
                | llm
                | StrOutputParser()
            )

            blog_text = chain.invoke({"input_documents": docs, "query": summarize_prompt_template})

            return json.loads(blog_text)

        except Exception as e:
            raise RuntimeError(f"Error generating blog from URL: {str(e)}")

    def generate_blog_from_topic(self, topic, generate_prompt_template=None):
        """
        Generate a blog from a given topic.

        Args:
            topic (str): The topic of the blog post.
            generate_prompt_template (PromptTemplate, optional): The template for generating content.

        Returns:
            dict: The generated blog content in JSON format.
        """
        try:
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

            # Use default prompt if none is provided
            if generate_prompt_template is None:
                generate_prompt_template = PromptTemplate(
                    template="""- you are a SEO specialist and you need to write a blog post on the following topic: {topic}
                        - use your knowledge include data from your own knowledge
                        - use your knowledge of SEO to write a blog post that will rank high on google
                        - blog should follow the following syntax: {syntax}
                        - strictly follow the format of the blog post, you can also add any markdown syntax you want to make the blog post look better
                        - content under each heading should also be very detailed and long and SEO optimized
                        - todays date is {date}""",
                    input_variables=["topic", "syntax", "date"]
                )

            chain = (
                generate_prompt_template
                | llm
                | StrOutputParser()
            )

            blog_text = chain.invoke({"topic": topic, "syntax": self.syntax, "date": date.today().strftime("%B %d, %Y")})

            return json.loads(blog_text)

        except Exception as e:
            raise RuntimeError(f"Error generating blog from topic: {str(e)}")

if __name__ == "__main__":
    syntax_file_path = "syntax.json"

    blog_generator = BlogGenerator(syntax_file_path)

    # Generate blog from topic with default prompts
    topic = "The Impact of Artificial Intelligence on Modern Business"
    try:
        blog_json_topic = blog_generator.generate_blog_from_topic(topic)
        print(blog_json_topic)
    except Exception as e:
        print(f"Error generating blog from topic: {str(e)}")

    # Generate blog from URL with default prompts
    url = "https://www.indiatoday.in/education-today/gk-&-current-affairs/story/indian-origin-ceos-leading-top-companies-across-the-world-1893342-2021-12-28"
    try:
        blog_json_url = blog_generator.generate_blog_from_url(url)
        print(blog_json_url)
    except Exception as e:
        print(f"Error generating blog from URL: {str(e)}")

    # Save blogs to file
    try:
        with open("blog_url.json", "w") as f:
            json.dump(blog_json_url, f)

        with open("blog_topic.json", "w") as f:
            json.dump(blog_json_topic, f)
    except Exception as e:
        print(f"Error saving blogs to file: {str(e)}")
