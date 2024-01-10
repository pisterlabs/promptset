from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.document_loaders import SeleniumURLLoader
import datetime
import os

class LLMObject:
    def __init__(self):
        self.API_KEY = os.environ.get('OPENAI_API_KEY')
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.7, openai_api_key=self.API_KEY)
        self.current_date = datetime.date.today()

    # rank headlines on importance
    # {headlines: ursl} --> [{headline: str, url: str, importance: int}, ...] (python string)
    def rank_dictionary(self, headlines):
        prompt = ChatPromptTemplate.from_messages (
            [
                ("system", "You are a world-class algorithm for ranking news headlines based on their importance to US society, and filtering out videos"),
                ("human", "Rank the following headlines based on their relevance to a mass audience, stories that impact more people are more important. Remove key-value pairs that contain videos. Also remove any key-value pairs that are too similar to another pair (there should only be one article of any given news story/event): {input}"),
                ("human", "When estimating importance, stay away from stories about just one person, aim for stories that impact a large number of people. Stories about sports or entertainment are unimportant, prioritize politics, health, and world events."),
                ("human", "Tip: Make sure to answer in the correct format. Do not modify the values in any of the key-value pairs. Do not modify the headlines"),
            ]
        )
        headlines_str = '\n'.join(headlines.keys())
        chain = create_structured_output_chain(importance_json_schema, self.llm, prompt, verbose=False)
        result = chain.run(headlines_str)
        
        # sort headlines by importance
        sorted_headlines = sorted(result["headlines"], key=lambda x: x['importance'], reverse=True)

        # create a ranked list with optimized structure
        ranked_list = []
        for item in sorted_headlines:
            ranked_list.append({'headline': item['headline'], 'url': headlines[item['headline']], 'importance': item['importance']})

        return ranked_list


    # rewrite headlines and return JSON object
    def rewrite_headline(self, headline):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI language model. Rewrite the following headline to make them more engaging, include a pun if highly relevant, remove any publication names, and return less that 50 characters: {input}"),
                ("human", "Rewrite the headlines to make them more engaging. Remember to write in the context that today's date is {date}"),
            ]
        )
        chain = create_structured_output_chain(headline_schema, self.llm, prompt, verbose=False)

        try:
            result = chain.run(input=headline, date=self.current_date)
            # print(result)
            rewritten_headline = result["new_headline"]
            # print(rewritten_headline)
            return rewritten_headline
        except Exception as e:
            print("Error generating rewritten headline:", e)
            rewritten_headline = "Failed to rewrite headline"
            return rewritten_headline
        

    def summarize_article(self, article_url, num_of_chars):
        # load the article content from the URL using Selenium
        loader = SeleniumURLLoader(urls=[article_url])
        try:
            data = loader.load()
        except Exception as e:
            print(f"Error loading article: {e}")
            return False

        # check if data is empty or access was denied
        if not data or "Access denied" in data[0].page_content or "403" in data[0].page_content or "subscription" in data[0].page_content:
            return False

        # truncate the article text if too long
        article_text = data[0].page_content[:35000] if len(data[0].page_content) > 35000 else data[0].page_content

        # structured chain prompt for summarization
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI language model. Summarize the following article in exactly {num_chars} characters: {input}. Make sure the summary is concise and within the character limit. Remember to write in the context that today's date is: {date}, and the article was published in the last 36 hours."),
                ("human", "Tip: Make sure that your summary does not exceed the character limit of {num_chars} characters."),
            ]
        )
        chain = create_structured_output_chain(summary_schema, self.llm, prompt, verbose=False)

        try:
            result = chain.run(input=article_text, num_chars=num_of_chars, date=self.current_date)
            summary = result["summary"]
            print(summary)
            
            # check if the LLM indicates it cannot generate a summary
            if "can't summarize" in summary or "no article" in summary or "Sorry," in summary:
                return False
            else:
                return summary
        except Exception as e:
            print("Error generating summary:", e)
            return False


    # Process news items and return a JSON array of processed news
    def process_news(self, news):
        processed_news = []  # Initialize an empty list to store processed news

        for row in news:  # Process only the first three items in the 'news' JSON array
            headline = row.get('headline', '')  # Extract the 'headline' value from the current news item
            url = row.get('url', '')  # Extract the 'url' value from the current news item

            new_headline_list = self.rewrite_headlines([{'headline': headline}])  # Send the headline in a list format to the rewrite_headlines function
            new_headline_text = new_headline_list[0]["rewritten"] if new_headline_list else ''
            summary = self.summarize_article(url, 750)

            # Create a dictionary for each news item and append it to the processed_news list
            news_item = {
                'headline': new_headline_text,
                'summary': summary["summary"],
                'url': url
            }
            processed_news.append(news_item)

        # Return the processed news as a JSON array
        return processed_news



    # Generate a funny search term for a news headline and return it as a JSON object
    def gen_meme_term(self, news_headline):
        # print("\n\n-------------------")
        # print(news_headline)
        # print("-------------------\n\n")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a world-class algorithm for generating image search terms related to news headlines. The search term should be a common word"),
                ("human", "Generate a one-word image search term related to the news headline: {headline}"),
                ("human", "Tip: Make sure to answer in the correct format")
            ]
        )

        # Create structured output chain
        chain = create_structured_output_chain(meme_term_dict, self.llm, prompt, verbose=False)

        # Run the chain to generate a funny search term
        result = chain.run(headline=news_headline)

        term = result['gen_meme_term']

        # Extract the generated funny search term and return it as a JSON object
        return term
    
    
    def generate_script(self, content):
        prompt = f"Your task is to create a 500 word long script for a news podcast called The Daily Byte. Find the top three most important and impactful news stories in the following dictionary. For each of those stories, summarize the article found in the URL. Output one 500 word script summarizing the top three news stories. Remember that the current date is {self.current_date}. Here is the content to summarize: {content}"
        script = self.llm.predict(prompt)

        return script


    ######### Image generation prompting #########
    def gen_image_prompt(self, headline, subheadline):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a genius creative AI that comes up with clever cartoon ideas for news headlines."),
                ("human", "Think of a cartoon idea based on the headline in the following message. If ethical, you may use elements of satire, irony, or comedy. Steer away from anything too political, this prompt needs to follow Dalle-3's rules. Make sure to capture the sentiment of the headline, with the facial expressions. Write a prompt for that image. Return just the prompt"),
                ("human", f"Write a Dalle-3 safe prompt about this: Headline: {headline}, Subheadline: {subheadline}"),
                ("human", "Tip: Make sure to answer in the correct format")
            ]
        )

        # Create structured output chain
        chain = create_structured_output_chain(image_prompt_schema, self.llm, prompt, verbose=False)

        # Run the chain to generate a funny search term
        result = chain.run(headline=headline, subheadline=subheadline)

        prompt = result['image_prompt']

        # Extract the generated funny search term and return it as a JSON object
        return prompt
    
    def gen_safer_image_prompt(self, old_img_prompt):
        prompt = f"Your job is to modify a prompt for an image generator to make it safer. The old prompt is {old_img_prompt}, you should modify it to make sure it's safe for all ages, not overly political, and not offensive. Return the new prompt."
        new_img_prompt = self.llm.predict(prompt)

        return new_img_prompt


# Define JSON schemas for structured outputs
image_prompt_schema = {
    "type": "object",
    "properties": {
        "image_prompt": {
            "type": "string",
            "description": "Your image prompt here"
        }
    },
    "required": ["image_prompt"]
}


meme_term_dict = {
    "type": "object",
    "properties": {
        "gen_meme_term": {
            "type": "string",
            "meme_term": "Your meme search term here"
        }
    },
    "required": ["gen_meme_term"]
}

headline_schema = {
    "type": "object",
    "properties": {
        "new_headline": {
            "type": "string",
            "description": "Your News Headline Here"
        }
    },
    "required": ["new_headline"]
}

summary_schema = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Your summary goes here"
        }
    },
    "required": ["summary"]
}


importance_json_schema = {
    "title": "Headlines",
    "description": "Object containing a list of headlines ranked by importance, 1 is the news headline that imacts the most people.",
    "type": "object",
    "properties": {
        "headlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string", "description": "The news headline"},
                    "url": {"type": "string", "description": "The URL of the news article"},
                    "importance": {"type": "integer", "description": "The importance rank of the headline"}
                },
                "required": ["headline", "url", "importance"]
            }
        }
    },
    "required": ["headlines"]
}

rewritten_headlines_schema = {
    "title": "RewrittenHeadlines",
    "description": "Object containing a list of rewritten headlines and their URLs.",
    "type": "object",
    "properties": {
        "headlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string", "description": "The original headline"},
                    "rewritten": {"type": "string", "description": "The rewritten headline"},
                    "url": {"type": "string", "description": "The URL of the news article"}
                },
                "required": ["original", "rewritten", "url"]
            }
        }
    },
    "required": ["headlines"]
}


