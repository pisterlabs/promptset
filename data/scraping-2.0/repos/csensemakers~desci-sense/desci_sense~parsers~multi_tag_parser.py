import os

from pathlib import Path
from typing import Optional, Dict
import desci_sense.configs as configs

from confection import Config

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema import (
    HumanMessage,
)

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..dataloaders.twitter.twitter_utils import scrape_tweet, extract_external_ref_urls
from ..dataloaders.mastodon.mastodon_utils import scrape_mastodon_post, extract_external_masto_ref_urls
from ..utils import extract_and_expand_urls, identify_social_media
from ..schema.post import RefPost
from ..prompting.post_tags_pydantic import PostTagsDataModel
from ..postprocessing.parser_utils import fix_json_string_with_backslashes


def format_answer(post_tags: PostTagsDataModel) -> dict:
    tags = list(post_tags.get_selected_tags())
    return {"final_answer": post_tags.get_selected_tags_str(),
                             "reasoning": "",
                             "single_tag": tags[:1],
                             "multi_tag": tags}

class MultiTagParser:
    def __init__(self, 
                 config: Config,
                 api_key: Optional[str]=None,
                 openapi_referer: Optional[str]=None
                 ) -> None:
        
        self.config = config
        
        # if no api key passed as arg, default to environment config
        openai_api_key = api_key if api_key else os.environ["OPENROUTER_API_KEY"]
        
        openapi_referer = openapi_referer if openapi_referer else os.environ["OPENROUTER_REFERRER"]


        # init model
        model_name = "mistralai/mistral-7b-instruct" if not "name" in config["model"] else config["model"]["name"]
        
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=self.config["model"]["temperature"],
            openai_api_key=openai_api_key,
            openai_api_base=configs.OPENROUTER_API_BASE,
            headers={"HTTP-Referer": openapi_referer}, 
        )



        # Instantiate the pydantic parser with the selected model.
        self.pydantic_parser = PydanticOutputParser(pydantic_object=PostTagsDataModel)

        self.prompt_template = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "Tag the post as accurately as possible.\n{format_instructions}\n{question}"
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": self.pydantic_parser.get_format_instructions(),
        },
        )

    def run(self, text: str) -> PostTagsDataModel:
        """Process post text into PostTagsDataModel which represents the tags selected based on the target text.

        Args:
            text (str): input text representing post

        Returns:
            PostTagsDataModel: Parsing result
        """

        # Generate the input using the updated prompt.
        user_query = (
            f"Target post: {text}"
        )
        _input = self.prompt_template.format_prompt(question=user_query)

        output = self.model(_input.to_messages())
        fixed_content = fix_json_string_with_backslashes(output.content)
        parsed: PostTagsDataModel = self.pydantic_parser.parse(fixed_content)

        return parsed
    
    def run_raw(self, text: str) -> dict:
        """Process post text without Pydantic post-processing

        Args:
            text (str): input text representing post

        Returns:
            Dict of raw parsing result
        """

        # Generate the input using the updated prompt.
        user_query = (
            f"Target post: {text}"
        )
        _input = self.prompt_template.format_prompt(question=user_query)

        output = self.model(_input.to_messages())
        fixed_content = fix_json_string_with_backslashes(output.content)

        return fixed_content

    def process_text(self, text: str) -> PostTagsDataModel:
        """

        """
        post_tags = self.run(text)


        result = {"text": text,
                  "answer": format_answer(post_tags)
                  }
        
        return result
    

    def process_tweet_url(self, tweet_url: str):

        # get tweet in RefPost format
        post: RefPost = scrape_tweet(tweet_url)

        result = self.process_ref_post(post)

        return result
    

    def process_toot_url(self, toot_url: str):
        """Scrape target toot and run parser on it.

        Args:
            toot_url (str): url of Mastodon post
        """

        # get toot in json format
        post: RefPost = scrape_mastodon_post(toot_url)

        result = self.process_ref_post(post)

        return result
    
    def process_ref_post(self, post: RefPost):
        """Run parser on target RefPost.

        Args:
            post - RefPost representation of mastodon post
        """

        # check if there is an external link in this post - if not, tag as <no-ref>
        if not post.has_refs():
            answer = {"reasoning": "[System msg: no urls detected - categorizing as <no-ref>]", 
                             "final_answer": "<no-ref>"}
            result = {"post": post,
                    "answer": answer
                    }
        else:
            # url detected, process to find relation of text to referenced url
            post_tags = self.run(post.content)

            result = {"post": post,
                    "answer": format_answer(post_tags)
                    }

        return result

    def process_url(self, post_url: str):
        """
        Scrape social media post and parse using model. 
        Supported types: Mastodon & Twitter
        """
        
        # check social media type
        social_type = identify_social_media(post_url)

        if social_type == "twitter":
            result = self.process_tweet_url(post_url)
        
        elif social_type == "mastodon":
            result = self.process_toot_url(post_url)

        else:
            raise IOError(f"Could not detect social media type of input URL: {post_url}")

        return result

        
