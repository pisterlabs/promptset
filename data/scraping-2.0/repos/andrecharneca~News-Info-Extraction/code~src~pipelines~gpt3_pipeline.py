import openai
import os
from typing import Union
from .utils.prompts import ENTITY_REL_ENTITY_PROMPT
from .utils.relationships import RELATIONSHIP_LIST
from .utils.utils import GPT3Utils
from .datamodules.datasets import ParagraphDataset
import json
from pipelines.paths import TEMP_DATA_DIR



class CompanyGPT3Pipeline():
    """
    Paragraph (dict, list[dict], ParagraphDataset) -> Prompt -> Post processing/filtering -> {'relationships': list[dict]}
        - relationships (dict {"entity_1": str, "entity_2": str, "relationship": str, "passage": str})
    
    Expects ParagraphDataset with keys: "text", "title", "url"

    The prompt is formatted for the completion to be |Company|Relationship|Company|Date|Passage|, and end in |end|.
    """
    def __init__(self,
                 prompt : str = ENTITY_REL_ENTITY_PROMPT,
                 debug : bool = False,
                 debug_article_ids : list[int] = None,
                 **gpt_kwargs):
        
        self.prompt = prompt
        self.gpt_kwargs = gpt_kwargs
        self.debug = debug
        self.input = {"article_id":-1, "paragrah_id":-1} # used for postprocessing
        openai.api_key = os.environ["OPENAI_API_KEY"]


    def _get_response(self, article_id, paragraph_id):
        response_path = os.path.join(TEMP_DATA_DIR, f"gpt3_responses/article_id_{article_id}_paragraph_id_{paragraph_id}.json")
        with open(response_path, "r") as f:
            response = json.load(f)
        return response

    def preprocess(self, input):
        self.input = input
        model_input = self.prompt.format(text=input["text"], title=input["title"], date="")
        return model_input
    
    def forward(self, model_input):
        if self.debug:
            model_output = self._get_response(self.input["article_id"], self.input["paragraph_id"])
        else:
            model_output = openai.Completion.create(prompt = model_input, **self.gpt_kwargs)
            # save response to json
            file_name = os.path.join(TEMP_DATA_DIR, f"gpt3_responses/article_id_{self.input['article_id']}_paragraph_id_{self.input['paragraph_id']}.json")
            json.dump(model_output, open(file_name, "w"))
        return model_output
    
    def postprocess(self, model_output):
        completion = model_output["choices"][0]["text"]
        relationships = GPT3Utils.parse_table(completion)
        entities = GPT3Utils.parse_entity_list(completion)

        relationships = GPT3Utils.remove_hallucinated_relationships(relationships, RELATIONSHIP_LIST)
        input_text = f"{self.input['title']} {self.input['text']}"
        relationships = GPT3Utils.filter_relationships_by_text(relationships, input_text)
        entities = GPT3Utils.filter_entities_by_text(entities, input_text)

        return {"relationships": relationships, "entities": entities}

    def __call__(self, paragraphs : Union[dict, list[dict], ParagraphDataset], *args, **kwargs):
        """
        Runs the pipeline sequentially on the input paragraphs.
        Works paragraph by paragraph.
        """
        if isinstance(paragraphs, ParagraphDataset):
            return self(paragraphs.data, *args, **kwargs)
        
        elif isinstance(paragraphs, list):
            return [self(paragraph, *args, **kwargs) for paragraph in paragraphs] 
        
        elif isinstance(paragraphs, dict):
            prep_paragraph = self.preprocess(paragraphs)
            model_output = self.forward(prep_paragraph)
            post_output = self.postprocess(model_output)
            return post_output