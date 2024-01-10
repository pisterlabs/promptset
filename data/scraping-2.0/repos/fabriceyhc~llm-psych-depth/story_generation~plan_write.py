from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from typing import List, Dict
from utils import *
from loader import *


premise =\
"""
Premise: {prompt}
"""
    
character_prompt =\
"""
Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.
Only respond with the characters' names and descriptions.
"""

plan_info =\
"""
Premise: {prompt}

Character Portraits:
{characters}
"""

story_prompt =\
"""
Task: Write a 500-word story based on the premise and character portraits. The story should be emotionally deep and impactful.
Only respond with the story.
"""

class CharactersOutput(BaseModel):
        character_list:   List[str] = Field(description="character list")


class PlanWritePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    class OutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text

    def generate_character_prompts(self, prompts):

        parser = PydanticOutputParser(pydantic_object=CharactersOutput)

        prompts_to_run = []

        for prompt_id, prompt in enumerate(prompts):

            system_profile_prompt = SystemMessagePromptTemplate.from_template(premise)
            human_message_prompt = HumanMessagePromptTemplate.from_template(character_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                partial_variables={"format_instructions": parser.get_format_instructions()},
                template_format='jinja2',
                output_parser=parser,
            )
            _input = chat_prompt.format_messages(prompt=prompt)

            prompts_to_run.append({
                "id": prompt_id,
                "premise": prompt,
                "characters_prompt": extract_string_prompt(_input)
            })

        return prompts_to_run


    def generate_story_prompts(self, planwrite_df):

        prompts_to_run = []

        for prompt_id, prompt in planwrite_df.iterrows():

            system_profile_prompt = SystemMessagePromptTemplate.from_template(plan_info)
            human_message_prompt = HumanMessagePromptTemplate.from_template(story_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                output_parser=self.OutputParser(),
            )
            _input = chat_prompt.format_messages(prompt=prompt['premise'],
                                                characters=create_numbered_string(prompt['character_list']))

            new_prompt = prompt.to_dict()
            new_prompt.update({
                "story_prompt": extract_string_prompt(_input)
            })
            prompts_to_run.append(new_prompt)

        return prompts_to_run


    def prompt_llm(self, prompts, save_dir, model_name, regen_ids=None, template_type='plan_write'):

        save_path = os.path.join(save_dir, model_name, template_type)
        os.makedirs(save_path, exist_ok=True)

        character_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", premise),
            ("human", character_prompt),
        ])
        story_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", plan_info),
            ("human", story_prompt),
        ])

        indexed_prompts = [(id, prompt) for id, prompt in enumerate(prompts)]

        if not regen_ids:
            indexed_prompts = [(i, prompt) for i, prompt in indexed_prompts if i in regen_ids]

        for id, prompt in indexed_prompts:

            character_chain = character_chat_prompt | self.llm | self.OutputParser()
            character_output = character_chain.invoke({"prompt": prompt})

            story_chain =  story_chat_prompt | self.llm | self.OutputParser()

            max_tries = 3
            min_words = 100
            num_words, tries = 0, 0
            while num_words < min_words and tries < max_tries:
                output = story_chain.invoke({"prompt": prompt, "characters": character_output})
                num_words = len(output.split())
                if num_words < min_words:
                    tries += 1
                    print(f"Generated fewer than {min_words} words. Trying {max_tries-tries} more times")

            print(id)
            print("-" * 20)
            print(output)
            print("=" * 50)

            save_info = {
                "id": id,
                "model_name": model_name,
                "story_prompt": prompt,
                "characters": character_output,
                "output": output
            }

            filename = f"{save_info['id']}_{first_n_words(save_info['story_prompt'])}_{generate_random_id()}.json"
            with open(os.path.join(save_path, filename), 'w') as f:
                json.dump(save_info, f, indent=4)