import json
from cat.log import log
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from kor import create_extraction_chain, from_pydantic, Object, Text
from enum import Enum

class CFormState(Enum):
    ASK_INFORMATIONS    = 0
    ASK_SUMMARY         = 1
    EXECUTE_ACTION      = 2

class ConversationalForm:

    def __init__(self, model, cat):
        self.model = model
        self.cat = cat
        self.state = CFormState.ASK_INFORMATIONS


    # Check if the form is completed
    def is_completed(self):
        for k,v in self.model.model_dump().items():
            if v in [None, ""]:
                return False
        return True


    # Return list of empty form's fields
    def _check_what_is_empty(self):
        ask_for = []
        for field, value in self.model.model_dump().items():
            if value in [None, "", 0]:
                ask_for.append(f'{field}')
        return ask_for


    # Queries the llm asking for the missing fields of the form, without memory chain
    def ask_missing_information(self) -> str:
       
        # Gets the information it should ask the user based on the fields that are still empty
        ask_for = self._check_what_is_empty()

        prefix = self.cat.mad_hatter.execute_hook("agent_prompt_prefix", '', cat=self.cat)
        user_message = self.cat.working_memory["user_message_json"]["text"]
        chat_history = self.cat.agent_manager.agent_prompt_chat_history(
            self.cat.working_memory["history"]
        )
        
        # Prompt
        prompt = f"""{prefix}
        Create a question for the user, 
        below are some things to ask the user in a conversational and confidential way, to complete the pizza order.
        You should only ask one question at a time even if you don't get all the information
        don't ask how to list! Don't say hello to the user! Don't say hi.
        Explain that you need some information. If the ask_for list is empty, thank them and ask how you can help them.
        Don't present the conversation history but just a question.
        ### ask_for list: {ask_for}
        ## Conversation until now:{chat_history}
        - Human: {user_message}
        - AI: """

        log.warning(f'MISSING INFORMATIONS: {ask_for}')
        response = self.cat.llm(prompt)

        return response 

    # Show summary of the form to the user
    def show_summary(self, cat):
        prefix = self.cat.mad_hatter.execute_hook("agent_prompt_prefix", '', cat=self.cat)
        user_message = self.cat.working_memory["user_message_json"]["text"]
        chat_history = self.cat.agent_manager.agent_prompt_chat_history(
            self.cat.working_memory["history"]
        )
        
        # Prompt
        prompt = f"""show the summary of the data in the completed form and ask the user if they are correct. 
        Don't ask irrelevant questions. 
        Try to be precise and detailed in describing the form and what you need to know.
        ### form data: {self.model}
        ## Conversation until now:{chat_history}
        - Human: {user_message}
        - AI: """

        # Change status
        self.state = CFormState.ASK_SUMMARY

        # Queries the LLM
        response = self.cat.llm(prompt)
        log.debug(f'show_summary: {response}')
        return response


    # Check user confirm the form data
    def check_confirm(self) -> bool:
        user_message = self.cat.working_memory["user_message_json"]["text"]
        
        # Prompt
        prompt = f"""
        respond with either YES if the user's message is affirmative 
        or NO if the user's message is not affirmative
        - Human: {user_message}
        - AI: """

        # Queries the LLM and check if user is agree or not
        response = self.cat.llm(prompt)
        log.debug(f'check_confirm: {response}')
        confirm = "YES" in response
        
        # If confirmed change status
        if confirm:
            self.state = CFormState.EXECUTE_ACTION

        return confirm


    # Updates the form with the information extracted from the user's response
    # (Return True if the model is updated)
    def update_from_user_response(self):

        # Extract new info
        user_response_json = self._extract_info_from_scratch()
        # user_response_json = self._extract_info_by_pydantic()
        # user_response_json = self._extract_info_by_kor()
        if user_response_json is None:
            return False

        # Gets a new_model with the new fields filled in
        non_empty_details = {k: v for k, v in user_response_json.items() if v not in [None, ""]}
        new_model = self.model.copy(update=non_empty_details)

        # Check if there is no information in the new_model that can update the form
        if new_model.model_dump() == self.model.model_dump():
            return False

        # Validate new_model (raises ValidationError exception on error)
        self.model.model_validate(new_model.model_dump())

        # Overrides the current model with the new_model
        self.model = self.model.model_construct(**new_model.model_dump())
        #print(f'updated model:\n{self.model.model_dump_json(indent=4)}')
        log.critical(f'MODEL : {self.model.model_dump_json()}')
        return True


    #### PYDANTIC & KOR IMPLEMENTATIONS ####

    # Extracted new informations from the user's response (by pydantic)
    def _extract_info_by_pydantic(self):
        parser = PydanticOutputParser(pydantic_object=type(self.model))
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        log.debug(f'get_format_instructions: {parser.get_format_instructions()}')
        
        user_message = self.cat.working_memory["user_message_json"]["text"]
        _input = prompt.format_prompt(query=user_message)
        output = self.cat.llm(_input.to_string())
        log.debug(f"output: {output}")

        #user_response_json = parser.parse(output).dict()
        user_response_json = json.loads(output)
        log.debug(f'user response json: {user_response_json}')
        return user_response_json


    # Extracted new informations from the user's response (by kor)
    def _extract_info_by_kor(self):
        user_message = self.cat.working_memory["user_message_json"]["text"]
        
        schema, validator = from_pydantic(type(self.model))   
        chain = create_extraction_chain(self.cat._llm, schema, encoder_or_encoder_class="json", validator=validator)
        log.debug(f"prompt: {chain.prompt.to_string(user_message)}")
        
        output = chain.run(user_message)["validated_data"]
        try:
            user_response_json = output.dict()
            log.debug(f'user response json: {user_response_json}')
            return user_response_json
        except Exception  as e:
            log.debug(f"An error occurred: {e}")
            return None


    #### FROM SCRATCH IMPLEMENTATION ####

    # Extracted new informations from the user's response (from sratch)
    def _extract_info_from_scratch(self):
        user_message = self.cat.working_memory["user_message_json"]["text"]
        prompt = self._get_pydantic_prompt(user_message)
        log.debug(f"prompt: {prompt}")
        json_str = self.cat.llm(prompt)
        user_response_json = json.loads(json_str)
        log.debug(f'user response json:\n{user_response_json}')
        return user_response_json

    # return pydantic prompt based from examples
    def _get_pydantic_prompt(self, message):
        prompt_examples = type(self.model).get_prompt_examples()
        lines = []
        for example in prompt_examples:
            lines.append(f"Sentence: {example['sentence']}")
            lines.append(f"JSON: {self._format_prompt_json(example['json'])}")
            lines.append(f"Updated JSON: {self._format_prompt_json(example['updatedJson'])}")
            lines.append("\n")
        result = "Update the following JSON with information extracted from the Sentence:\n\n"
        result += "\n".join(lines)
        result += f"Sentence: {message}\nJSON:{json.dumps(self.model.dict(), indent=4)}\nUpdated JSON:"
        return result

    #format json for prompt
    def _format_prompt_json(self, values):
        attributes = list(self.model.__annotations__.keys())
        data_dict = dict(zip(attributes, values))
        return json.dumps(data_dict, indent=4)
