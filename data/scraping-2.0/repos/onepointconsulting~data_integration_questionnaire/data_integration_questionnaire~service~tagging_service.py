from langchain.chains import create_tagging_chain

from data_integration_questionnaire.config import cfg
from data_integration_questionnaire.log_init import logger

schema = {
    "properties": {
        "sentiment": {"type": "string"},
        "affirmative": {
            "type": "string",
            "enum": ["yes", "no", "undecided", "leans towards yes", "leans towards no"],
        },
        "contains_question": {
            "type": "boolean"
        },
        "confidence_degree": {
            "type": "float"
        }
    },
    "required": ["affirmative", "sentiment", "contains_question", "confidence_degree"],
}

chain = create_tagging_chain(schema, cfg.llm)

def tag_response(response: str) -> dict:
    res = chain(response)
    return res

if __name__ == "__main__":
    def process_answer(answer: str):
        logger.info(type(answer))
        logger.info(answer)
    # Does your organization support an event driven architecture for data integration?
    process_answer(tag_response("Yes, it does"))
    process_answer(tag_response("Well, since you are asking, I am not quite sure about it."))
    # Does your organization take more than 3 weeks for data integration between 2 systems?
    process_answer(tag_response("Well, that depends on the size of the project. But in most cases yes."))
    process_answer(tag_response("Almost we never finish integrations before that."))