from langchain.prompts import StringPromptTemplate,FewShotPromptTemplate
from pydantic import BaseModel, validator

PROMPT = """
In addition to the scoring rubrics in the examples above,give feedback and score the {component_name} using the attempted answer's(correct/incorrect) reasoning-based rubrics and their definitions below.
Please include both the previous scoring rubrics and the following reasoning-based rubrics before giving the feedback for a particular aspect and add up the scores for all the aspects for the total scores of the {component_name}. 
Many of these feedback points for the {component_name} depend upon the reasoning and the attempted answer correctness so consider that while giving feedback for the {component_name}.
{component_name} reasoning-based rubrics: {reasoning_rubrics}
Format instructions : {format_instructions}
Clinical note: {clinical_note}
Keypoint: {keypoint}
Topic: {topic}
Context: {context}
Question: {question}
Correct answer: {correct_answer}
Distractor options: {distractor_options}
"""


class ComponentFeedbackPromptTemplate(FewShotPromptTemplate):
    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""
    examples : dict 
    def format(self, **kwargs) -> str:
        prompt = PROMPT.format(
            component_name=kwargs["component_name"],
            clinical_note = kwargs["clinical_note"],
            keypoint = kwargs["keypoint"],
            topic=kwargs["topic"],
            context=kwargs["context"],
            question=kwargs["question"],
            correct_answer=kwargs["correct_answer"],
            distractor_options=kwargs["distractor_options"],
            attempted_answer=kwargs["attempted_answer"],
            reasoning=kwargs["reasoning"],
            reasoning_rubrics = kwargs["reasoning_rubrics"]
        )
        return prompt

    def _prompt_type(self):
        return "component-feedback"