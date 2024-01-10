from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import openai
import os
import rclpy

from angel_system_nodes.base_intent_detector import BaseIntentDetector, INTENT_LABELS

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# The following are few shot examples when prompting GPT.
FEW_SHOT_EXAMPLES = [
    {"utterance": "Go back to the previous step!", "label": "prev_step."},
    {"utterance": "Next step, please.", "label": "next_step"},
    {"utterance": "How should I wrap this tourniquet?", "label": "inquiry"},
    {"utterance": "The sky is blue", "label": "other"},
]


class GptIntentDetector(BaseIntentDetector):
    def __init__(self):
        super().__init__()
        self.log = self.get_logger()

        # This node additionally includes fields for interacting with OpenAI
        # via LangChain.
        if not os.getenv("OPENAI_API_KEY"):
            self.log.info("OPENAI_API_KEY environment variable is unset!")
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not os.getenv("OPENAI_ORG_ID"):
            self.log.info("OPENAI_ORG_ID environment variable is unset!")
        else:
            self.openai_org_id = os.getenv("OPENAI_ORG_ID")
        if not bool(self.openai_api_key and self.openai_org_id):
            raise ValueError("Please configure OpenAI API Keys.")
        self.chain = self._configure_langchain()

    def _configure_langchain(self):
        def _labels_list_parenthetical_str(labels):
            concat_labels = ", ".join(labels)
            return f"({concat_labels})"

        def _labels_list_str(labels):
            return ", ".join(labels[:-1]) + f" or {labels[-1]}"

        all_intents_parenthetical = _labels_list_parenthetical_str(INTENT_LABELS)
        all_intents = _labels_list_str(INTENT_LABELS)

        # Define the few shot template.
        template = (
            f"Utterance: {{utterance}}\nIntent {all_intents_parenthetical}: {{label}}"
        )
        example_prompt = PromptTemplate(
            input_variables=["utterance", "label"], template=template
        )
        prompt_instructions = f"Classify each utterance as {all_intents}.\n"
        inference_sample = (
            f"Utterance: {{utterance}}\nIntent {all_intents_parenthetical}:"
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=FEW_SHOT_EXAMPLES,
            example_prompt=example_prompt,
            prefix=prompt_instructions,
            suffix=inference_sample,
            input_variables=["utterance"],
            example_separator="\n",
        )

        # Please refer to https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py
        openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            # Only 2 tokens needed for classification (tokens are delimited by use of '_', i.e.
            # 'next_step' counts as 2 tokens).
            max_tokens=2,
        )
        return LLMChain(llm=openai_llm, prompt=few_shot_prompt)

    def detect_intents(self, msg):
        """
        Detects the user intent via langchain execution of GPT.
        """
        return self.chain.run(utterance=msg), 0.5


def main():
    rclpy.init()
    intent_detector = GptIntentDetector()
    rclpy.spin(intent_detector)
    intent_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
