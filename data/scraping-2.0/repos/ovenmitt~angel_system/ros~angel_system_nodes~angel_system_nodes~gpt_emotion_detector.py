from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import openai
import os
import rclpy

from angel_system_nodes.base_emotion_detector import BaseEmotionDetector, LABEL_MAPPINGS

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# The following are few shot examples when prompting GPT.
FEW_SHOT_EXAMPLES = [
    {
        "utterance": "Go back to the previous step you dumb machine!",
        "label": "negative.",
    },
    {"utterance": "Next step, please.", "label": "neutral"},
    {"utterance": "We're doing great and I'm learning a lot!", "label": "positive"},
]


class GptEmotionDetector(BaseEmotionDetector):
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

        all_labels_parenthetical = _labels_list_parenthetical_str(
            list(LABEL_MAPPINGS.values())
        )
        all_labels = _labels_list_str(list(LABEL_MAPPINGS.values()))

        # Define the few shot template.
        template = (
            f"Utterance: {{utterance}}\nEmotion {all_labels_parenthetical}: {{label}}"
        )
        example_prompt = PromptTemplate(
            input_variables=["utterance", "label"], template=template
        )
        prompt_instructions = f"Classify each utterance as {all_labels}.\n"
        inference_sample = (
            f"Utterance: {{utterance}}\nIntent {all_labels_parenthetical}:"
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=FEW_SHOT_EXAMPLES,
            example_prompt=example_prompt,
            prefix=prompt_instructions,
            suffix=inference_sample,
            input_variables=["utterance"],
            example_separator="\n",
        )
        openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            max_tokens=1,
        )
        return LLMChain(llm=openai_llm, prompt=few_shot_prompt)

    def get_inference(self, msg):
        """
        Detects the user intent via langchain execution of GPT.
        """
        return (self.chain.run(utterance=msg.utterance_text), 0.5)


def main():
    rclpy.init()
    emotion_detector = GptEmotionDetector()
    rclpy.spin(emotion_detector)
    emotion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
