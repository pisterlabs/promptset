from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import emoji

EXAMPLES = [
    {"user": "Who is the president of Gabon?",
        "assistant": "🇬🇦 President of Gabon"},
    {"user": "Who is Julien Chaumond?", 
        "assistant": "🧑 Julien Chaumond"},
    {"user": "what is 1 + 1?", 
        "assistant": "🔢 Simple math operation"},
    {"user": "What are the latest news?", 
        "assistant": "📰 Latest news"},
    {"user": "How to make a great cheesecake?",
        "assistant": "🍰 Cheesecake recipe"},
    {"user": "what is your favorite movie? do a short answer.",
        "assistant": "🎥 Favorite movie"},
    {"user": "Explain the concept of artificial intelligence in one sentence", 
        "assistant": "🤖 AI definition"}]


class TitleLlm:
    def __init__(self):
        self.chat_model = ChatOpenAI(model="gpt-3.5-turbo-0301")

    def __text_has_emoji(self, text):
        for character in text:
            if emoji.is_emoji(character):
                return True
        return False

    async def generate_summary(self, initial_user_message: str) -> str:
        instruction = "You are a summarization AI. You'll never answer a user's question directly, but instead summarize the user's request into a single short sentence of four words or less. Always start your answer with an emoji relevant to the summary."
        example_template = """
        user: {user}
        assistant: {assistant}
        """
        example_prompt = PromptTemplate(
            input_variables=["user", "assistant"],
            template=example_template
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=EXAMPLES,
            example_prompt=example_prompt,
            prefix=instruction,
            suffix="user: {input}\nassistant:",
            input_variables=["input"],
            example_separator="\n\n",
        )
        response = await self.chat_model.ainvoke(
            few_shot_prompt.format(input=initial_user_message))
        summary = response.content.strip()

        # Add an emoji if none is found in the first three characters
        if not self.__text_has_emoji(summary):
            summary = "💬 " + summary
        return summary
