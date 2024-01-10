from os import getenv, path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Priject imports
from publishers.IPublisher import IPublisher

class InstagramPublisher(IPublisher):
    def __init__(self, credentials: dict) -> None:
        super().__init__(credentials=credentials)
    
    def login(self) -> None:
        pass

    def build(self) -> None:
        content = dict()
        if self.ig: 
            # TODO: Below line assumes gif file is present
            # adopt defensive programming here to prevent errors
            content["image"] = path.join('./videos/', self.story_name + ".gif")

            # Get caption from the story text in English
            text_to_get_caption_from = self.text.get("English")

            # Set up the translation prompt, grammer (e.g. articles) omitted for brevity
            caption_template = '''
            Create a highly engaging summary from the given text between tags <TEXT> and </TEXT>, for publishing as caption on a Instagram post.

            Return only the summary, not the original text. Character Vikram should not be summary.

            <TEXT>{text}</TEXT>
            '''

            caption_prompt = PromptTemplate(template=caption_template, input_variables=['text'])

            chain2 = LLMChain(llm=self.llm,prompt=caption_prompt)

            # Extract the translated text from the API response
            input = {'text': text_to_get_caption_from}
            content["caption"] = chain2.run(input)
    
    def publish(self, content) -> None:
        image = content.get("image")
        caption = content.get("caption")

    def logout(self) -> None:
        pass    