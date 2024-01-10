from storyconnect.settings import OPENAI_API_KEY
from .exceptions import RoadUnblockerException
import books.models as books_models
import ai_features.utils as utils
import logging
from openai import OpenAI


# openai.api_key = OPENAI_API_KEY
logger = logging.getLogger(__name__)


class RoadUnblocker:
    # openai parameters
    BASE_MODEL = "gpt-3.5-turbo-instruct"
    CHAT_MODEL = "gpt-3.5-turbo-1106"
    MAX_TOKENS = 5000
    TEMPERATURE = 0.2

    SYS_ROLE = "You are an AI writing assistant. You are here to help the user write their story."
    PRE_MESSAGE = "Hello! I am Road Unblocker. Your personal AI writing assistant. I am here to help you write your story."

    def __init__(self):
        self.last_response = None
        self.sys_message = {"role": "system", "content": self.SYS_ROLE}

    # def _summarize_chapter(self, chapter_id):
    #     """Summarizes a chapter using the BASE_MODEL. Stores in ChapterSummary table"""

    #     chapter = books_models.Chapter.objects.get(pk=chapter_id)
    #     prompt = "Summarize the following text:\n\n"
    #     prompt += chapter.content

    #     response = openai.Completion.create(model = self.BASE_MODEL,
    #                                        prompt = prompt,
    #                                        max_tokens = self.MAX_TOKENS,
    #                                        temperature = self.TEMPERATURE,
    #                                        )
    #     summary = response.choices[0]['text']
    #     return summary

    # def _summarize_book(self, book_id):
    #     """Summarizes a book using the BASE_MODEL. Stores in BookSummary table"""

    #     running_summary = ""

    #     book = books_models.Book.objects.get(pk=book_id)
    #     for chapter in book.get_chapters():
    #         ch_sum, created = ai_models.get_or_create_chapter_summary(chapter=chapter)
    #         if created:
    #             ch_sum.summary = self._summarize_chapter(chapter.id)
    #             ch_sum.save()

    #         running_summary += ch_sum.summary + "\n"

    #     response = openai.Completion.create(model = self.BASE_MODEL,
    #                                        prompt = prompt,
    #                                        max_tokens = self.MAX_TOKENS,
    #                                        temperature = self.TEMPERATURE,
    #                                        )
    #     summary = response.choices[0]['text']
    #     return summary

    def _generate_context(self, chapter_id):
        # print("Generating context")
        chapter = books_models.Chapter.objects.get(pk=chapter_id)
        book = chapter.book

        bk_chapters = book.get_chapters()

        messages = [
            self.sys_message,
        ]

        # include summary only if more than one chapter
        if bk_chapters.count() > 1:
            logger.info("more than one chapter")
            u_msg_summary = "Here is a summary of my book so far:\n\n"

            # TODO: Remember that this is now chat model
            logger.info("summarizing book ")
            u_msg_summary += utils.summarize_book_chat(book.id)[0]
            logger.info("summarized book")
            messages.append({"role": "user", "content": u_msg_summary})

        # include chapter content
        u_msg_chapter = "Here is the content of the chapter you are working on:\n\n"
        u_msg_chapter += chapter.content
        messages.append({"role": "user", "content": u_msg_chapter})

        u_ex = "Do you have any suggestions for this chapter?"
        messages.append({"role": "user", "content": u_ex})

        assist_ex = """Chapter Suggestions:

                    1. Enhance the setting: Describe the studio and garden in more detail, using sensory details to immerse the reader in the environment. Expand on the scents, sounds, and visuals to create a vivid atmosphere.

                    2. Deepen the character dynamics: Explore the relationship between Lord Henry and Basil Hallward further. Show their contrasting personalities and perspectives through their dialogue and actions. Highlight the tension and fascination that arises from their discussions about Dorian Gray.

                    3. Develop the mystery of Dorian Gray: Drop subtle hints and foreshadowing about the hidden depths of Dorian's character. Create intrigue around his role in Basil's art and the impact he has on the people around him. Build anticipation for the dark and twisted path that lies ahead.

                    4. Expand on the theme of art and beauty: Use Lord Henry and Basil's conversation to delve deeper into their contrasting views on art and its relationship to the artist and the subject. Examine the idea of art as a reflection of the artist's soul and the potential consequences of revealing too much of oneself in art.

                    5. Foreshadowing and tension: Inject moments of tension and foreshadowing in the chapter to keep the reader engaged. Hint at the potential conflicts and challenges that may arise in the future as Lord Henry and Basil's obsession with Dorian Gray grows.
                    """
        messages.append({"role": "assistant", "content": assist_ex})

        logger.info("Context generated")
        return messages

    def get_suggestions(self, selection, question, chapter_id):
        # logger.info("Getting suggestions")
        logger.info("Generating context for road unblocker")

        messages = self._generate_context(chapter_id)

        if selection is not None and selection != "":
            content = "Heres the particular selection I want to work on:\n" + selection
            messages.append({"role": "user", "content": content})

        messages.append({"role": "user", "content": "Question: " + question})

        # debug write to file
        with open("ai_features/test_files/test_prints.txt", "w") as f:
            for m in messages:
                f.write(str(m) + "\n")

        logger.info("Sending messages to openai")
        client = OpenAI(api_key=OPENAI_API_KEY)

        try:
            self.last_response = client.chat.completions.create(
                timeout=300,
                model=self.CHAT_MODEL,
                messages=messages,
            )
            # returns first suggestion
            # TODO: handle multiple suggestions, serializer and front end give multi suggest not chat bubble
            response_content = self.last_response.choices[0].message.content

            logger.info("Got suggestions")
            logger.info(response_content)

            return response_content
        except Exception as e:
            logger.error("Error in road unblocker")
            logger.error(e)
            raise RoadUnblockerException()
