from tasks.abstractTask import AbstractTask
import logging

class EmbeddingTask(AbstractTask):
    GPT_MODEL = "text-embedding-ada-002"
    PHRASE_TO_EMBEDD = "Hawaiian pizza"

    def process_task_details(self):

        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=self.GPT_MODEL
        )
        try:
            query_result = embeddings.embed_query(self.PHRASE_TO_EMBEDD)
        except Exception as e:
            logging.error("Exception occured: {e}")

        logging.info(f"Embedded answer is: {query_result}")
        return query_result

    def solve_task(self):
        self.assignment_solution = self.process_task_details()
        if self.send_to_aidevs:
            logging.info("Sending answer to aidevs...")
            self.return_answer()
        else:
            logging.info("Won't send to aidevs... END")