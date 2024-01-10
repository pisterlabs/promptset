import random
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator
from dotenv import load_dotenv
import os
import time
import logging
from itertools import cycle


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is not None:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    os.environ["OPENAI_API_KEY"] = "your_default_api_key"

data_path = "./test/regression/regression_test005"

max_retries = 5
retry_delay = 60  # in seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for attempt in range(1, max_retries + 1):
    try:
        documents = SimpleDirectoryReader(data_path).load_data()
        half_point = len(documents) // 2  # Get the index for the halfway point of the documents
        random.seed(42)
        random.shuffle(documents)

        gpt_35_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
        )

        question_gen_query = (
            "You are a Teacher/ Professor. Your task is to setup "
            "a quiz/examination. Using the provided context, formulate "
            "a single question that captures an important fact from the "
            "context. Restrict the question to the context information provided."
        )

        def generate_and_save_questions(documents, output_file, num_questions):
            dataset_generator = DatasetGenerator.from_documents(
                documents,
                question_gen_query=question_gen_query,
                service_context=gpt_35_context
            )
            questions = []
            # Create an iterator that cycles through available documents
            documents_cycle = cycle(documents)
            
            # Generate questions until reaching the desired count
            while len(questions) < num_questions:
                # Use the next document in the cycle
                next_document = next(documents_cycle)
                dataset_generator = dataset_generator.from_documents([next_document])
                
                # Generate questions from the updated dataset
                new_questions = dataset_generator.generate_questions_from_nodes(num=num_questions - len(questions))
                questions.extend(new_questions)
                
            logger.info(f"Generated {len(questions)} questions")
            
            with open(output_file, "w") as f:
                for question in questions:
                    f.write(question + "\n")

        generate_and_save_questions(documents[:half_point], f'{data_path}/generated_data/train_questions.txt', 40)
        generate_and_save_questions(documents[half_point:], f'{data_path}/generated_data/eval_questions.txt', 40)

        break
    except Exception as e:
        logger.error(f"Error in attempt {attempt}: {e}")
        time.sleep(retry_delay * attempt)

logger.info("Question generation process completed.")