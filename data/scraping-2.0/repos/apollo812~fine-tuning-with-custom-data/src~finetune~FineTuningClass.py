import os
import openai
import random
import time
import json
from itertools import cycle

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from dotenv import load_dotenv

class FineTuningClass:
    def __init__(self, data_path, api_key='', model='gpt-3.5-turbo', temperature=0.3, max_retries=5):
        self.data_path = data_path
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = 60
        self.set_api_key(api_key)
        self.set_document(data_path)
        self.generate_subfolder(data_path)

    def set_api_key(self, api_key):
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.api_key
            openai.api_key = self.api_key
            return True
        else:
            # Handle the absence of the environment variable
            # You might want to log an error, raise an exception, or provide a default value
            # For example, setting a default value
            os.environ["OPENAI_API_KEY"] = "your_default_api_key"
            openai.api_key = "openai_api_key"
            return False


    def set_document(self, data_path):
        self.documents = SimpleDirectoryReader(
            data_path
        ).load_data()

    def generate_subfolder(self, data_path):
        subfolder_name = "generated_data"
        subfolder_path = os.path.join(data_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

    def train_generation(self):
        for attempt in range(1, self.max_retries + 1):
            try:
                half_point = len(self.documents) // 2  # Get the index for the halfway point of the documents
                random.seed(42)
                random.shuffle(self.documents)

                gpt_35_context = ServiceContext.from_defaults(
                    llm=OpenAI(model=self.model, temperature=self.temperature)
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
                        
                    print(f"Generated {len(questions)} questions")
                    
                    with open(output_file, "w") as f:
                        for question in questions:
                            f.write(question + "\n")

                generate_and_save_questions(self.documents[:half_point], f'{self.data_path}/generated_data/train_questions.txt', 40)
                generate_and_save_questions(self.documents[half_point:], f'{self.data_path}/generated_data/eval_questions.txt', 40)

                break
            except Exception as e:
                print(f"Error in attempt {attempt}: {e}")
                time.sleep(self.retry_delay * attempt)
                
    def initial_eval(self):
        questions = []
        with open(f'{self.data_path}/eval_questions.txt', "r") as f:
            for line in f:
                questions.append(line.strip())

        # limit the context window to 2048 tokens so that refine is used
        gpt_35_context = ServiceContext.from_defaults(
            llm=OpenAI(model=self.model, temperature=self.temperature), context_window=2048
        )

        index = VectorStoreIndex.from_documents(
            self.documents, service_context=gpt_35_context
        )

        query_engine = index.as_query_engine(similarity_top_k=2)
        contexts = []
        answers = []

        for question in questions:
            response = query_engine.query(question)
            contexts.append([x.node.get_content() for x in response.source_nodes])
            answers.append(str(response))


        # initial eval
        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
        )

        result = evaluate(ds, [answer_relevancy, faithfulness])
        print(result)

    def jsonl_generation(self):
        finetuning_handler = OpenAIFineTuningHandler()
        callback_manager = CallbackManager([finetuning_handler])

        gpt_4_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-4", temperature=self.temperature),
            context_window=2048,  # limit the context window artifically to test refine process
            callback_manager=callback_manager,
        )

        questions = []
        with open(f'{self.data_path}/generated_data/train_questions.txt', "r") as f:
            for line in f:
                questions.append(line.strip())

        try:
            index = VectorStoreIndex.from_documents(
                self.documents, service_context=gpt_4_context
            )
            query_engine = index.as_query_engine(similarity_top_k=2)
            for question in questions:
                response = query_engine.query(question)
        except Exception as e:
            # Handle the exception here, you might want to log the error or take appropriate action
            print(f"An error occurred: {e}")
        finally:
            finetuning_handler.save_finetuning_events(f'{self.data_path}/generated_data/finetuning_events.jsonl')

        
    def finetune(self):
        file_upload = openai.files.create(file=open(f'{self.data_path}/generated_data/finetuning_events.jsonl', "rb"), purpose="fine-tune")
        print("Uploaded file id", file_upload.id)

        while True:
            print("Waiting for file to process...")
            file_handle = openai.files.retrieve(file_id=file_upload.id)
            if file_handle and file_handle.status == "processed":
                print("File processed")
                break
            time.sleep(3)

        try:
            job = openai.fine_tuning.jobs.create(training_file=file_upload.id, model=self.model)

            while True:
                print("Waiting for fine-tuning to complete...")
                job_handle = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=job.id)
                if job_handle.status == "succeeded":
                    print("Fine-tuning complete")
                    print("Fine-tuned model info", job_handle)
                    print("Model id", job_handle.fine_tuned_model)

                    with open(f'{self.data_path}/generated_data/model.txt', "w") as f:
                        f.write(job_handle.fine_tuned_model + "\n")
                    
                    # Load the JSON data from the file
                    with open(f'{self.data_path}/payload/chatting_payload.json', 'r') as file:
                        payload = json.load(file)

                    # Update the model_id with specific data
                    payload['model_id'] = job_handle.fine_tuned_model

                    # Write the updated JSON back to the file
                    with open(f'{self.data_path}/payload/chatting_payload.json', 'w') as file:
                        json.dump(payload, file, indent=4)

                    return job_handle.fine_tuned_model
                time.sleep(3)
        except Exception as e:
            print(f"An error occurred during fine-tuning: {e}")
