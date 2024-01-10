import json
import time
import openai
import logging


class FlashcardCreator:

    def __init__(self, assistant_id, api_key):
        self.api_key = api_key
        self.assistant_id = assistant_id

    '''
        OpenAI still hasn't implemented the ability to upload files to the API, so we have to chunk the OCR data.
        This is a workaround to the 30,000 character limit of GPT-4.
        I have added a max_length parameter for future use, but for now, we will use the default value of 30,000.
        We could move the max_length parameter to the __init__ function, but I don't think it's necessary.
        ~Stefan 
        
    '''
    def chunk_text(self, text, max_length=30000):
        # Function to chunk text into parts with max_length
        chunks = []
        current_chunk = ""
        for sentence in text.split('.'):
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + '.'
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + '.'
                logging.debug(f"Chunk {len(chunks)} created")
        if current_chunk:
            chunks.append(current_chunk)
            logging.debug(f"Chunk {len(chunks)}created")
        return chunks

    def ai_generate_flashcards(self, ocr_data):

        #Before we send the OCR data to OpenAI, we need to chunk it into parts
        ocr_chunks = self.chunk_text(ocr_data)

        client = openai.Client(api_key=self.api_key)

        def wait_for_run_completion(thread_id):
            while True:
                runs = client.beta.threads.runs.list(thread_id=thread_id)
                if not runs.data or runs.data[-1].status in ["completed", "failed"]:
                    break
                time.sleep(1)

        #Init response list for the responses that we receive from OpenAI
        all_responses = []
        current_chunk = 1
        for chunk in ocr_chunks:
            logging.debug(f"Processing chunk {current_chunk} of {len(ocr_chunks)}")
            logging.info("Creating thread with OCR chunk...")
            # Create a thread for each chunk
            thread = client.beta.threads.create(
                messages=[{
                    "role": "user",
                    "content": chunk
                }]
            )

            logging.info("Creating run with assistant...")
            client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            logging.info("Waiting for OCR chunk run to complete...")
            wait_for_run_completion(thread.id)

            logging.info("Fetching final response for the chunk...")
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages.data:
                if message.role == 'assistant':
                    # Process each chunk's response
                    message_text = message.content[0].text.value
                    try:
                        if message_text[0] != '{':
                            message_text = message_text[message_text.find('{'):]
                        if message_text[-1] != '}':
                            message_text = message_text[:message_text.rfind('}') + 1]
                        flashcards_chunk = json.loads(message_text)["flashcards"]
                        all_responses.extend(flashcards_chunk)
                    except json.JSONDecodeError:
                        continue

        # Combine all flashcards into a single JSON object
        final_flashcards_json = {"flashcards": all_responses}

        #save final flashcards json to file
        with open('flashcards.json', 'w') as outfile:
            json.dump(final_flashcards_json, outfile)

        return final_flashcards_json