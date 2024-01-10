import json
import time
import os
from datetime import datetime, timedelta

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

from VectorKnowledgeGraph import VectorKnowledgeGraph

from langchain.schema import SystemMessage, HumanMessage, Document, AIMessage, AgentAction
import threading

from util import load_secrets


class ConversationFileLogger:
    def __init__(self, directory="logs"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_log_file_path(self, date_str):
        return os.path.join(self.directory, f"{date_str}.txt")

    def log_tool_output(self, tool_name, output):
        # create a directory for the tool if it doesn't exist
        tool_directory = os.path.join(self.directory, tool_name)
        if not os.path.exists(tool_directory):
            os.makedirs(tool_directory)
            # create a timestamped file for the output (one output per file)
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        file_path = os.path.join(tool_directory, f"{timestamp}.txt")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(output)
        # return the path to the file
        return file_path

    def log_message(self, message_to_log, name=""):
        # Write the message to the log file
        date_str = time.strftime("%Y-%m-%d", time.localtime())
        with open(self.get_log_file_path(date_str), 'a', encoding="utf-8") as f:
            f.write(message_to_log + '\n')

    def load_last_n_lines(self, n):
        lines_to_return = []
        current_date = datetime.now()
        while n > 0 and current_date > datetime(2000, 1, 1):  # Assuming logs won't be from before the year 2000
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = self.get_log_file_path(date_str)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    if len(lines) <= n:
                        lines_to_return = lines + lines_to_return
                        n -= len(lines)
                    else:
                        lines_to_return = lines[-n:] + lines_to_return
                        n = 0
            current_date -= timedelta(days=1)

        return lines_to_return


class LongTermMemoryStore:
    def __init__(self, model, agent_name="", lines_to_load=50):
        self.thread_lock = threading.Lock()
        self.conversation_logger = ConversationFileLogger(agent_name + "_logs")
        self.message_buffer = self.conversation_logger.load_last_n_lines(lines_to_load)
        self.ltm_processing_size = 10  # number of messages to process at a time
        self.ltm_overlap_size = 0  # number of messages to overlap between processing
        self.unprocessed_message_count = 0  # number of messages that have not been processed
        self.max_buffer_length = 50  # number of messages to overlap between processing
        self.current_topic = "<nothing yet>"
        self.topics = []
        self.relevant_entities = ["<not yet set>"]
        self.knowledge_store = VectorKnowledgeGraph(path=agent_name + "_GraphStoreMemory")
        self.model = model

    def accept_message(self, message, name=""):
        # create a timestamped message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # message has a field called "content" that contains the text of the message, then store it
        if isinstance(message, AIMessage):
            message_text = f"({timestamp}) {name}: {message.content}"
        else:
            message_text = f"({timestamp}) {name}: {message}"

        # log the message
        self.conversation_logger.log_message(message_text)

        self.append_to_message_buffer(message_text)

    def append_to_message_buffer(self, message_text, source="conversation"):
        # add the message to the buffer
        self.message_buffer.append(message_text)
        self.unprocessed_message_count += 1
        if self.unprocessed_message_count >= self.ltm_processing_size:
            # create a copy of the last unprocessed messages
            last_n_messages = self.message_buffer[-self.unprocessed_message_count:]

            # might be unnecessary, but just in case make a copy of the buffer
            message_buffer_copy = last_n_messages.copy()

            # adjust the buffer length (keep this from growing forever)
            self.message_buffer = self.message_buffer[-self.max_buffer_length:]

            # we have processed all the messages in the buffer, ensure that we have overlap if desired
            self.unprocessed_message_count = self.ltm_overlap_size

            # get the log reference to store with the triples
            date_str = time.strftime("%Y-%m-%d", time.localtime())
            log_reference = self.conversation_logger.get_log_file_path(date_str)

            # process the batch of messages in a separate thread (as a one shot daemon)
            threading.Thread(target=self.process_buffer, args=(message_buffer_copy, source, log_reference), daemon=True).start()

    def accept_tool_output(self, response):
        for step in response["intermediate_steps"]:
            step_input = step[0]
            step_output = step[1]

            if isinstance(step_input, AgentAction):
                # create a timestamped message
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                tool_info = f"(Tool:{step_input.tool} Tool Input:{step_input.tool_input}"
                string_to_log = f"({timestamp}) System: {tool_info} Tool Output:{step_output} "


                # if output is longer than 256 characters, write it to a separate file
                if len(step_output) > 256:
                    log_file_path = self.conversation_logger.log_tool_output(step_input.tool, string_to_log)
                    log_reference = log_file_path
                    step_output = log_file_path
                else:
                    date_str = time.strftime("%Y-%m-%d", time.localtime())
                    log_reference = self.conversation_logger.get_log_file_path(date_str)

                threading.Thread(target=self.process_buffer, args=(string_to_log, tool_info, log_reference), daemon=True).start()
                # log the message
                string_to_log = f"({timestamp}) System: {tool_info} Tool Output:{step_output} "
                self.conversation_logger.log_message(string_to_log)

    def process_buffer(self, message_buffer, reference="conversation", log_reference=""):
        # lock the thread
        self.thread_lock.acquire()
        print(f"\nStarting to process a batch of messages from {reference}")
        try:
            # get the graph from the conversation
            self.update_ltm(message_buffer, reference, log_reference)
        finally:
            # release the lock
            self.thread_lock.release()
            print("\nFinished processing a batch of messages")

    def update_ltm(self, buffer, reference="conversation", log_reference=""):
        # create a string from the conversation buffer using join
        conversation_string = "\n".join(buffer)

        # Prepare metadata for the graph
        # Get the current timestamp
        timestamp = datetime.now().isoformat()
        metadata = {'timestamp': timestamp, "reference": reference, "log_reference": log_reference}

        # update the graph from the conversation string
        self.knowledge_store.process_text(conversation_string, metadata=metadata)
        self.knowledge_store.save()

    def get_current_topic(self):
        return self.current_topic

    def get_relevant_entities(self):
        return self.relevant_entities

    def extract_main_topics(self, input_text, num_topics=3):
        extract_topics_prompt = f"""
        You are tasked with identifying the speakers and top {num_topics} main topics from the text provided. 
        Extract the topics based on the prominence and frequency of terms and concepts mentioned in the text.
        List the topics in descending order of importance. Also, if there is a new topic at the end of the text,
        ennsure that it is included in the list of topics.
        Format the output as a JSON string like this:
        {{ "topics": ["speaker 1", "speaker 2", "topic 1", "topic 2"] }}
        """

        chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        message = chat_llm(
            [
                SystemMessage(role="TopicExtractor", content=extract_topics_prompt),
                HumanMessage(content=input_text),
            ]
        )

        # Assume message.content contains a JSON string with the topics
        data = json.loads(message.content)
        topics = data.get('topics', [])

        return topics

    def summarize_history(self, summary_start_index=-100, summary_end_index=-15):
        # summarize conversation to this point
        summary_chain = load_summarize_chain(ChatOpenAI(temperature=0,
                                                        model_name="gpt-3.5-turbo-16k"),
                                             chain_type="stuff")
        # turn the conversation history into a list of documents
        docs = [Document(page_content=msg, metadata={"source": "local"}) for msg in self.message_buffer[summary_start_index:summary_end_index]]
        summary = "This is a summary of the conversation so far: " + summary_chain.run(docs)
        print(summary)
        recent_messages = self.message_buffer[summary_end_index:]
        topics = self.extract_main_topics("\n".join(recent_messages))
        topic_knowledge_summaries = self.knowledge_store.get_summaries_from_topics(topics)
        recent_history = ["These are the last few exchanges: "] + recent_messages
        return recent_history, summary, topic_knowledge_summaries


if __name__ == "__main__":
    load_secrets()
    test_ltm = LongTermMemoryStore(model=None, agent_name="Sophia", lines_to_load=100)
    print("done")
    test_triples = test_ltm.knowledge_store.build_graph_from_noun("Quantum", similarity_threshold=0.5)
    test_triples_2 = test_ltm.knowledge_store.build_graph_from_noun("Heather")
