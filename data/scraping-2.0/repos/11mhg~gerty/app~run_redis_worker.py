from rq import Connection
from rq.local import LocalStack
from rq.worker import SimpleWorker
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain import PromptTemplate
import argparse, json, os
from gerty.gerty import Gerty
from gerty.embed_db import valid_path
from typing import Union, Optional
from gerty.utils import import_from_path 
from rq.logutils import blue, green, yellow
from rq.connections import pop_connection, push_connection
from rq.utils import as_text, utcnow
from rq.timeouts import JobTimeoutException
import traceback 
import sys


def extract_memory(chain):
    return chain.memory.chat_memory.messages

def serialize_memory(chain: Chain) -> str :
    messages = extract_memory(chain)
    messages = messages_to_dict(messages)
    return json.dumps( messages )

def deserialize_memory(messages_str: Optional[str], llm) -> ConversationSummaryBufferMemory:
    if messages_str is None or len(messages_str) == 0:
        return ConversationSummaryBufferMemory(
            llm = llm,
            memory_key="chat_history", 
            ai_prefix="Assistant",
            return_messages = True,
            max_token_limit=512,
        )
    messages = json.loads( messages_str )
    messages = messages_from_dict( messages )
    retrieved_chat_history = ChatMessageHistory(messages=messages)
    prompts = import_from_path(os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'gerty',
            'models',
            'nous-hermes-llama-2-7b',
            'prompts.py'
        )
    ), 'prompts')
    return ConversationSummaryBufferMemory(
        llm = llm, 
        memory_key="chat_history",
        ai_prefix="Assistant",
        chat_memory = retrieved_chat_history, 
        return_messages=True,
        max_token_limit=512,
        prompt=PromptTemplate(input_variables = prompts.DEFAULT_CONVO_SUMMARY_VARIABLES,
                              template = prompts.DEFAULT_CONVO_SUMMARY_TEMPLATE ),
    )
class GertyWorker(SimpleWorker):
    """
    A simple gerty worker that initializes gerty and gets it ready for use/reuse.
    """
    def __init__(self, queues, n_ctx, db, model_path, *args, **kwargs):
        self.__gerty: Gerty = Gerty(n_ctx = n_ctx, model_path = model_path)
        self.__gerty.load_db( db )
        self.__qa = self.__gerty.get_qa_model()
        super(SimpleWorker, self).__init__(queues = queues, *args, **kwargs)
        

    def execute_job(self, job, queue):
        job.args = (*job.args, self.__qa)
        return self.perform_job(job, queue)

    def perform_job(self, job, queue) -> bool:
        """Performs the actual work of a job.  Will/should only be called
        inside the work horse's process.

        Args:
            job (Job): The Job
            queue (Queue): The Queue

        Returns:
            bool: True after finished.
        """
        push_connection(self.connection)
        started_job_registry = queue.started_job_registry
        self.log.debug('Started Job Registry set.')

        try:
            remove_from_intermediate_queue = len(self.queues) == 1
            self.prepare_job_execution(job, remove_from_intermediate_queue)

            job.started_at = utcnow()
            timeout = job.timeout or self.queue_class.DEFAULT_TIMEOUT
            with self.death_penalty_class(timeout, JobTimeoutException, job_id=job.id):
                self.log.debug('Performing Job...')
                rv = job._execute()
                self.log.debug('Finished performing Job ID %s', job.id)

            job.ended_at = utcnow()

            # Pickle the result in the same try-except block since we need
            # to use the same exc handling when pickling fails
            job._result = rv
            job.args = job.args[:len(job.args)-1] 
            job.heartbeat(utcnow(), job.success_callback_timeout)
            job.execute_success_callback(self.death_penalty_class, rv)

            self.handle_job_success(job=job, queue=queue, started_job_registry=started_job_registry)
        except:  # NOQA
            job.args = job.args[:len(job.args)-1]
            self.log.debug('Job %s raised an exception.', job.id)
            job.ended_at = utcnow()
            exc_info = sys.exc_info()
            exc_string = ''.join(traceback.format_exception(*exc_info))

            try:
                job.heartbeat(utcnow(), job.failure_callback_timeout)
                job.execute_failure_callback(self.death_penalty_class, *exc_info)
            except:  # noqa
                exc_info = sys.exc_info()
                exc_string = ''.join(traceback.format_exception(*exc_info))

            self.handle_job_failure(
                job=job, exc_string=exc_string, queue=queue, started_job_registry=started_job_registry
            )
            self.handle_exception(job, *exc_info)
            return False

        finally:
            pop_connection()

        self.log.info('%s: %s (%s)', green(job.origin), blue('Job OK'), job.id)
        if rv is not None:
            self.log.debug('Result: %r', yellow(as_text(str(rv))))

        if self.log_result_lifespan:
            result_ttl = job.get_result_ttl(self.default_result_ttl)
            if result_ttl == 0:
                self.log.info('Result discarded immediately')
            elif result_ttl > 0:
                self.log.info('Result is kept for %s seconds', result_ttl)
            else:
                self.log.info('Result will never expire, clean up result key manually')

        return True



def query_job(messages, query, qa):
    print("Messages: ", messages)
    print("query: ", query)
    llm = qa.question_generator.llm
    memory = deserialize_memory( messages, llm )
    qa.memory = memory
    response = qa.run( query ).strip()
    return {
        'response': response,
        'messages': serialize_memory(qa)
    }


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser("Launch redis based gerty worker.")
    parser.add_argument("-db", "--database", type=valid_path, help="Cache with knowledge base")
    parser.add_argument("--context-length", type=int, default=2048, help = "Context length")
    parser.add_argument("--model_path", type=str, default = os.path.join(
            os.path.dirname(__file__),
            "models",
            "nous-hermes-llama-2-7b"
        ), help = "Path to model directory."
    )
    args = parser.parse_args()

    with Connection():
        worker = GertyWorker(queues = ['default'], 
            n_ctx = args.context_length, 
            db = args.database, 
            model_path = args.model_path 
        )
        worker.work()
