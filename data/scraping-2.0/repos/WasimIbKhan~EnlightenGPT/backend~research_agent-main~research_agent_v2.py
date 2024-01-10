import uuid
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from yachalk import chalk

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from chains_v2.create_questions import QuestionCreationChain
from chains_v2.most_pertinent_question import MostPertinentQuestion
from chains_v2.retrieval_qa import retrieval_qa
from chains_v2.research_compiler import research_compiler
from chains_v2.question_atomizer import QuestionAtomizer
from chains_v2.refine_answer import RefineAnswer

def language_model(
    model_name: str = "gpt-3.5-turbo", temperature: float = 0, verbose: bool = False
):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, verbose=verbose)
    return llm

from helpers.response_helpers import result2QuestionsList
from helpers.response_helpers import qStr2Dict
from helpers.questions_helper import getAnsweredQuestions
from helpers.questions_helper import getUnansweredQuestions
from helpers.questions_helper import getSubQuestions
from helpers.questions_helper import getHopQuestions
from helpers.questions_helper import getLastQuestionId
from helpers.questions_helper import markAnswered
from helpers.questions_helper import getQuestionById

def print_iteration(current_iteration):
    print(
        chalk.bg_yellow_bright.black.bold(
            f"\n   Iteration - {current_iteration}  ▷▶  \n"
        )
    )


def print_unanswered_questions(unanswered):
    print(
        chalk.cyan_bright("** Unanswered Questions **"),
        chalk.cyan("".join([f"\n'{q['id']}. {q['question']}'" for q in unanswered])),
    )


def print_next_question(current_question_id, current_question):
    print(
        chalk.magenta.bold("** 🤔 Next Questions I must ask: **\n"),
        chalk.magenta(current_question_id),
        chalk.magenta(current_question["question"]),
    )


def print_answer(current_question):
    print(
        chalk.yellow_bright.bold("** Answer **\n"),
        chalk.yellow_bright(current_question["answer"]),
    )


def print_final_answer(answerpad):
    print(
        chalk.white("** Refined Answer **\n"),
        chalk.white(answerpad[-1]),
    )


def print_max_iterations():
    print(
        chalk.bg_yellow_bright.black.bold(
            "\n ✔✔  Max Iterations Reached. Compiling the results ...\n"
        )
    )

def print_result(result):
    print(chalk.italic.white_bright((result["text"])))


def print_sub_question(q):
    print(chalk.magenta.bold(f"** Sub Question **\n{q['question']}\n{q['answer']}\n"))

class Agent:
    ## Create chains
    def __init__(self, agent_settings, scratchpad, store, verbose):
        self.store = store
        self.scratchpad = scratchpad
        self.agent_settings = agent_settings
        self.verbose = verbose
        self.question_creation_chain = QuestionCreationChain.from_llm(
            language_model(
                temperature=self.agent_settings["question_creation_temperature"]
            ),
            verbose=self.verbose,
        )
        self.question_atomizer = QuestionAtomizer.from_llm(
            llm=language_model(
                temperature=self.agent_settings["question_atomizer_temperature"]
            ),
            verbose=self.verbose,
        )
        self.most_pertinent_question = MostPertinentQuestion.from_llm(
            language_model(
                temperature=self.agent_settings["question_creation_temperature"]
            ),
            verbose=self.verbose,
        )
        self.refine_answer = RefineAnswer.from_llm(
            language_model(
                temperature=self.agent_settings["refine_answer_temperature"]
            ),
            verbose=self.verbose,
        )

    def run(self, question):
        ## Step 0. Prepare the initial set of questions
        atomized_questions_response = self.question_atomizer.run(
            question=question,
            num_questions=self.agent_settings["num_atomistic_questions"],
        )

        self.scratchpad["questions"] += result2QuestionsList(
            question_response=atomized_questions_response,
            type="subquestion",
            status="unanswered",
        )

        for q in self.scratchpad["questions"]:
            q["answer"], q["documents"] = retrieval_qa(
                llm=language_model(
                    temperature=self.agent_settings["qa_temperature"],
                    verbose=self.verbose,
                ),
                retriever=self.store.as_retriever(
                    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
                ),
                question=q["question"],
                answer_length=self.agent_settings["intermediate_answers_length"],
                verbose=self.verbose,
            )
            q["status"] = "answered"
            print_sub_question(q)

        
        current_context = "".join(
            f"\n{q['id']}. {q['question']}\n{q['answer']}\n"
            for q in self.scratchpad["questions"]
        )
        
        self.scratchpad["answerpad"] += [current_context]

        current_iteration = 0

        while True:
            current_iteration += 1
            print_iteration(current_iteration)

            # STEP 1: create questions
            start_id = getLastQuestionId(self.scratchpad["questions"]) + 1
            questions_response = self.question_creation_chain.run(
                question=question,
                context=current_context,
                previous_questions=[
                    "".join(f"\n{q['question']}") for q in self.scratchpad["questions"]
                ],
                num_questions=self.agent_settings["num_questions_per_iteration"],
                start_id=start_id,
            )
            self.scratchpad["questions"] += result2QuestionsList(
                question_response=questions_response,
                type="hop",
                status="unanswered",
            )

            # STEP 2: Choose question for current iteration
            unanswered = getUnansweredQuestions(self.scratchpad["questions"])
            unanswered_questions_prompt = self.unanswered_questions_prompt(unanswered)
            print_unanswered_questions(unanswered)
            response = self.most_pertinent_question.run(
                original_question=question,
                unanswered_questions=unanswered_questions_prompt,
            )
            current_question_dict = qStr2Dict(question=response)
            current_question_id = current_question_dict["id"]
            current_question = getQuestionById(
                self.scratchpad["questions"], current_question_id
            )
            print_next_question(current_question_id, current_question)

            # STEP 3: Answer the question
            current_question["answer"], current_question["documents"] = retrieval_qa(
                llm=language_model(
                    temperature=self.agent_settings["qa_temperature"],
                    verbose=self.verbose,
                ),
                retriever=self.store.as_retriever(
                    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
                ),
                question=current_question["question"],
                answer_length=self.agent_settings["intermediate_answers_length"],
                verbose=self.verbose,
            )
            markAnswered(self.scratchpad["questions"], current_question_id)
            print_answer(current_question)
            current_context = current_question["answer"]

            ## STEP 4: refine the answer
            refinement_context = current_question["question"] + "\n" + current_context
            refine_answer = self.refine_answer.run(
                question=question,
                context=refinement_context,
                answer=self.get_latest_answer(),
            )
            self.scratchpad["answerpad"] += [refine_answer]
            print_final_answer(self.scratchpad["answerpad"])

            if current_iteration > self.agent_settings["max_iterations"]:
                print_max_iterations()
                break

    def unanswered_questions_prompt(self, unanswered):
        return (
            "[" + "".join([f"\n{q['id']}. {q['question']}" for q in unanswered]) + "]"
        )

    def notes_prompt(self, answered_questions):
        return "".join(
            [
                f"{{ Question: {q['question']}, Answer: {q['answer']} }}"
                for q in answered_questions
            ]
        )

    def get_latest_answer(self):
        answers = self.scratchpad["answerpad"]
        answer = answers[-1] if answers else ""
        return answer


# In[9]:
def get_final_answer(input_question, namespace):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    embeddings = OpenAIEmbeddings()
    max_weber_store = Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings, namespace=namespace)

    run_id = str(uuid.uuid4())

    scratchpad = {
        "questions": [],  # list of type Question
        "answerpad": [],
    }

    store = max_weber_store

    agent_settings = {
        "max_iterations": 1,
        "num_atomistic_questions": 2,
        "num_questions_per_iteration": 4,
        "question_atomizer_temperature": 0,
        "question_creation_temperature": 0.4,
        "question_prioritisation_temperature": 0,
        "refine_answer_temperature": 0,
        "qa_temperature": 0,
        "analyser_temperature": 0,
        "intermediate_answers_length": 200,
        "answer_length": 500,
    }

    agent = Agent(agent_settings, scratchpad, store, True)
    agent.run(input_question)


    final_answer = scratchpad['answerpad'][-1] if scratchpad['answerpad'] else ''
    return final_answer




