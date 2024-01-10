"""This module implements a chat structured around a decision tree."""

import random
from enum import Enum

import openai
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

import common as cm

logger = cm.setup_logger(__name__)


class Response(str, Enum):
    A = "A"
    B = "B"


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Choice(BaseModel):
    prompt: str
    next_node_id: int


class DecisionNode(BaseModel):
    """Represents a node in a decision tree chatbot.

    Attributes:
        id: Unique integer ID of node
        name: Name of node
        prompt: Prompt template to query user
        choices: Possible choices to transition to next nodes
    """

    id: int
    name: str
    prompt: str | None = None
    choices: list[Choice] = []


class DecisionTree(BaseModel):
    nodes: list[DecisionNode]


class Answer(BaseModel):
    research: str
    assessment: str
    response: Response
    confidence: float


class Message(BaseModel):
    role: Role
    content: str


class ExcerptConfiguration(BaseModel):
    literal_queries: list[str]
    semantic_queries: list[str]


class ChatTreeConfiguration(BaseModel):
    chat_initialization: list[Message]
    excerpt_configuration: ExcerptConfiguration
    initial_node_id: int
    error_node_id: int
    nodes: list[DecisionNode]


def log_stoppage(doi, location, reason):
    # NOTE: closes over logger.
    logger.info(f"Eval stopped: [{doi}] at [{location}], reason: [{reason}]")


def lookup_by(xs, item, by="id"):
    return next((x for x in xs if getattr(x, by) == item), None)


def permute(xs):
    out = list(xs).copy()
    random.shuffle(out)
    return out


def build_prompt(node, context, label_enum=Response):
    prompt = node.prompt.format(**context) + "\n"

    labels = [e.value for e in label_enum]
    assert len(labels) == len(node.choices)

    # Randomize option order.
    label_map = dict(zip(labels, permute(node.choices)))

    for lb, choice in label_map.items():
        prompt += f"[{lb}]: {choice.prompt}\n"

    return (
        prompt,
        dict((lb, choice.next_node_id) for lb, choice in label_map.items()).get,
    )


def evaluate_node(node, chat, messages, context, max_reprompts=0):
    """Evaluates a decision node.

    Builds a prompt, queries the chat model, and returns the response.

    """
    prompt, decision_fn = build_prompt(node, context)

    logger.debug(f"[{context['doi']}] [PROMPT] {prompt}")
    messages += [HumanMessage(content=prompt)]

    for n in range(max_reprompts + 1):
        try:
            response = chat(messages)
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request error: [{e}]")
            answer = None
            next_node_id = None
            break
        logger.debug(f"[{context['doi']}] [RESPONSE] {response.content}")
        messages += [response]

        try:
            answer = Answer.model_validate_json(response.content)
            next_node_id = decision_fn(answer.response)
            break
        except ValidationError:
            # Try to clean the response by removing markdown code fencing
            cleaned_response = (
                response.content.replace("```json", "").replace("```", "").strip()
            )
            try:
                answer = Answer.model_validate_json(cleaned_response)
                logger.debug("JSON became valid after markdown fence removal.")
                next_node_id = decision_fn(answer.response)
                break
            except ValidationError:
                # If it still fails, then reprompt
                reprompt = """\
                That response is not conformant JSON.

                Consult the chat history carefully and answer the question again, using
                the specified format.
                """
                messages += [HumanMessage(content=reprompt)]
                logger.info(f"Reprompt attempt [{n}]")

    else:
        answer = None
        next_node_id = None

    logger.info(f"[{context['doi']}] [Next Node ID] {next_node_id}")
    return messages, answer, next_node_id


def evaluate_tree(chat, tree, context, max_reprompts=0, seed=1111):
    """Evaluates an entire decision tree by traversing nodes based on chat responses."""
    messages = [SystemMessage(content=tree.chat_initialization[0].content)]

    random.seed(f"{context['doi']}-{seed}")

    current_node = lookup_by(tree.nodes, tree.initial_node_id)

    node_history = [current_node.name]
    answer_history = []
    while True:
        messages, answer, next_node_id = evaluate_node(
            current_node, chat, messages, context, max_reprompts=max_reprompts
        )
        answer_history += [answer]
        if next_node_id is None:
            node_history += [lookup_by(tree.nodes, tree.error_node_id).name]
            log_stoppage(context["doi"], current_node.name, "invalid-json")
            break

        next_node = lookup_by(tree.nodes, next_node_id)
        node_history += [next_node.name]
        if not next_node.choices:
            log_stoppage(context["doi"], current_node.name, "terminal-node")
            break

        logger.info(f"Transitioning from [{current_node.name}] to [{next_node.name}]")

        current_node = next_node

    return messages, answer_history, node_history
