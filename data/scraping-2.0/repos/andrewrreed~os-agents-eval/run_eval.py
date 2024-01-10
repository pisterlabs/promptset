from langchain.chat_models import ChatOpenAI
from evaluation import run_full_eval
from prompts import build_eval_prompt
from agents import build_hf_agent, build_openai_agent
from langchain.llms import HuggingFaceEndpoint
from chat_wrapper import HuggingFaceChatWrapper
from data import build_dataset


async def run_eval():
    dataset = build_dataset()
    os_agent_info = {
        "zephyr-7b-beta": "https://b64oqapulf4lv8w1.us-east-1.aws.endpoints.huggingface.cloud",
        "mistral-7b-instruct": "https://epho5s2agxpyv657.us-east-1.aws.endpoints.huggingface.cloud",
        "openhermes-2.5-mistral-7b": "https://jsfpmcv7wgv9acvh.us-east-1.aws.endpoints.huggingface.cloud",
        # "solar-10.7b-instruct": "https://ye89fyvdxups3idt.us-east-1.aws.endpoints.huggingface.cloud",
    }

    oai_agent_info = {
        "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview": "gpt-4-1106-preview",
    }

    # build agents
    os_agents = {
        name: build_hf_agent(endpoint) for name, endpoint in os_agent_info.items()
    }
    oai_agents = {
        name: build_openai_agent(model_id) for name, model_id in oai_agent_info.items()
    }
    agents = {**os_agents, **oai_agents}

    # build prometheus evaluator
    eval_endpoint = "https://hg1fvppohh20ufdf.us-east-1.aws.endpoints.huggingface.cloud"
    prometheus_llm = HuggingFaceEndpoint(
        endpoint_url=eval_endpoint,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )
    prometheus_evaluator = HuggingFaceChatWrapper(llm=prometheus_llm)

    # build openai evaluator
    openai_evaluator = ChatOpenAI(
        model="gpt-4-1106-preview", temperature=0, max_tokens=512
    )

    # get evaluation prompt template
    correctness_prompt_template = build_eval_prompt()

    # run eval
    await run_full_eval(
        dataset=dataset,
        agents=agents,
        prometheus_evaluator=prometheus_evaluator,
        openai_evaluator=openai_evaluator,
        eval_prompt_template=correctness_prompt_template,
    )


if __name__ == "__main__":
    run_eval()
