from statistics import mean

from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

from juan.AutomaticPromptEngineering import config


class Evaluator:

    def __init__(self, model_name, temperature):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def get_evaluation(self, prompt, params):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt)
        )
        with get_openai_callback() as cb:
            result = llm_chain(params)
            costs = cb

        return result, {'total_cost': costs.total_cost}

    @staticmethod
    def print_evaluation(metrics_dict, answers):
        agg = {}
        for key, value in metrics_dict.items():
            agg[key] = mean(value)

        for metric_name, average in agg.items():
            print(f"🎯{metric_name}: {average*100}%")

        print()
        for key, values in metrics_dict.items():
            for i in range(0, len(answers)):
                expected_value = 1 if key in config.positive else 0
                if values[i] != expected_value:
                    print(f"❌Failed {key} for question `{answers[i]['question']}`")
                    print(f"- 📜 {answers[i]['knowledge']}")
                    print(f"- 🤔 {answers[i]['answer']}")

