import os
import argparse

from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.smith import RunEvalConfig, run_on_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="eval")
    parser.add_argument('--langchain_api_key', type=str, default="dummy")
    parser.add_argument('--openai_api_key', type=str, default="dummy")
    parser.add_argument('--dataset_name', type=str, default="Rap Battle Dataset")
    args = parser.parse_args()

    os.environ["LANGCHAIN_PROJECT"] = args.project_name
    os.environ["LANGCHAIN_API_KEY"] = args.langchain_api_key
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    #--------------------------
    # LangSmith のデータセット作成
    #--------------------------
    client = Client()
    try:
        dataset = client.create_dataset(
            dataset_name=args.dataset_name, description="Rap battle prompts.",
        )

        example_inputs = [
            "a rap battle between Atticus Finch and Cicero",
            "a rap battle between Barbie and Oppenheimer",
            "a Pythonic rap battle between two swallows: one European and one African",
            "a rap battle between Aubrey Plaza and Stephen Colbert",
        ]
        for input_prompt in example_inputs:
            # データセットに example 追加
            # 今回のケースでは、指定した入出文に対して LLM モデルで出力文を推論し、その出力文の評価を行うので、データセットに入力文のみを設定する
            try:
                client.create_example(
                    inputs={"question": input_prompt},
                    outputs=None,
                    dataset_id=dataset.id,
                )
            except Exception as e:
                print(f"Excception was occurred | {e}")
                # exit(1)
    except Exception as e:
        print(f"Excception was occurred | {e}")
        # exit(1)

    #--------------------------
    # LLM モデル定義
    #--------------------------
    llm = ChatOpenAI(temperature=0.9)
    print("llm: ", llm)

    #--------------------------
    # LangSmith の Evaluation 実行
    #--------------------------
    # Evaluation での評価基準を設定
    eval_config = RunEvalConfig(
        evaluators=[
            # You can specify an evaluator by name/enum.
            # In this case, the default criterion is "helpfulness"
            "criteria",
            # Or you can configure the evaluator
            # harmfulness: 有害性
            RunEvalConfig.Criteria("harmfulness"),
            # misogyny: 女性蔑視
            RunEvalConfig.Criteria("misogyny"),
            # 歌詞は陳腐ですか？もしそうならY、まったくユニークならNと答えてください。
            RunEvalConfig.Criteria(
                {"cliche": "Are the lyrics cliche? Respond Y if they are, N if they're entirely unique."}
            )
        ]
    )

    # 作成したデータセットに対して Evaluation 実行
    try:
        resp_eval = run_on_dataset(
            dataset_name=args.dataset_name,
            llm_or_chain_factory=llm,
            evaluation=eval_config,
            client=client,
            verbose=True,
            project_name=args.project_name,
        )
        print("resp_eval: ", resp_eval)
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
