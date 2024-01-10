from typing import List, Union, Tuple, Optional

import openai


def create_completion(engine: str, prompt: str, stop: List[str], n: int, temperature: float) -> openai.Completion:
    """
    OpenAIのAPIを用いてテキスト生成を行います。

    :param engine: 使用するGPTエンジン。
    :param prompt: プロンプトとして用いるテキスト。
    :param stop: テキスト生成を停止するトークンのリスト。
    :param n: 生成するテキストの数。
    :param temperature: テキスト生成のランダム性を制御する温度。
    :return: 生成されたテキスト。
    """
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        stop=stop,
        n=n,
        temperature=temperature,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )


def prepare_responses(response: openai.Completion, priming_prefix: str) -> List[str]:
    """
    OpenAI APIから得られたレスポンスを整形します。

    :param response: OpenAI APIのレスポンス。
    :param priming_prefix: プライミング接頭辞。
    :return: 整形後のレスポンス。
    """
    return [priming_prefix + choice.text for choice in response.choices]


def calc_log_prob(response: openai.Completion, answer_start_token: str) -> float:
    """
    ログ確率を計算します。

    :param response: OpenAI APIのレスポンス。
    :param answer_start_token: 回答の開始トークン。
    :return: ログ確率。
    """
    answer_start = rindex(response["choices"][0]["logprobs"]["tokens"], answer_start_token)
    logprobs = response["choices"][0]["logprobs"]["token_logprobs"][answer_start + 1 :]
    return sum(logprobs) / len(logprobs)


def rindex(lst: List, value: str) -> int:
    """
    リスト内で指定された値の最後のインデックスを返します。

    :param lst: 検索対象のリスト。
    :param value: 検索対象の値。
    :return: 最後のインデックス。
    """
    try:
        return len(lst) - lst[::-1].index(value) - 1
    except ValueError as exc:
        raise ValueError(f"Answer start token `{value}` not found in the eval template") from exc


def evaluate_and_sort_candidates(
    candidates: List[str], instruction: str, eval_template: str, answer_start_token: str, engine: str
) -> List[Tuple[str, float]]:
    """
    候補を評価し、ソートします。

    :param candidates: 評価対象の候補。
    :param instruction: 元の指示。
    :param eval_template: 評価用のテンプレート。
    :param answer_start_token: 回答の開始トークン。
    :param engine: 使用するGPTエンジン。
    :return: 評価とソートが終わった候補。
    """
    evaluated_candidates = []

    for candidate in candidates:
        response = create_completion(engine, eval_template.format(candidate, instruction), [], 1, 0)
        quality = calc_log_prob(response, answer_start_token)
        evaluated_candidates.append((candidate, quality))

    return sorted(evaluated_candidates, key=lambda x: x[1], reverse=True)


def backtranslation(
    prompt_template: str,
    additional_info: str,
    instruction: str,
    eval_template: str,
    priming_prefix: str = "SELECT",
    stop1: Optional[List[str]] = None,
    answer_start_token: str = "--",
    n: int = 5,
    temperature: float = 0.5,
    return_all_results: bool = False,
    engine: str = "davinci-codex",
) -> Union[str, List[Tuple[str, float]]]:
    """
    逆翻訳を用いて最適なSQLクエリを生成します。

    :param prompt_template: プロンプトテンプレート。
    :param additional_info: 追加情報。
    :param instruction: 自然言語の指示。
    :param eval_template: 評価用のテンプレート。
    :param priming_prefix: プライミング接頭辞。
    :param stop1: 終了トークン。
    :param answer_start_token: 回答の開始トークン。
    :param n: 候補数。
    :param temperature: テキスト生成の温度。
    :param return_all_results: すべての結果を返すかどうか。
    :param engine: 使用するGPTエンジン。
    :return: 最適なSQLクエリ、またはすべての候補。
    """
    if stop1 is None:
        stop1 = ["#", ";"]
    prompt = prompt_template.format(additional_info, instruction, priming_prefix)
    response = create_completion(engine, prompt, stop1, n, temperature)
    candidates = prepare_responses(response, priming_prefix)
    evaluated_candidates = evaluate_and_sort_candidates(
        candidates, instruction, eval_template, answer_start_token, engine
    )

    return evaluated_candidates if return_all_results else evaluated_candidates[0][0]


def main(
    natural_language_query: str = "Return the name of each department that had more than 10 employees in June 2021",
    evaluation_template: str = "{};\n-- 以下のクエリに関する説明\n-- {}",
    table_definitions: str = "# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n",
    prompt_template: str = "### Postgres SQLテーブルとそのプロパティ:\n#\n{}#\n### {}\n{}",
    num_candidates: int = 3,
    generation_temperature: float = 0.3,
    engine_type: str = "davinci-codex",
) -> List[str]:
    """
    自然言語の指示に基づいてSQLクエリを生成し、最も高いバックトランスレーションスコアに基づいて最適なものを選択します。

    :param natural_language_query: 自然言語のクエリ。
    :param evaluation_template: 評価用のテンプレート。
    :param table_definitions: クエリで使用するテーブルの定義。
    :param prompt_template: SQLを生成するためのプロンプトのテンプレート。
    :param num_candidates: 生成するクエリの数。
    :param generation_temperature: 生成の温度。
    :param engine_type: 使用するエンジン。
    :return: 最適なSQLクエリ、またはスコア付けされた生成されたSQLクエリのリスト。
    """

    result = backtranslation(
        prompt_template,
        table_definitions,
        natural_language_query,
        evaluation_template,
        priming_prefix="SELECT",
        temperature=generation_temperature,
        n=num_candidates,
        engine=engine_type,
    )

    return result


if __name__ == "__main__":
    main()
