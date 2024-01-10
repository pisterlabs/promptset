# This python script contains utility functions for
# generating prompt, generating input and computing similarity

import openai
import json
import pandas as pd
from pathlib import Path
from openai.embeddings_utils import get_embedding, cosine_similarity
from typing import Union, Dict, List
import time
import logging
import os
from dotenv import load_dotenv
from datetime import datetime


# ---
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_CHAT_MODEL = "ChatGPT"
DEFAULT_SIM_THRESHOLD = 0.7
DEFAULT_SIM_TOPK = 3
DEFAULT_COLS_FOR_ASSESSMENT = [
    "ip_id",
    "ip_title",
    "github_url",
    "toolkit_url",
    "short_explanation",
    "explanation",
]

OAI_STATUS_SUCCESS = "Success"
OAI_STATUS_FAIL = "Fail"

VALUE_MAPPING = {"high": 100, "moderate": 50, "low": 10}


class LLMUtils:
    def __init__(
        self,
        document_parquet_path: Union[Path, str],
        llm_config_path: Union[Path, str],
        raw_doc_path: Union[Path, str],
    ):
        """
        initiate the LLMUtils class
        - load pre-computed document index
        - load prompt template

        Args:
            document_parquet_path (Union[Path, str]): path to the stored
            pre-computed document embedding
            llm_config_path (Union[Path, str]): path to the stored prompt and
            model id
            raw_doc_path (Union[Path, str]): Path to raw ip doc
            logging_level (logging.level): logging level
        """
        self._config = json.load(open(llm_config_path, "r"))
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(self._config.get("LOGGING_LEVEL", "INFO"))
        # load raw document content
        self._raw_doc_df = pd.DataFrame(json.load(open(raw_doc_path, "r"))).T[
            DEFAULT_COLS_FOR_ASSESSMENT
        ]
        # load document index
        if not Path(document_parquet_path).exists():
            self.generate_doc_embedding(document_parquet_path)
        self._document_df = pd.read_parquet(document_parquet_path)
        self._sim_threshold = self._config.get(
            "DEFAULT_SIM_THRESHOLD", DEFAULT_SIM_THRESHOLD
        )
        self._topk = self._config.get("DEFAULT_SIM_TOPK", DEFAULT_SIM_TOPK)

    def ideation(self, user_input: str, usecase_number: int = 5) -> Dict:
        """
        generate brainstorming use case based on users' input,
        return json format results

        Args:
            user_input (str):  user input business problem
            usecase_number(int): number of use case we "expect" from openAI

        Returns:
            Dict: expect gpt return a json format object in the following format
            {
                "status": status_code (Success or Fail),
                "results": {
                "use_case_1":{
                    "use_case_name": xxx,
                    "explanations": xxx,
                    "business_value": xxx,
                    "feasibility": xxx
                }
            }
            }
        """
        status_code = OAI_STATUS_SUCCESS
        ideation_prompt = self._config.get("IDEATION_PROMPT")
        ideation_prompt = ideation_prompt.replace(
            "{{USE_CASE_NUM}}", str(usecase_number)
        )
        user_prompt = f"{ideation_prompt}```{user_input}```"
        tmp_conv = [
            self._config.get("SYSTEM_DEFAULT_MSG"),
            {"role": "user", "content": user_prompt},
        ]
        current_response = openai.ChatCompletion.create(
            deployment_id=self._config.get("CHAT_MODEL", DEFAULT_CHAT_MODEL),
            messages=tmp_conv,
            temperature=self._config.get("TEMP_IDEATION", 0),
            max_tokens=4000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        results = []
        try:
            if current_response.choices[0].finish_reason == "stop":
                result_dict = json.loads(current_response.choices[0].message.content)
                for idea_id, idea_dict in result_dict.items():
                    idea_dict.update({"use_case_id": idea_id})
                    biz_value = idea_dict.get("business_value").lower()
                    idea_dict.update(
                        {"business_value_num": VALUE_MAPPING.get(biz_value, 20)}
                    )
                    feasibility_value = idea_dict.get("feasibility").lower()
                    idea_dict.update(
                        {"feasibility_num": VALUE_MAPPING.get(feasibility_value, 20)}
                    )
                    overall_score = (
                        idea_dict["business_value_num"] + idea_dict["feasibility_num"]
                    ) * 0.5 - 1
                    idea_dict.update({"overall_score": overall_score})
                    results.append(idea_dict)
                results.sort(
                    key=lambda x: (x["business_value_num"], x["feasibility_num"]),
                    reverse=True,
                )
            else:
                status_code = OAI_STATUS_FAIL
                self._logger.error(
                    f"Response generation failed. finish code {current_response.choices[0].finish_reason}"
                )
        except Exception as ex:
            # Not a good practice to catch all exceptions, but I cannot predict
            # what exceptions might happen with OpenAI
            status_code = OAI_STATUS_FAIL
            self._logger.error(
                f"There is an error occurred {ex}.\n\n{current_response.choices[0].message.content}"
            )
        return {"status": status_code, "results": results}

    def recommend_ips(self, original_input: str, generated_idea: Dict) -> Dict:
        """
        find relevant ips in the DStoolkit and recommend to user.
        Args:
            original_input (str): the original input usecase by user
            generated_idea (Dict): the generated ideas. the following two fields
                                must exist
            {
                "use_case_name": xxx,
                "explanations": xxx
            }

        Returns:
            Dict: {
                "status": status_code,
                "results":  [
                    {
          "ip_id": xxx,
          "usage": "xxxx",
          "url": "xxxxxx",
          "ip_title": "xxx"
                }
                ]
            }
        """
        if (
            "use_case_name" not in generated_idea
            or "explanations" not in generated_idea
        ):
            self._logger.error(
                "The generated idea does not have use_case_name or explanations, unable to recommend ip"
            )
            return {"status": OAI_STATUS_FAIL, "results": None}

        ip_candidate_list = self._find_top_k_similar_ip(generated_idea)
        # TODO: remove this when we don't have throttling
        self._logger.debug(
            f"Before assess, the length of initial candidate ip list is {len(ip_candidate_list)}"
        )
        assessment_results = self._assess_ip_relevance(
            original_input, generated_idea, ip_candidate_list
        )
        recommended_ip = []
        for ip_obj in assessment_results.get("results"):
            # get original ip id
            ip_id = int(ip_obj.get("ip_name").split("_")[-1].strip())
            if ip_obj.get("relevance", "not relevant").lower() != "not relevant":
                meta_data = self._raw_doc_df[self._raw_doc_df.ip_id == ip_id][
                    ["github_url", "ip_title", "toolkit_url"]
                ].to_dict("list")
                recommended_ip.append(
                    {
                        "ip_id": ip_id,
                        "usage": ip_obj.get("usage", ""),
                        "github_url": meta_data["github_url"][0],
                        "ip_title": meta_data["ip_title"][0],
                        "toolkit_url": meta_data["toolkit_url"][0],
                    }
                )
        self._logger.debug(
            f"After assess the relevance the number of relevant ip is {len(recommended_ip)}"
        )
        return recommended_ip

    def _assess_ip_relevance(
        self, original_user_input: str, generated_idea: Dict, similar_ip_list: List
    ) -> Dict:
        """
        assess the top ranked ip results usin GPT chat mode.
        return a list of results

        example return format:
        ```
        [{"use_case_name": "Connection Extraction Model", "ip_name": "Energy Profiling & Prediction_8", "relevance": "not relevant", "usage": "na"}]
        ```

        Args:
            original_user_input (str): original user input
            generated_idea (Dict): ideation generated in the first round of conversation
            similar_ip_list (List): list of top-ranked similar ip

        Returns:
            {
                "status": status_code,
                "results": list of ip assessment results
            }
        """
        results = []
        status = OAI_STATUS_SUCCESS
        if len(similar_ip_list) == 0:  # no ip to validate
            return results
        assessment_prompt = self._get_doc_assessment_prompt(
            original_user_input, generated_idea, similar_ip_list
        )
        assessment_conv = [
            self._config.get("SYSTEM_DEFAULT_MSG"),
            assessment_prompt,
        ]
        current_response = openai.ChatCompletion.create(
            deployment_id=self._config.get("CHAT_MODEL", DEFAULT_CHAT_MODEL),
            messages=assessment_conv,
            temperature=self._config.get("TEMP_ASSESSMENT", 1),
            max_tokens=4000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        results = []
        if current_response.choices[0].finish_reason == "stop":
            # post_processing response, just in case exception format happens
            response_text = current_response.choices[0].message.content
            start = response_text.find("[")
            end = response_text.find("]")
            try:
                results = json.loads(response_text[start : end + 1])
            except Exception as ex:
                # Not a good practice to catch all exceptions, but I cannot predict
                # what exceptions might happen with OpenAI
                self._logger.error(
                    f"Encounter a problem {ex}\n\n The response is: {response_text[start: end+1]}"
                )
                status = OAI_STATUS_FAIL
        else:
            status = OAI_STATUS_FAIL
            self._logger.error(
                f"Response generation failed. finish code {current_response.choices[0].finish_reason}"
            )
        return {"status": status, "results": results}

    def _get_doc_assessment_prompt(
        self, original_user_input: str, generated_idea: Dict, similar_ip_list: List
    ) -> Dict:
        """
        generate prompt for assessing the relevance of the generated use case
        and the input document

        The prompt is something like this:

        I will provide you a business context, use case and a list of ip
        delimited by triple backticks. For each ip, assess if the it is relevant
        to the use case under the business context. If the ip is not relevant,
        response with 'not relevant' and usage as 'na'. If the it is  relevant,
        response 'relevant' and provide how you will use  this ip.\n Your
        response should only be JSON objects in the following format:
        [{'use_case_name': ... , 'ip_name': ...., 'relevance': ...,
        'usage':....}, {'use_case_name': ... , 'ip_name': ...., 'relevance':
        ..., 'usage':....}]
        ```business context: .....```
        ```use case: ....```
        List of ips are: ```....```

        Args:
            original_user_input (str): original user input. We need this as a context
            generated_idea (Dict): the brainstorming idea and explanation
            similar_ip_list (List): topk similar ip list

        Returns:
            Dict:  {"role": "user", "content": prompt}
        """
        assessment_prompt = self._config.get("ASSESSMENT_PROMPT_PREFIX")
        ip_content = ""
        curr_ip_df = self._raw_doc_df[
            self._raw_doc_df.ip_id.isin(similar_ip_list)
        ].to_dict("list")
        for i in range(len(similar_ip_list)):
            ip_content += (
                "ip: "
                + f"{curr_ip_df['ip_title'][i]}_{curr_ip_df['ip_id'][i]}"
                + ".\nDescription: "
                + curr_ip_df["explanation"][i]
                + "\n"
            )
        prompt = (
            assessment_prompt
            + f"```business context: {original_user_input}```\n```use case: {generated_idea['use_case_name']}\nDescription: {generated_idea['explanations']}```\n"
            + f"List of ips are: ```{ip_content}```\n"
        )
        return {"role": "user", "content": prompt}

    def _find_top_k_similar_ip(self, idea: Dict) -> List:
        """
        based on the generated query embedding, find top-k similar
        ip document according to cosine similarity
        We will consider threshold first and then get topk.

        Args:
            curr_embedding (List): current embedding for generated use case
            topk (int, optional): cutoff for rank list. Defaults to 3.
            threshold (float, optional): similarity threshold. Defaults to 0.8.

        Returns:
            List: List of similar ip that may be recommended
        """
        dist_scores = []
        full_input = idea["use_case_name"] + ". " + idea["explanations"]
        qry_embedding = get_embedding(
            full_input,
            model=self._config.get("EMBEDDING_MODEL_QUERY", DEFAULT_EMBEDDING_MODEL),
        )

        for _, doc_row in self._document_df.iterrows():
            ip_id = doc_row.ip_id
            similarity_score = cosine_similarity(qry_embedding, doc_row.embedding)
            if similarity_score >= self._sim_threshold:
                dist_scores.append({"ip_id": ip_id, "sim_score": similarity_score})
        dist_scores = sorted(dist_scores, key=lambda x: x["sim_score"], reverse=True)
        doc_list = (
            []
            if len(dist_scores) == 0
            else [x["ip_id"] for x in dist_scores[: self._topk]]
        )
        return doc_list

    def generate_doc_embedding(
        self,
        embedding_parquet_path: Union[str, Path],
        embedding_fields=["topic", "explanation"],
    ):
        """

        Generate document embedding and output to a parquet file

        Args:
            embedding_parquet_path (Union[str, Path]): output embedding results
            as parquet file
        """
        results = []
        for (ip_id, ip_title), curr_ip_topics in self._raw_doc_df.groupby(
            ["ip_id", "ip_title"]
        ):
            curr_ip_dict = curr_ip_topics.to_dict("list")
            current_content = ip_title + ". "
            current_content += curr_ip_dict["top_keywords"][0]
            for field_str in embedding_fields:
                current_content += curr_ip_dict[field_str][0] + ". "
            curr_embedding = get_embedding(
                current_content,
                model_id=self._config.get(
                    "EMBEDDING_MODEL_DOC", DEFAULT_EMBEDDING_MODEL
                ),
            )
            results.append(
                {
                    "ip_id": ip_id,
                    "content": current_content,
                    "embedding": curr_embedding,
                }
            )
        pd.DataFrame(results).to_parquet(embedding_parquet_path, index=False)


# ---uncomment the main function if you want to try the class without ui
# if __name__ == "__main__":
#     #
#     import json

#     logger = logging.getLogger(__name__)
#     llm_utils = LLMUtils(
#         document_parquet_path=Path("assets/ip_description_topic_embedding.parquet"),
#         llm_config_path=Path("assets/llm_config.json"),
#         raw_doc_path=Path("assets/ip_description.json"),
#     )
#     test_case_dict = json.load(open("assets/test_case.json", "r"))
#     input_str = test_case_dict[6].get("input")
#     brainstorming_idea = llm_utils.ideation(input_str, usecase_number=5)
#     time.sleep(60)
#     logger.debug(json.dumps(brainstorming_idea, indent=2))
#     if brainstorming_idea.get("status", OAI_STATUS_FAIL) == OAI_STATUS_SUCCESS:
#         for idea in brainstorming_idea.get("results"):
#             recommended_ip = llm_utils.recommend_ips(input_str, idea)
#             logger.debug(json.dumps(recommended_ip, indent=2))
#             time.sleep(60)
