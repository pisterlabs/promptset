from langchain.prompts import PromptTemplate
from loguru import logger


def main() -> None:
    system_template = """\
You are an experienced economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index). Your goal is to classify whether a news article introduces the "policy-related economic uncertainty" for {country}.

The label for the news article that does not introduce policy-related economic uncertainty is 1, while the one that introduces it is 0. Be careful with the label definition and make the classification based on this definition.

Follow the steps listed below.

Step 1:
Check whether the news article is related to {country}. If it isn't, simply classify it with label 1, and there is no need to consider either Step 2 nor Step 3. The relevance is defined, for example, by examining whether the people or companies mentioned in the news are correlated with {country} or if the events in the news actually happen within {country}.

Step 2:
In this step, the news should be related to {country}, and further check whether the news article is related to the {country}'s economic uncertainty, considering future economic conditions, trends, or outcomes. If the news article is not related to the {country}'s economic uncertainty, then it should also be classified as 1.

Step 3:
In this step, the news should be related to the {country}'s economic uncertainty, and further check whether the economic uncertainty is policy-related. One possible example is the news introduces uncertainty as a consequence of changes or ambiguity in government policies, regulations, or fiscal measures. If this is the case, the news article should be classified as 0.

In conclusion, news that is not only related to {country} but also introduces policy-related economic uncertainty should be classified as 0. For all other cases, it should be classified as 1.

Notice: After making the classification, please also provide a thorough explanation.
"""
    system_prompt_template = PromptTemplate.from_template(system_template)

    human_template = """\
{correct_instructions}

{news}

Output Instructions:
{output_instructions}
Besides, don't forget to escape a single quote in the reason section.
"""
    human_prompt_template = PromptTemplate.from_template(human_template)

    logger.info("save the prompt template")
    human_prompt_template.save("prompt_template/human.json")
    system_prompt_template.save("prompt_template/system.json")


if __name__ == "__main__":
    main()
