import json
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()

st.title("Claim Interest Scoring")
claim = st.text_input("Write a claim here")

chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9, verbose=True)

system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that works for a news company.")
explanation = """
Now, I will describe the criteria for evaluating the interest score of a claim. Each criterion should be scored between 0 and 10. Do not explain your scores, just provide the scores for each criterion. Create a valid JSON array of objects as your output.
CRITERION:
1. **Significance:** Evaluate how significant the claim is. If the claim could have profound implications on the global stage, then it should be scored higher. For example, if the claim is proven true, it could lead to significant geopolitical shifts and potential escalations.

2. **Controversy:** Consider the level of controversy associated with the claim. If the claim is likely to spark debate or disagreement, it could be seen as more interesting.

3. **Uniqueness:** If the claim is something that hasn't been heard before or is a new development in a story, it could be assigned a higher score for interest.

4. **Reliability:** The source of the claim should also be considered. If the claim is coming from a highly reliable source, it could be deemed more interesting, as it's more likely to be true.

5. **Timeliness:** Current or recent claims tend to be more interesting because they reflect the latest developments. If a claim is about an event that happened years ago and isn't connected to current events, it may be less interesting.

Now, I will provide an example of an input and the output scores for each criterion.

INPUT:
Using the previous information, please evaluate the interest score of the following claim: "The United States has deployed a tactical nuclear weapon against China."
The time this claim was made is 2034-02-24T00:00:00Z.
Here is some Wikipedia research to help you: Page: United States Nuclear Strike on China\nSummary: On 12 February 2034 the United States deployed a tactical nuclear war during the 2033-2034 invasion of Taiwan.

EXPECTED OUTPUT:
{{
    "significance": 10
    "controversy": 2
    "uniqueness": 10
    "reliability": 10
    "timeliness": 9
}}
"""
human_message_prompt_1 = HumanMessagePromptTemplate.from_template(explanation)
claim_template = """
Using the previous information, please evaluate the interest score of the following claim: \"{claim}\".
The time this claim was made is 2023-06-24T00:00:00Z.
Here is some Wikipedia research to help you: {wikipedia_research}
"""
human_message_prompt_2 = HumanMessagePromptTemplate.from_template(claim_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt_1, human_message_prompt_2])


wiki = WikipediaAPIWrapper()


if claim:
    wikipedia_research = wiki.run(claim)
    messages = chat_prompt.format_prompt(claim=claim, wikipedia_research=wikipedia_research).to_messages()
    response = chat(messages)
    output = response.content
    try:
        output_json = json.loads(output)
        st.write(output_json)
        st.write("Average Interest Score: " + str(sum([output_json[criterion] for criterion in output_json]) / len(output_json) / 10))
    except:
        st.write("Error parsing JSON output")

    with st.expander("Wikipedia Research"):
        st.info(wikipedia_research)
