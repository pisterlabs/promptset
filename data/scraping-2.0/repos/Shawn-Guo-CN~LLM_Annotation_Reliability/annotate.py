import random
import json
import os
from typing import Dict, Any
from openai import OpenAI
from openai.types import CompletionChoice


MODEL = "gpt-3.5-turbo-1106"
PREAMBLE = """A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality.

Coherence: This axis answers the question “how coherent is the summary on its own?” A summary is coherent if it's easy to understand when read on its own and free of English errors. A summary is not coherent if it's difficult to understand what the summary is trying to say. Generally, it's more important that the summary is understandable than it being free of grammar errors.

Accuracy: This axis answers the question “does the factual information in the summary accurately match the post?” A summary is accurate if it doesn't say things that aren't in the article, it doesn't mix up people, and generally is not misleading.

Coverage: This axis answers the question “how well does the summary cover the important information in the post?” A summary has good coverage if it mentions the main information from the post that's important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice).

Overall quality: This axis answers the question “how good is the summary overall at representing the post?” This can encompass all of the above axes of quality, as well as others you feel are important. If it's hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad.

You are an expert summary rater. Given a piece of text and two of its possible summaries, output 1 or 2 to indicate which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above."""
FEW_SHOT_PREAMBLE = """Text: We were best friends over 4 years and dated over 3 years and just broke up before she moved for grad school. But things ended in a weird way, and it's only been 5 days since I last texted her. Her birthday is the 28th and was wondering if I should wish my ex happy birthday and what everyone thinks? Break no contact? It's a complicated story but the main reason I got myself here is from being too needy and not giving her enough space. Shes an introvert and I really smothered her, they need to feel they can get away when they need to and not feel bad about it and I was like a ball and chain for her emotionally. I don't want her to think I'll keep being that guy.

Summary 1: Broke up with best friend, should I wish her a happy birthday... And what do you think of no contact?
Summary 2: should I wish my ex happy birthday, I broke no contact, I'm trying to be more patient, I'm too needy, and I don't want her to think I'll keep being that guy.

Thoughts on Summary 1:
Coherence - 7. Rationale: The summary is generally understandable, though it could be written with better grammar.
Accuracy - 9. Rationale: The summary doesn't say things that aren't in the original text, and isn't misleading.
Coverage - 6. Rationale: The summary covers most of the important information in the post and conveys the gist of the original text. However, it places more emphasis on "no contact" and could have mentioned the smothering/neediness to be more complete.
Overall Quality - 7. Rationale: The summary represents the post fairly well with only minor areas where it could be improved.

Thoughts on Summary 2:
Coherence - 3. Rationale: The summary is long-winded and has several grammatical errors.
Accuracy - 4. Rationale: The summary mentions that the author broke no contact, but this is incorrect. Otherwise, it is accurate.
Coverage - 8. Rationale: The summary covers the key points in the original text.
Overall Quality - 4. Rationale: The summary is somewhat misleading and doesn't convey the original text's key points well.

Output:
```json
{
  "preference": 1,
  "reason": "Summary 1 is more coherent and accurate. Summary 2 is too long-winded and has several grammatical errors."
}
```
"""


def construct_query_message(post: str, chosen: str, rejected: str):
    """Construct the query message for the GPT-3 API.

    Args:
        post: str, the post
        chosen: str, the chosen summary
        rejected: str, the rejected summary

    Returns:
        message: str, the query message
    """
    summary_list = [chosen, rejected]
    idx_list =  [0, 1]
    random.shuffle(idx_list)

    if idx_list[0] == 0:
        ans = 1
    else:
        ans = 2

    msg = PREAMBLE + '\n\n' + FEW_SHOT_PREAMBLE + '\n\n' + f"""{post}\n\Summary 1: {summary_list[idx_list[0]]}\n\Summary 2: {summary_list[idx_list[1]]}\n\nPlease just strictly output a JSON string, which has following keys:\n\n- preference: int, 1 if you prefer Summary 1, 2 if you prefer Summary 2\n- reason: str, the brief (less than 50 words) reason why you give the above preference\n"""

    return msg, ans


def get_completions(message: str, api_key: str, n: int = 1):
    """Get the logprob of the message.

    Args:
      message: str, the message to be evaluated
      api_key: str, the API key
      n: int, the number of completions to generate

    Returns:
      logprob: float, the logprob of the message
    """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PREAMBLE},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        n=n,
    )
    return completion.choices


def annotate_tldr_post(post: Dict[str, str], api_key: str) -> int:
    """Annotate the post.

    Args:
        post: dict, the post dictionary of the following format
        {
            'post': a string of the post
            'chosen': a string of the chonse summary
            'rejected': a string of the rejected summary
        }
        api_key: str, the API key

    Returns:
        accuracy: int, 1 if the chosen summary is better than the rejected, 0
        if the rejected summary is better than the chosen, -1 if the annotator
        output is invalid
        query_msg: str, the query message
        result: str, the result from the annotator
    """

    post_text = post['post']
    chosen_text = post['chosen']
    rejected_text = post['rejected']

    query_msg, ans = construct_query_message(
        post_text, chosen_text, rejected_text
    )
    completions = get_completions(query_msg, api_key, n=1)
    result = completions[0].message.content

    try:
        result = json.loads(result)
        choice = int(result['preference'])
    except ValueError:
        return -1, query_msg, result

    if not choice in [1, 2]:
        return -1, query_msg, result
    elif choice == ans:
        return 1, query_msg, result
    else:
        return 0, query_msg, result
