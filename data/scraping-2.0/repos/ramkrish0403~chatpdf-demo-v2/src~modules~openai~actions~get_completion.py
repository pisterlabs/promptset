# from .get_client import client
from .get_client import openai as client

def get_completion(prompt: str, temperature: float = 0.5, max_tokens: int | None= None):
    # chat_completion = client.chat.completions.create(
    chat_completion = client.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo-16k",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    '''
    Sample:
    ChatCompletion(id='chatcmpl-8JmeSAx68aMuqHvzXVTjDOgqMGAqa', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='The termination policy states that in appropriate circumstances, management will provide verbal and written warnings, and if the conduct is not sufficiently altered, demotion, transfer, forced leave, or termination of employment may occur. Failure to adhere to certain policies such as the Nonsolicitation/Nondistribution Policy or non-returning of equipment and passwords may also result in termination.', role='assistant', function_call=None, tool_calls=None))], created=1699725940, model='gpt-3.5-turbo-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=72, prompt_tokens=380, total_tokens=452))
    '''
    response = chat_completion.choices[0].message.content
    return response

if __name__ == "__main__":
    prompt = '''
    Answer the question based on the context retrieved from the employee handbook.
    context: Again, any attempt at progressive discipline does not imply that your employment is anything other than on an "at- will" basis consistent with applicable law. Note that the specific terms of your employment relationship, including termination procedures, are governed by the laws of the state in which you are employed. In appropriate circumstances, management will first provide you with a verbal warning, then with one or more written warnings, and if the conduct is not sufficiently altered, eventual demotion, transfer, forced leave, or termination of employment. Any outside employment that will conflict with your duties and obligations to the Company should be reported to your Manager. Failure to adhere to this policy may result in discipline up to and including termination. 5.6  Pay Raises Depending on financial health and other Company factors, efforts will be made to give pay raises consistent with Zania, Inc. profitability, job performance, and the consumer price index. At the time of employment termination, all such equipment and passwords must be returned to the Company in operable condition. Violation of this policy may result in discipline, up to and including termination of employment. 6.4  Nonsolicitation/Nondistribution Policy To avoid disruption of business operations or disturbance of employees, visitors, and others, Zania, Inc. has implemented a Nonsolicitation/Nondistribution Policy. Nothing in this policy is intended to prevent employees from engaging in protected concerted activity under the NLRA. You will be subject to disciplinary action up to and including termination of employment for violation of this policy. 6.8  Personal Data Changes It is your obligation to provide Zania, Inc. with your current contact information, including current mailing address and telephone number.
    question: What is the termination policy?
    answer:
    '''
    completion = get_completion(prompt)
    print(completion)

    # Usage:
    # python -m src.modules.openai.actions.get_completion