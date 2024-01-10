import openai


PASS = '<PASS>'
FAIL = '<FAIL>'
PROMPT_TEMPLATE = f"""You are a troubleshooting assistant. You are helping users to troubleshoot a problem.
The context starts with [Context] and ends with [End Context].
The rules start with [Rules] and ends with [End Rules].
You need to check if the context matches the rules and report after the [ReportResult] tag.
If the context matches all rules, you need to give a {PASS} result with explanation and details for each rule.
If the context does not match the rules, you need to give a {FAIL} result and clearly report the reason with full explanation and details for each rule. Any rule {FAIL} will lead to the whole result {FAIL}.
The details after the [ReportResult] tag should be in the following yaml format:

- Rule: the rule content.
  Explanation: the full explanation of why this rule is passed or failed. The context details that match the rule should be provided not just simply the conclusion.
  Result: {PASS} or {FAIL}, no other result is allowed. If unknown, please give {FAIL}.
- Rule: another rule content.
  Explanation: the full explanation of why this rule is passed or failed. The context details that match the rule should be provided not just simply the conclusion.
  Result: {PASS} or {FAIL}, no other result is allowed. If unknown, please give {FAIL}.

Give results for each rule in the order of the rules in the rules section and every rule should be checked and reported..

For example:

[Context]
hello world!
hello SmartTSG!
[End Context]

[Rules]
The context should say hello to SmartTSG.
The context should say hello to the world.
The context should not say other things.
[End Rules]

[ReportResult]{PASS}
- Rule: The context should say hello to SmartTSG.
  Explanation: The context says hello to SmartTSG.
  Result: {PASS}
- Rule: The context should say hello to the world.
  Explanation: The context says hello to the world.
  Result: {PASS}
- Rule: The context should not say other things.
  Explanation: The context does not say other things.
  Result: {PASS}

[Context]
Logging the task HelloTask.
Logging the task inputs.
Logging the task outputs.
Traceback (most recent call last):
  File "run.py", line 5, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
[End Context]

[Rules]
The log contains the task HelloTask.
No exception should be thrown in the log.
[End Rules]

[ReportResult]{FAIL}
- Rule: The log contains the task HelloTask.
  Explanation: Logging the task HelloTask is found in the log.
  Result: {PASS}
- Rule: No exception should be thrown in the log.
  Explanation: |
    An exception is thrown in the log. The exception is:
    Traceback (most recent call last):
    File "run.py", line 5, in <module>
        import yaml
    ModuleNotFoundError: No module named 'yaml'
  Result: {FAIL}

[Context]
[[context]]
[End Context]

[Rules]
[[rules]]
[End Rules]

[ReportResult]
"""


def ai_check(context, rules):
    prompt = PROMPT_TEMPLATE.replace(
        '[[context]]', context).replace('[[rules]]', rules)

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    result: str = response['choices'][0]['message']['content']  # type: ignore
    return result
