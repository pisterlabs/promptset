import os
import openai

# docs: https://beta.openai.com/docs/api-reference/completions/create?lang=python
# OPENAI_API_KEY=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
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = '''
The following is a conversation between a Team Manager and a Lead engineer concerning the performance of another associate on the team, 
who the Lead engineer has complained about.

-------
Manager: So you wanted to talk to me?
Engineer: Yes, I have concerns about the new engineer that was hired last month, Jack. He does not seem to doing very well with the work \
he is being given
Manager: Do you have any examples?
Engineer: Mostly, it seems to me that he takes a very long time to complete relatively simple tasks, and his updates in the morning are always \
vague and unhelpful. I never really know what he is working on, it seems to me that he spends a lot of time doing non-work related things.
Manager: Well, he is still fairly new, and getting used to the technologies, so it makes sense that he would take longer to complete tasks \
than a more experienced engineer.
Engineer: He seems to be struggling more than others. 
Manager: I don't think it is any cause for concern, when you started you took a long time to complete work as well, and I remember you constantly \
asking questions to figure things out. I think his problem is more just lack of communication, I can talk to him and let him know he needs to \
comunicate better.
Engineer: I would appreciate that, thank you.
-------

tl;dr:
'''

completion = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  temperature=0.3,
  max_tokens=50
)

print(f"{completion['choices'][0].text}")

# best responses(temp=0.3):
# -----------------
# The manager is trying to help the engineer understand that the new engineer is struggling, and that he needs to be patient.
# The engineer is being a dick and is trying to get the new engineer fired.
####
# New people don't know what they don't know, so cut them slack
####
# The Lead engineer is not happy with the performance of the new engineer, and
####
# The engineer is not performing well, but the manager is not concerned because the engineer is new and still learning. The manager will talk to the engineer and let him know he needs to communicate better.
####
# The Lead engineer is not happy with the performance of the new engineer, and is complaining about him to the manager. The manager is trying to defend the new engineer, and is telling the Lead engineer that he is just new and needs to get used
