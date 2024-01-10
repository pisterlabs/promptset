import openai

openai.Completion.create(
  engine="davinci",
  prompt="Make a list of astronomical observatories:"
)