import os
import openai

openai.api_key = "<api key here>"

response = openai.Completion.create(
  engine="davinci",
  prompt="The following is Javascript:\n\n###\n\nclass HelloMessage extends React.Component {\n  render() {\n    return (\n      <div>\n        Hello {this.props.name}\n      </div>\n    );\n  }\n}\n\nReactDOM.render(\n  <HelloMessage name=\"Taylor\" />,\n  document.getElementById('hello-example')\n);\n\n###\n\nSummarize the code in plain English:\n\n###",
  temperature=0.7,
  max_tokens=64,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["###"]
)

print(response)