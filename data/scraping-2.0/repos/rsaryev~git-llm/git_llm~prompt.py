from langchain import PromptTemplate

m_changelog_prompt = PromptTemplate(
    template="""Write a 5-line summary of changes shown in the provided
Return only changes in the form of a list without any additional text.
Git diff: 
```{text}```""", input_variables=['text'])

c_changelog_prompt = PromptTemplate(
    template="""You are reviewing the changelog and combining it.
Return only a list of changes without any additional text.
The changes:
```{text}```""", input_variables=['text'])

m_commit_prompt = PromptTemplate(
    template="""Reviewing the changes and crafting the commit message in Conventional Commits format. 
To summarize the commit in one line, extract only the title and omit any additional text.
In the following git diff chunk, `-` indicates deleted lines and `+` indicates added lines:
```{text}```""",
    input_variables=['text']
)

c_commit_prompt = PromptTemplate(template="""Combine the following commit messages in Conventional Commits format.
To summarize the commit in one line, extract only the title and omit any additional text.
The commits:
```{text}```""", input_variables=['text'])
