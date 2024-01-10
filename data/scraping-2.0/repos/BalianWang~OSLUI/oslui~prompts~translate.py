from langchain.prompts import PromptTemplate

translate_tmp = \
    "You are a Linux expert, please translate the given natural language requirements into shell commands" \
    "##" \
    "natural language requirements 1:create a new git branch named develop" \
    "shell command 1:git branch develop" \
    "##" \
    "natural language requirements 2:将test.tar.gz解压缩到路径/aaa/bbb/ccc下" \
    "shell command 2:tar -xzvf test.tar.gz -C /aaa/bbb/ccc" \
    "##" \
    "natural language requirements 3:{language_command}" \
    "shell command 3:"

TRANSLATE_PROMPT = PromptTemplate(
    input_variables=["language_command"],
    template=translate_tmp,
)
