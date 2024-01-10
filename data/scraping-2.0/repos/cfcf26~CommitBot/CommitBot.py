import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


st.title("Commit Bot")

st.info("""### How to use:
1. Enter openai_key.\n
2. select Commit Convention.\n
3. enter git diff. (git diff --cached | pbcopy after git add will copy it.)\n
4. Press the Submit button.
If your git diff is too large, it won't work.\n
Make sure to keep it small. It is recommended to use one file or one hader + one related file.\n
Only available in English. Other language will be updated later.\n
If you have any questions or suggestions for updates, please let us know here.(ccf2994@gmail.com) \n
""")

api_key = st.sidebar.text_input('OpenAI API Key', type='password')

options = ["Conventional Commits", "Angular Commit Guidelines", "Gitmoji Commit Guidelines", "Semantic Commit Messages", "Karma Runner Commit Msgs", "Atom Editor Commit Messages", "Custom Commit Convention"]
selected_option = st.sidebar.radio("원하는 선택지를 선택하세요:", options, index=0)

if selected_option == "Conventional Commits":
    selected_option = """Conventional Commits
    Format of the Commit Message:
    The commit message should be structured as follows:
    ```
    <type>[optional scope]: <description>

    [optional body]

    [optional footer]
    ```
    Type:
    This specifies the type of commit. Common types are:

    feat: Introduces a new feature to the codebase.
    fix: Fixes a bug.
    chore: Routine tasks or maintenance.
    docs: Documentation changes.
    style: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
    refactor: Code changes that neither fix a bug nor add a feature.
    perf: Performance improvements.
    test: Adding missing tests or correcting existing ones.
    Scope (Optional):
    This specifies the part of the codebase the commit pertains to, e.g., auth, ui, api, etc. The scope is enclosed in parentheses.

    Description:
    A brief description of the commit. It should:

    Use the imperative mood: "add" not "added", "fix" not "fixed".
    Not end with a period.
    Body (Optional):
    Provides a more detailed explanation of the commit. This is separated from the description by an empty line.

    Footer (Optional):
    This is where you'd reference any issues related to the commit, e.g., "Closes #123", "Related #456".

    Breaking Changes:
    If a commit introduces changes that break backward compatibility, it should have BREAKING CHANGE: in its footer or body. This is then followed by a description of what changed and its implications."""
elif selected_option == "Angular Commit Guidelines":
    selected_option = """Angular Commit Guidelines
    Format of the Commit Message:
    ```
    <type>(<scope>): <short summary>
    ```
    <body>

    <footer>
    ```
    Type:
    Must be one of the following:

    feat: A new feature.
    fix: A bug fix.
    docs: Documentation-only changes.
    style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
    refactor: A code change that neither fixes a bug nor adds a feature.
    perf: A code change that improves performance.
    test: Adding missing or correcting existing tests.
    chore: Changes to the build process or auxiliary tools and libraries such as documentation generation.
    Scope (Optional):
    Refers to the module or part of the codebase affected by the change, e.g., router, http, cli, etc. This provides additional contextual information.

    Short Summary:

    Provides a concise description of the changes.
    Uses imperative mood: "add" not "added", "change" not "changed".
    Doesn't end with a period.
    Body (Optional):

    Elaborates on the details of the commit.
    Explains the motivation for the change, contrasting its current behavior with the new behavior.
    Uses imperative mood: "add" not "added", "change" not "changed".
    Footer (Optional):

    Contains any information about Breaking Changes and is also the place to reference GitHub issues that this commit Closes.
    Breaking Changes should start with the word BREAKING CHANGE: with a space or two newlines. The rest of the footer should then describe what broke and the migration path, if possible."""
elif selected_option == "Gitmoji Commit Guidelines":
    selected_option = """Gitmoji Commit Guidelines
    Format of the Commit Message:
    ```
    :emoji: Short description of the commit
    ```
    Emoji Selection:
    The emoji at the beginning of the commit message should reflect the primary purpose or the main change made in the commit. Some commonly used emojis include:

    ✨ (:sparkles:) for introducing a new feature.
    🐛 (:bug:) for fixing a bug.
    📚 (:books:) for documentation changes.
    🎨 (:art:) to improve the structure or format of the code.
    ♻️ (:recycle:) for refactoring code.
    🚀 (:rocket:) to improve performance.
    🔒 (:lock:) for addressing security concerns.
    ... and many more.
    The Gitmoji guide provides a comprehensive list of emojis and their associated meanings.

    Short Description:
    A concise and clear description of the change made. Ideally, it should:

    Use the imperative mood, e.g., "add" not "added", "fix" not "fixed".
    Provide context for what the commit does, especially if the emoji alone is not self-explanatory."""
elif selected_option == "Semantic Commit Messages":
    selected_option = """Semantic Commit Messages
    Format of the Commit Message:
    ```
    <type>: <description>
    ```
    Type:
    Must be one of the following:

    feat: A new feature.
    fix: A bug fix.
    docs: Documentation-only changes.
    style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
    refactor: A code change that neither fixes a bug nor adds a feature.
    perf: A code change that improves performance.
    test: Adding missing or correcting existing tests.
    chore: Changes to the build process or auxiliary tools and libraries such as documentation generation.
    Scope (Optional):
    Refers to the module or part of the codebase affected by the change, e.g., router, http, cli, etc. This provides additional contextual information.

    Short Summary:

    Provides a concise description of the changes.
    Uses imperative mood: "add" not "added", "change" not "changed".
    Doesn't end with a period.
    Body (Optional):

    Elaborates on the details of the commit.
    Explains the motivation for the change, contrasting its current behavior with the new behavior.
    Uses imperative mood: "add" not "added", "change" not "changed".
    Footer (Optional):

    Contains any information about Breaking Changes and is also the place to reference GitHub issues that this commit Closes.
    Breaking Changes should start with the word BREAKING CHANGE: with a space or two newlines. The rest of the footer should then describe what broke and the migration path, if possible.
    """
elif selected_option == "Karma Runner Commit Msgs":
    selected_option = """Karma Runner Commit Msgs
    Format of the Commit Message:
    ```
    <type>(<scope>): <short description>
    <BLANK LINE>
    <body>
    <BLANK LINE>
    <footer>
    ```
    Type:
    A brief description of the nature of the change. Some of the common types include:

    feat: A new feature.
    fix: A bug fix.
    docs: Documentation-only changes.
    style: Changes that do not affect the meaning of the code (e.g., formatting).
    refactor: A code change that neither fixes a bug nor adds a feature.
    perf: Performance improvements.
    test: Adding missing tests or fixing them.
    chore: Routine tasks or maintenance.
    Scope (Optional):
    The scope gives more context and usually refers to the module or feature that the change is related to.

    Short Description:
    A brief and clear description of the change:

    It should be concise.
    Use imperative mood, like "add" instead of "adds" or "added".
    Body (Optional):
    A more detailed description of the change:

    Explain the motivation for the change and contrast it with the previous behavior.
    Again, use the imperative mood.
    Footer (Optional):
    Place to reference any issues or breaking changes.

    For linking issues, you can use "Closes #123" or "Related #123".
    For breaking changes, start with "BREAKING CHANGE:" followed by a detailed description of what changed.
    """
elif selected_option == "Atom Editor Commit Messages":
    selected_option = """Atom Editor Commit Messages
    Format of the Commit Message:
    ```
    <type>: Short description (less than 50 chars)
    <BLANK LINE>
    Longer description (if necessary), wrapped to about 72 characters.
    <BLANK LINE>
    Issue reference (if applicable)
    ```
    Type:
    Describes the nature of the change. Some common types include:

    :art:: Improving structure or format of the code.
    :racehorse:: Performance improvements.
    :non-potable_water:: Memory leaks or plugging other memory issues.
    :memo:: Documentation.
    :penguin:: Linux-specific changes or fixes.
    :apple:: macOS-specific changes or fixes.
    :checkered_flag:: Windows-specific changes or fixes.
    ... among others.
    It's worth noting that Atom uses emoji-like markers (but without the actual emojis) to represent the type of commit.

    Short Description:

    Provides a concise summary of the change.
    Should be less than 50 characters.
    Uses an imperative tone: "fix" not "fixes" or "fixed".
    Longer Description (Optional):

    If needed, provides a more in-depth explanation of the change.
    Each line should be wrapped at around 72 characters.
    Explains the context and reasons for the change, differences in the current behavior vs. the past, and any alternative solutions considered.
    Issue Reference (Optional):

    If the commit addresses a specific issue or pull request, reference it in this section.
    Using phrases like "Closes #123" or "Refs #123" to link the commit to its relevant issue or pull request.
    """

Custom_Commit = st.sidebar.text_area("Custom Commit Convention:", height=100)

system = "Please refer to the git diff and write your commit message according to the commit convention. Do not include unnecessary comments beyond the commit message."

def copy_text_to_clipboard(text):
    st.markdown(
        f'<textarea id="textarea" readonly rows="3" style="width:100%">{text}</textarea>'
        '<button onclick="copyText()">Copy</button>'
        '<script type="text/javascript">'
        'function copyText() {'
        '   var copyText = document.getElementById("textarea");'
        '   copyText.select();'
        '   document.execCommand("copy");'
        '}'
        '</script>',
        unsafe_allow_html=True
    )

def generate_commit_message(text_input, commit_convention, system, api_key=None, flag=0):
    if not api_key:
        st.error('API Key is missing.')
        return

    data = {'code': text_input, 'commit_convention': commit_convention, 'system': system}

    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-16k-0613", openai_api_key=api_key)  # 수정된 부분
    
    default_prompt = ChatPromptTemplate.from_template(
        """
        instruction : 
        ```
        {system}
        ```
        Commit Convention : 
        ```
        {commit_convention}
        ```
        git diff : 
        ```
        {code}
        ```
        """
    )
    default_chain = LLMChain(llm=llm, prompt=default_prompt, output_key="commit_message")
    overall_chain = SequentialChain(
        chains=[default_chain],
        input_variables=["code", "commit_convention", "system"],
        output_variables=["commit_message"],
        verbose=False
    )
    commit_message = overall_chain(data)['commit_message']
    mk = f"""```bash
{commit_message}
```"""
    st.markdown(mk)

with st.form(key='my_form'):
    text_input = st.text_area(label='Enter your commit message here', height=300)
    submit_button = st.form_submit_button(label='Submit')
    if not api_key.startswith('sk-'):
        st.error('Please enter a valid OpenAI API Key')
    if submit_button and api_key.startswith('sk-'):
        # generate_commit_message(text_input, 'angular', system, api_key=api_key)  # api_key를 명시적으로 전달
        if selected_option == "Custom Commit Convention":
            if Custom_Commit:
                generate_commit_message(text_input, Custom_Commit, system, api_key=api_key, flag=1)
        else:
            generate_commit_message(text_input, selected_option, system, api_key=api_key, flag=0)

    st.markdown("""### Update Log
 - Added the ability to directly copy generated commit messages.(10.25.2023) """)
