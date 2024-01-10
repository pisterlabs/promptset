from huey.contrib.djhuey import db_task
from llm.models import Webpage
import re
import openai
import difflib


def extract_changed_parts(code1, code2):
    differ = difflib.Differ()
    diff = differ.compare(code1.splitlines(), code2.splitlines())
    added_lines = []
    removed_lines = []
    unchanged_lines = []
    for line in diff:
        if line.startswith('+'):
            added_lines.append(line[2:])
        elif line.startswith('-'):
            removed_lines.append(line[2:])
        elif line.startswith(' '):
            unchanged_lines.append(line[2:])
    added_part = '\n'.join(added_lines[:200])
    removed_part = '\n'.join(removed_lines[:200])
    unchanged_part = '\n'.join(unchanged_lines[:200])
    added_part += '\n' * (200 - min(len(added_lines), 200))
    removed_part += '\n' * (200 - min(len(removed_lines), 200))
    unchanged_part += '\n' * (200 - min(len(unchanged_lines), 200))

    return added_part, removed_part, unchanged_part


def extract_changed_part(code):
    differ = difflib.Differ()
    diff = differ.compare(code.splitlines(), ''.splitlines())
    added_lines = []
    removed_lines = []

    for line in diff:
        if line.startswith('+'):
            added_lines.append(line[2:])
        elif line.startswith('-'):
            removed_lines.append(line[2:])
    changed_lines = added_lines + removed_lines
    changed_part = '\n'.join(changed_lines[:200])
    return changed_part


def extract_code(input_string):
    # Extract HTML code
    html_match = re.search(r'index.html:(.*)styles.css:', input_string, re.DOTALL)
    html_code = html_match.group(1) if html_match else ''
    html_code = html_code.strip()
    html_code = html_code.replace('```html', '')
    html_code = html_code.replace('```', '')
    html_code = html_code.strip()

    # Extract CSS code
    css_match = re.search(r'styles.css:(.*)scripts.js:', input_string, re.DOTALL)
    css_code = css_match.group(1) if css_match else ''
    css_code = css_code.strip()
    css_code = css_code.replace('```css', '')
    css_code = css_code.replace('```', '')
    css_code = css_code.strip()

    # Extract JS code
    js_match = re.search(r'scripts.js:(.*)```', input_string, re.DOTALL)
    js_code = js_match.group(1) if js_match else ''
    js_code = js_code.strip()
    js_code = js_code.replace('```javascript', '')
    js_code = js_code.replace('```', '')
    js_code = js_code.strip()

    return html_code, css_code, js_code


def get_completion_from_messages(
    messages,
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]


def generate_commit_message(
    webpage: Webpage,
    code: str,
    old_code: str = None
):
    if not old_code:
        changed_part = extract_changed_part(code)
    else:
        added_part, removed_part, unchanged_part = extract_changed_parts(old_code, code)
        changed_part = added_part if len(added_part) > len(removed_part) else removed_part if len(removed_part) > len(added_part) else unchanged_part

    messages = [
        {
            'role': 'system',
            'content': """
                You are a git user.
                You are given a code.
                You need to generate a commit message based on the code.
                The code will be delimited with ``` characters.
                You should only return the commit message.
                The commit message should be based on the code.
                It should be of the form: Updated the <part> of the webpage. or Added a <part> to the webpage. or Removed a <part> from the webpage. or Fixed a <part> of the webpage.
                The generated commit message should be human readable and understandable.
                The commit message should be delimited inside #### characters.
                
                EXAMPLE:
                ####Updated the header of the webpage.####
                ####Added a footer to the webpage.####
                ####Removed a button from the webpage.####
                ####Fixed a bug in the webpage.####
            """
        },
        {
            'role': 'user',
            'content': f"""
                ```
                {changed_part}
                ```
            """
        }
    ]
    openai.api_key = webpage.session.user.openai_key
    response = get_completion_from_messages(messages)
    commit_message = re.search(r'####(.*)####', response, re.DOTALL).group(1)
    return commit_message


@db_task()
def generate_static_pages(webpage: Webpage):
    webpage.state = 'GENERATING'
    webpage.save()
    messages = [
        {
            'role': 'system',
            'content': """
                You are a html, css, js developer.
                You are given a topic and specifications. 
                You need to generate a static website based on the topic and specifications.
                You should only return html, css and javascript codes.
                Do not include all the code in one file. 
                Divide it to index.html, styles.css and scripts.js.
                While returning the code, you should return it in the form:
                index.html: ```<html code>``` styles.css: ```<css code>``` scripts.js: ```<js code>```
                The topic will be delimited with @@@@ characters.
                The specifications will be delimited with #### characters.
            """
        },
        {
            'role': 'user',
            'content': f"""
                @@@@{webpage.topic}@@@@
                ####{webpage.specifications}####
            """
        }
    ]
    openai.api_key = webpage.session.user.openai_key
    response = get_completion_from_messages(messages)
    html_code, css_code, js_code = extract_code(response)
    webpage.htmlContent = html_code
    webpage.cssContent = css_code
    webpage.jsContent = js_code
    webpage.state = 'GENERATED'
    webpage.save()
    if webpage.hostOnGithub:
        webpage.publish_webpage()


@db_task()
def update_static_page(webpage: Webpage):
    previous_state = webpage.state
    webpage.state = 'GENERATING'
    webpage.save()
    messages = [
        {
            'role': 'system',
            'content': """
                You are a html, css, js developer.
                You are given a topic and specifications. 
                You need to update the static website based on the topic and specifications. 
                You should only return html, css and javascript codes.
                Do not include all the code in one file. 
                Divide it to index.html, styles.css and scripts.js.
                The previous code will be given to you.
                The previous html code will be delimited with ```html characters.
                The previous css code will be delimited with ```css characters.
                The previous js code will be delimited with ```javascript characters.
                The topic will be delimited with @@@@ characters.
                The specifications will be delimited with #### characters.
                While returning the code, you should return it in the form:
                index.html: ```<html code>``` styles.css: ```<css code>``` scripts.js: ```<js code>```
            """
        },
        {
            'role': 'user',
            'content': f"""
                ```html
                {webpage.htmlContent}
                ```
                ```css
                {webpage.cssContent}
                ```
                ```javascript
                {webpage.jsContent}
                ```
                @@@@{webpage.topic}@@@@
                ####{webpage.specifications}####
            """
        }
    ]
    old_code = webpage.htmlContent + webpage.cssContent + webpage.jsContent
    openai.api_key = webpage.session.user.openai_key
    response = get_completion_from_messages(messages)
    html_code, css_code, js_code = extract_code(response)
    webpage.htmlContent = html_code
    webpage.cssContent = css_code
    webpage.jsContent = js_code
    webpage.state = 'GENERATED'
    webpage.save()
    if previous_state == 'PUBLISHED':
        update_deployment(webpage, old_code)


def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = text.lower()
    return text


@db_task()
def publish_webpage(webpage: Webpage):
    webpage.state = 'PUBLISHING'
    webpage.save()
    from llm.utils.github import create_github_repo
    github_username = webpage.session.user.github_username
    github_token = webpage.session.user.github_token
    repo_name = slugify(webpage.topic)[:20]
    webpage.repo_name = repo_name
    html_code = webpage.htmlContent
    css_code = webpage.cssContent
    js_code = webpage.jsContent
    commit_message = generate_commit_message(
        webpage=webpage,
        code=html_code + css_code + js_code
    )
    hosted_url = create_github_repo(
        github_username,
        github_token,
        repo_name,
        html_code,
        css_code,
        js_code,
        commit_message
    )
    webpage.hostedUrl = hosted_url
    webpage.state = 'PUBLISHED'
    webpage.save()


@db_task()
def update_deployment(
    webpage: Webpage,
    old_code: str
):
    webpage.state = 'PUBLISHING'
    webpage.save()
    from llm.utils.github import update_github_repo
    github_username = webpage.session.user.github_username
    github_token = webpage.session.user.github_token
    repo_name = webpage.repo_name
    html_code = webpage.htmlContent
    css_code = webpage.cssContent
    js_code = webpage.jsContent
    commit_message = generate_commit_message(
        webpage,
        code=html_code + css_code + js_code,
        old_code=old_code
    )
    update_github_repo(
        username=github_username,
        token=github_token,
        repo_name=repo_name,
        new_js_content=js_code,
        new_css_content=css_code,
        new_html_content=html_code,
        commit_message=commit_message
    )
    webpage.state = "PUBLISHED"
    webpage.save()


@db_task()
def delete_deployment(webpage: Webpage):
    webpage.state = "PUBLISHING"
    webpage.save()
    from llm.utils.github import delete_github_repo
    github_username = webpage.session.user.github_username
    github_token = webpage.session.user.github_token
    repo_name = webpage.repo_name
    delete_github_repo(
        username=github_username,
        token=github_token,
        repo_id=repo_name
    )
    webpage.state = "GENERATED"
    webpage.hostOnGithub = False
    webpage.save()


__all__ = [
    'generate_static_pages',
    'update_static_page',
    'publish_webpage',
    'delete_deployment',
    'update_deployment'
]
