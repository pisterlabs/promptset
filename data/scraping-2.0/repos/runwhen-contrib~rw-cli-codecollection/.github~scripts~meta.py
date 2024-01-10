"""
Utility file to generate metadata file for each 
runbook. Initial purpose is to hold static 
command explanations for use with runwhen-local 
documentation. 

parse_robot_file written by Kyle Forster

Author: Shea Stewart
"""
import sys
import os
import json
import fnmatch
import re
import requests
import yaml
from urllib.parse import urlencode
from robot.api import TestSuite


def parse_robot_file(fpath):
    """
    Parses a robot file in to a python object that is
    json serializable, representing all kinds of interesting
    bits and pieces about the file contents (for UI purposes).
    """
    suite = TestSuite.from_file_system(fpath)
    # pprint.pprint(dir(suite))
    ret = {}
    ret["doc"] = suite.doc  # The doc string
    ret["type"] = suite.name.lower()
    ret["tags"] = []

    for k, v in suite.metadata.items():
        if k.lower() in ["author", "name"]:
            ret[k.lower()] = v
    tasks = []
    for task in suite.tests:
        tags = [str(tag) for tag in task.tags if tag not in ["skipped"]]
        # print (task.body)
        tasks.append(
            {
                "id": task.id,
                "name": task.name,
                # "tags": tags,
                "doc": str(task.doc),
                "keywords": task.body,
            }
        )
        ret["tags"] = list(set(ret["tags"] + tags))
    ret["tasks"] = tasks
    resourcefile = suite.resource
    ret["imports"] = []
    for i in resourcefile.imports:
        ret["imports"].append(i.name)
    return ret


def find_files(directory, pattern):
    """
    Search for files given directory and its subdirectories matching a pattern.

    Args:
        directory (str): The path of the directory to search.

    Returns:
        A list of file paths that match the search criteria.
    """
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def parse_yaml(fpath):
    with open(fpath, "r") as file:
        data = yaml.safe_load(file)
    return data


def search_keywords(parsed_robot, search_list):
    """
    Search through the list of keywords in the robot file,
    looking for interesting patterns that we can extrapolate.


    Args:
        parsed_robot (object): The parsed robot contents.
        search_list (list): A list of strings to match desired keywords on.

    Returns:
        A list of commands that matched the search pattern.
    """
    commands = []
    for task in parsed_robot["tasks"]:
        for keyword in task["keywords"]:
            if hasattr(keyword, "name"):
                for item in search_list:
                    # if item in keyword.name:
                    ## Switched to args to only match on  render_in_commandlist=true
                    ## Not sure if this is the most scalable approach, so it's just
                    ## a test for now
                    if item in keyword.args:
                        # commands.append(f'#{task["name"]}  \n')
                        # cmd = cmd_expansion(keyword.args, parsed_runbook_config)
                        # commands.append(f'{cmd}  \n')
                        command = {
                            "name": task["name"],
                            "command": cmd_expansion(keyword.args),
                        }
                        commands.append(command)
    return commands


def remove_escape_chars(cmd):
    cmd = cmd.replace("\\\%", "%")
    cmd = cmd.replace("\\\n", "")
    cmd = cmd.replace("\\\\", "\\")
    cmd = cmd.encode().decode("unicode_escape")
    ## Handle cases where a wrapped quote has returned
    if cmd[0] == cmd[-1] == '"':
        cmd = cmd[1:-1]
    if cmd[0] == cmd[-1] == "'":
        cmd = cmd[1:-1]
    return cmd


def cmd_expansion(keyword_arguments):
    """
    Cleans up the command details as sent in from robot parsing.
    Substitutes major binaries in for better command explanation,
    and escapes special characters.


    Args:
        keyword_arguments (object): The cmd arguments as parsed from robot.

    Returns:
        A cleaned up and variable expanded command string.
    """
    cmd_components = str(keyword_arguments)

    ## Clean up the parsed cmd from robot
    cmd_components = cmd_components.lstrip("(").rstrip(")")

    ## Split by comma if comma is not wrapped in single or escaped quotes
    ## this is needed to separate the command from the args as
    ## parsed by the robot parser
    split_regex = re.compile(r"""((?:[^,'"]|'(?:(?:\\')|[^'])*'|"(?:\\"|[^"])*")+)""")
    cmd_components = split_regex.split(cmd_components)[1::2]
    if cmd_components[0].startswith(("'cmd=", '"cmd=', "cmd=")):
        cmd_components[0] = cmd_components[0].replace("cmd=", "")

        ## Substitute in the proper binary
        ## TODO Consider a check for Distribution type
        ## Jon Funk mentioned that distrubiton type might not be used
        cmd_str = cmd_components[0]

        if "binary_name" in cmd_str:
            cmd_str = cmd_str.replace("${binary_name}", "kubectl")
        if "BINARY_USED" in cmd_str:
            cmd_str = cmd_str.replace("${BINARY_USED}", "kubectl")
        if "KUBERNETES_DISTRIBUTION_BINARY" in cmd_str:
            cmd_str = cmd_str.replace("${KUBERNETES_DISTRIBUTION_BINARY}", "kubectl")
    elif cmd_components[0].startswith("'bash_file="):
        script = cmd_components[0].replace("bash_file=", "")
        matched_cmd_override = None
        for arg in cmd_components:
            arg = arg.strip()
            if arg.strip().startswith("'cmd_override"):
                matched_cmd_override = arg
                break
        if matched_cmd_override is not None:
            cmd_parts = matched_cmd_override.split()
            cmd_arguments = " ".join(cmd_parts[1:])
            cmd_arguments = cmd_arguments.strip("'")
            cmd_components[0] = f"bash {script} {cmd_arguments}"
        else:
            cmd_components[0] = f"bash {script}"
        cmd_str = cmd_components[0]
    else:
        cmd_str = "Could not render command"
    # Set var for public command before configProvided substitutiuon
    # This is used for the Explain function and guarantees no sensitive information
    cmd = remove_escape_chars(cmd_str)

    return cmd


def check_url(url):
    try:
        response = requests.head(url)
        return response.status_code != 404
    except requests.RequestException:
        return False


def strip_md_codeblock_formatting(string):
    string = string.replace("```bash", "")
    string = string.replace("```shell", "")
    string = string.replace("```sh", "")
    string = string.replace("```", "")
    return string


def generate_metadata(directory_path):
    """
    Gets passed in a directory to scan for robot files.
    Performs variable substitution only for command binaries,
    written out  metadata file.

    Args:
        args (str): The path the output contents from map-builder.

    Returns:
        Object
    """
    explainUrl=f'https://papi.test.runwhen.com/bow/raw?'
    headers = {"Content-Type": "application/json"}
    search_list = ["render_in_commandlist=true", "show_in_rwl_cheatsheet=true"]
    runbook_files = find_files(directory_path, "runbook.robot")
    for runbook in runbook_files:
        print(f"generating meta for {runbook}")
        parsed_robot = parse_robot_file(runbook)
        interesting_commands = search_keywords(parsed_robot, search_list)
        commands = []

        for item in interesting_commands:
            name = item["name"]
            command = item["command"]

            # Remove special chars for vars in name
            name = name.replace("`", "").replace("${", "").replace("}", "")
            # Convert name to lower snake case
            name_snake_case = re.sub(r"\W+", "_", name.lower())

            # Generate metadata depending if it's a one-liner command or bash script
            if command.startswith("bash"):
                script_name = command.split()[1].strip("'")
                script_path_parts = runbook.split("/")
                script_path = "/".join(script_path_parts[:-1])
                full_script_path = f"{script_path}/{script_name}"
                print(f"Generating metadata for script {full_script_path}")
                with open(full_script_path, "r") as script:
                    script_contents = script.read()
                # What it does
                query_what_it_does_prompt = f"Provide a one or two sentence explanation of this script as if I was loosely familiar with the concepts but it wasn't part of my daily job."
                query_what_it_does_with_command = (
                    f"{query_what_it_does_prompt} \n{script_contents}"
                )
                print(f"generating explanation for {name_snake_case}")
                explain_query_what_it_does = {"prompt": query_what_it_does_with_command}
                explain_query_json = json.dumps(explain_query_what_it_does)
                response_what_it_does = requests.post(
                    explainUrl, data=explain_query_json, headers=headers
                )
                if response_what_it_does.status_code == 200:
                    explanation = response_what_it_does.json()
                    explanation_content = explanation["explanation"]
                else:
                    explanation_content = "Explanation not available"
                # Generate multi-line explanation
                query_multi_line_with_comments_prompt = f"Augment this script with comments (written it it's language) for newer or less experienced devops engineers:"
                query_multi_line_with_command = (
                    f"{query_multi_line_with_comments_prompt}\n{script_contents}"
                )
                # Generate external doc links
                query_doc_links_prompt = r"Given the following explanation, generate some links that provide helpful documentation for a reader who want's to learn more about the topics used in the command. Format the output in a single YAML list with the keys of `description` and `url` for each link with the values in double quotes. Ensure each description and url are on separate lines, ensure an empty blank line separates each item. Ensure there are no other keys or text or extra characters other than the items. The script contents are:  "
                # query_doc_links_with_command = f'{query_doc_links_prompt}\n{multi_line_content}'
                query_doc_links_with_command = (
                    f"{query_doc_links_prompt}\n{explanation_content}"
                )
            else:
                # What it does
                query_what_it_does_prompt = f"Provide a one or two sentence explanation of this command as if I was loosely familiar with the concepts but it wasn't part of my daily job."
                query_what_it_does_with_command = (
                    f"{query_what_it_does_prompt} \n{command}"
                )
                print(f"generating explanation for {name_snake_case}")
                explain_query_what_it_does = {"prompt": query_what_it_does_with_command}
                explain_query_json = json.dumps(explain_query_what_it_does)
                response_what_it_does = requests.post(
                    explainUrl, data=explain_query_json, headers=headers
                )
                if response_what_it_does.status_code == 200:
                    explanation = response_what_it_does.json()
                    explanation_content = explanation["explanation"]
                else:
                    explanation_content = "Explanation not available"
                # Generate multi-line explanation
                query_multi_line_with_comments_prompt = f"Convert this command to a multi-line command with helpful and educational comments for newer or less experienced devops engineers: "
                query_multi_line_with_command = (
                    f"{query_multi_line_with_comments_prompt}\n{command}"
                )
                # Generate external doc links
                query_doc_links_prompt = r"Given the following command explanation, generate some links that provide helpful documentation for a reader who want's to learn more about the topics used in the command. Format the output in a single YAML list with the keys of `description` and `url` for each link with the values in double quotes. Ensure each description and url are on separate lines, ensure an empty blank line separates each item. Ensure there are no other keys or text or extra characters other than the items. The command is:  "
                # query_doc_links_with_command = f'{query_doc_links_prompt}\n{multi_line_content}'
                query_doc_links_with_command = (
                    f"{query_doc_links_prompt}\n{explanation_content}"
                )

            print(f"generating multi-line code with comments for {name_snake_case}")
            # explain_query_multi_line_with_comments = urlencode({'prompt': query_multi_line_with_command})
            # url_multi_line_with_comments = f'{explainUrl}{explain_query_multi_line_with_comments}'
            explain_query_multi_line_with_comments = {
                "prompt": query_multi_line_with_command
            }
            explain_query_multi_line_json = json.dumps(
                explain_query_multi_line_with_comments
            )
            response_multi_line_with_comments = requests.post(
                explainUrl, data=explain_query_multi_line_json, headers=headers
            )
            if response_multi_line_with_comments.status_code == 200:
                multi_line = response_multi_line_with_comments.json()
                multi_line_content = multi_line["explanation"]

                print(f"generating doc-links for {name_snake_case}")
                # explain_query_doc_links = f"{'prompt': query_doc_links_with_command}"
                # url_doc_links = f'{explainUrl}{explain_query_doc_links}'
                # response_doc_links = requests.get(url_doc_links)
                explain_query_doc_links = {"prompt": query_doc_links_with_command}
                explain_query_doc_links_json = json.dumps(explain_query_doc_links)
                response_doc_links = requests.post(
                    explainUrl, data=explain_query_doc_links_json, headers=headers
                )
                if response_doc_links.status_code == 200:
                    doc_links = response_doc_links.json()
                    doc_links_content = doc_links["explanation"]
                    # Try to clean up poor openAI formatting
                    corrected_input = re.sub(
                        r"url:(?=[^\s])", r"url: ", doc_links_content
                    )

                    # Try to be flexible about spaces and indentation
                    pattern = r"^\s*-.*\bdescription:.*\n\s+(?:url\s*:|url)\s*:.*"
                    # Find all matches using the pattern
                    matches = re.findall(pattern, corrected_input, re.MULTILINE)
                    # Join the matched lines to create the final output
                    output_string = "\n".join(matches)
                    non_404_urls = []
                    # Siltently Discard any poor yaml - some content from openAI is still inconsistent
                    # Otherwise build a list of URLS that still exist (as openAI generates some links that are 404s)
                    try:
                        yaml_data = yaml.safe_load(output_string)
                        if yaml_data:  # Check if yaml_data is not None
                            for item in yaml_data:
                                if (
                                    isinstance(item, dict)
                                    and isinstance(item.get("description"), str)
                                    and isinstance(item.get("url"), str)
                                ):
                                    if check_url(item["url"]):
                                        non_404_urls.append(item)
                    except yaml.YAMLError:
                        pass
                    markdown_links = []
                    for link in non_404_urls:
                        if "description" in link and "url" in link:
                            description = link["description"]
                            url = link["url"]
                            markdown_links.append(
                                f'[{description}]({url}){{:target="_blank"}}'
                            )
                    # Format markdown lines
                    formatted_markdown_lines = "\n".join(
                        [f"- {item}" for item in markdown_links]
                    )
                    doc_links_content = formatted_markdown_lines

                else:
                    doc_links_content = "Documentation links not available"
            else:
                multi_line_content = "Multi-line script not available"
                doc_links_content = "Documentation links not available"

            command_meta = {
                "name": name_snake_case,
                "command": command,
                "explanation": strip_md_codeblock_formatting(explanation_content),
                "multi_line_details": strip_md_codeblock_formatting(multi_line_content),
                "doc_links": f"\n{doc_links_content}",
            }
            # Add the command meta to the list of commands
            commands.append(command_meta)

        # Create a dictionary with the commands list
        yaml_data = {"commands": commands}

        # Write out the YAML file
        dir_path = os.path.dirname(runbook)
        file_path = os.path.join(dir_path, "meta.yaml")
        with open(file_path, "w") as f:
            yaml.dump(yaml_data, f)
        print(f"writing meta.yml for {runbook}")


if __name__ == "__main__":
    generate_metadata(sys.argv[1])
