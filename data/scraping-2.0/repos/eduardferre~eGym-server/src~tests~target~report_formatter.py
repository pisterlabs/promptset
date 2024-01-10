import os
import openai
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

# Remove the existing formatted_report.txt file if it exists
if os.path.exists("formatted_report.txt"):
    os.remove("formatted_report.txt")

if os.path.exists("table_report.txt"):
    os.remove("table_report.txt")

if os.path.exists("report.html"):
    os.remove("report.html")

if os.path.exists("table_git.txt"):
    os.remove("table_git.txt")


def ask_chat_gpt(content: str):
    conversation = [
        {
            "role": "system",
            "content": "You are my description test completion assistant. Your responses should be only the description, without any confirmation. The description should be similar to 'Test getting posts (204 - No Content)', so, the structure is 'description (HTTP_Code - HTTP_Status)'.",
        }
    ]

    conversation.append({"role": "user", "content": content})

    response = openai.Completion.create(
        engine="text-davinci-003",
        messages=conversation,
        max_tokens=100,
        api_key=openai.api_key,
    )

    chatgpt_response = response.choices[0].message["content"].strip()

    return chatgpt_response


# Read the report from a file (or you can modify this to read from any source)
with open("report_logs.txt", "r") as file:
    lines = file.readlines()

prev_lines = list()
report_lines = list()
post_lines = list()
flag = True

for count, line in enumerate(lines):
    if count < 8:
        prev_lines.append(line)
    elif not "[100%]" in lines[count - 1] and flag == True:
        report_lines.append(line)
    else:
        post_lines.append(line)
        flag = False

# Calculate the maximum row length
max_length = max(len(line) for line in report_lines)


# Function to format each line
def format_line(line):
    # Find the index of '%'
    percent_index = line.find("[")

    # Calculate the number of spaces needed between checkbox and '[ XX%]'
    spaces_needed = max_length - len(line)

    # Format the line with extra spaces
    formatted_line = line[:percent_index] + " " * spaces_needed + line[percent_index:]
    return formatted_line


# Format all lines
formatted_lines = [format_line(line) for line in lines]

# Write the formatted content to a new file
with open("formatted_report.txt", "w") as output_file:
    for count, line in enumerate(formatted_lines):
        if count < 8:
            output_file.write(prev_lines[count])
        elif not "[100%]" in formatted_lines[count - 1]:
            output_file.write(line)
        else:
            break

    for line in post_lines:
        output_file.write(line)

with open("table_report.txt", "w") as output_table_file:
    number = 0
    for count, line in enumerate(formatted_lines):
        if count < 9:
            None
        elif not "[100%]" in formatted_lines[count - 1]:
            number += 1
            output_table_file.write(str(number) + "      " + line)
        else:
            break

    output_table_file.write(f"\n\nThere are {number} tests\n\n")

with open("formatted_report.txt", "w") as output_file:
    for count, line in enumerate(formatted_lines):
        if count < 8:
            output_file.write(prev_lines[count])
        elif not "[100%]" in formatted_lines[count - 1]:
            output_file.write(line)
        else:
            break

    for line in post_lines:
        output_file.write(line)

with open("formatted_report.txt", "r") as file:
    text = file.read()

    html_text = "<!DOCTYPE html>\n<html>\n<head>\n<meta http-equiv='Content-Type' content='text/html; charset=utf-8'>\n<link rel='stylesheet' href='style.css' type='text/css'>\n</head>\n<body class='pyfile'>\n"

    for line in text.split("\n"):
        html_text += f"<p>{line}</p>\n"

    html_text += "</body>\n</html>"

with open("report.html", "w") as file:
    file.write(html_text)


def write_table():
    with open("table_report.txt", "r") as file:
        test_number = 0

        file_lines = file.readlines()
        table_text = "| # | Test Name | Description | Status |\n|---|-----------|-------------|--------|\n"

        for line in file_lines:
            if line == "\n":
                break
            test_number += 1
            test_name_pattern = r"_test\.py::(.*?) PASSED"
            test_name = str(re.findall(test_name_pattern, line))
            test_name = test_name.replace("['", "`").replace("']", "`")
            # test_description = ask_chat_gpt(
            #     f"Give me a description for this test: {test_name}"
            # )
            test_description = "Test description"
            test_status = "✅" if "PASSED" in line else "❌"
            table_text += f"| {test_number} | {test_name} | {test_description} | {test_status} |\n"

        return table_text


def create_table():
    with open("table_git.txt", "w") as table:
        table.write(write_table())


create_table()
