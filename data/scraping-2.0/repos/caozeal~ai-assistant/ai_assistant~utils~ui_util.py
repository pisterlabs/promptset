from openai.pagination import SyncCursorPage
from openai.types.beta.threads.thread_message import ThreadMessage
from tabulate import tabulate

def print_message(message:SyncCursorPage[ThreadMessage]):
     # Prepare a list of lists where each inner list represents a row
    table_data = []
    for message in reversed(message.data):
    # Assuming 'message.content' is a dictionary with keys as column names
    # and values as cell values in the row
        for content in message.content:
            row = [message.role, content.text.value]
        table_data.append(row)

    print(tabulate(table_data, headers=["Role", "Content"]))

def choose_assistants(assistant_infos):
    print("请选择助理：")
    table_data = []
    for i in range(len(assistant_infos)):
        row = [i, assistant_infos[i].description]
        table_data.append(row)
    table_data.append(["add", "新增助理"])
    print(tabulate(table_data, headers=["Index", "Description"]))
    return input(">>> ")