import toml
import openai

log_file = open("message_log.txt", "w")
with open("config.toml") as config_file:
    config = toml.load(config_file)

if not config["OPENAI_API_KEY"]:
    print("ERROR! Please set your OPENAI_API_KEY in config.toml")
    exit()


def write_output(msg):
    print(msg)
    log_file.write(msg + "\n")


client = openai.Client(api_key=config["OPENAI_API_KEY"])
messages = client.beta.threads.messages.list(thread_id=config["THREAD_ID"])
contents = []
roles = []
for this_msg in messages.data:
    roles.append(this_msg.role)
    contents.append(this_msg.content[0].text.value)
contents.reverse()
roles.reverse()
write_output("-" * 70)
for i in range(len(contents)):
    write_output("[" + roles[i] + "]")
    write_output("-" * 70)
    write_output(contents[i])
    write_output("-" * 70)
