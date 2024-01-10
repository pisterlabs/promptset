import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

import requests

API_URL = "https://vbvhusef1oe4a22d.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Authorization": "Bearer xxx",
	"Content-Type": "application/json"
}

def query(input: str):
    # set inpu to be last 1000
	response = requests.post(API_URL, headers=headers, json={
	    "inputs": input,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 512,
        }
    })
	output = response.json()
	print(output)
	return output[0]['generated_text']
	

def extract_code_blocks(text: str) -> str:
    # Check if triple backticks exist in the text
    if '```' not in text:
        return text.replace("bpftrace -e", "").strip().strip("'")

    pattern = r'```(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    res = "\n".join(matches)
    return res.replace("bpftrace -e", "").strip().strip("'")

question = """
Write a bpftrace program that:

Monitors the rate of specific hardware interrupts and logs the interrupt sources

### Examples

Here are some examples to help you get started with bpftrace:

Below are some simple examples of bpftrace usage:

# trace processes calling sleep
'kprobe:do_nanosleep { printf("PID %d sleeping...
", pid); }'

# count syscalls by process name
'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Files opened by process
'tracepoint:syscalls:sys_enter_open { printf("%s %s
", comm, str(args->filename)); }'

# Syscall count by program
'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Read bytes by process:
'tracepoint:syscalls:sys_exit_read /args->ret/ { @[comm] = sum(args->ret); }'

# Read size distribution by process:
'tracepoint:syscalls:sys_exit_read { @[comm] = hist(args->ret); }'

# Show per-second syscall rates:
'tracepoint:raw_syscalls:sys_enter { @ = count(); } interval:s:1 { print(@); clear(@); }'

# Trace disk size by process
'tracepoint:block:block_rq_issue { printf("%d %s %d
", pid, comm, args->bytes); }'

# Count page faults by process
'software:faults:1 { @[comm] = count(); }'

# Count LLC cache misses by process name and PID (uses PMCs):
'hardware:cache-misses:1000000 { @[comm, pid] = count(); }'

# Profile user-level stacks at 99 Hertz, for PID 189:
'profile:hz:99 /pid == 189/ { @[ustack] = count(); }'

# Files opened, for processes in the root cgroup-v2
'tracepoint:syscalls:sys_enter_openat /cgroup == cgroupid("/sys/fs/cgroup/unified/mycg")/ { printf("%s
", str(args->filename)); }'

You can refer to the above examples to write your own bpftrace program to Monitors the rate of specific hardware interrupts and logs the interrupt sources.
"""

template = f"""<s>[INST] <<SYS>>
You should only write the bpftrace program itself.
No words other than bpftrace program.
No explaination and no instructions, no markers like ```.
output format should be: bpftrace -e <FILL_ME>
<</SYS>> {question} [/INST]
"""

res = query(template)
print(res)
print(extract_code_blocks(res))