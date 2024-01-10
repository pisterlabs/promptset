import os
from langchain.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

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

llm = DeepInfra(model_id="bigcode/starcoder")
llm.model_kwargs = {
    "temperature": 0.5,
    "max_new_tokens": 1024,
    "top_p": 0.9,
}

template = f"""
You should write a bpftrace program for me.
No words other than bpftrace program.
No explaination, no examples and no instructions, no markers like ```.
{question}
Please complete the following bpftrace program in format bpftrace -e <fill_me>:

bpftrace -e 
"""

res = llm.predict(template)
print(res)
print(extract_code_blocks(res))