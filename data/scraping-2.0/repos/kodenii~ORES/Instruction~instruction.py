import openai
import asyncio
import time
import traceback
import openai.error
import random
import os
from tqdm import tqdm
openai.api_key = "YOUR_API_KEY"

TASK_DISCRIPTION = f'''A user will input an image and concept(s), you should generate a new image thoroughly replace given concept to the opposite one. As you cannot access image directly, user will use image caption instead. You should also output image caption in a short sentence with few words. Skip concept(s) irrelevant to the image. The input is always valid.'''

INSTRUCTION_INIT = f'''You are working to help other LLM to complete the task. Task Description: {TASK_DISCRIPTION}

You can formulate some rules or steps. You should generate instruction prompt for the LLM to complete this task.

Instruction:
'''

OPTIMIZE_PROMPT = '''Here are results from the LLM. You can formulate some rules or steps. Update or rewrite the instruction for it based on your evaluation.

{}

Instruction:
'''
def pack_gt(samples):
    batch = []
    for sample in samples:
        input_caption, concept, pred_caption, gt_caption = sample
        pack = f'''Input Caption: {input_caption}
Ground-truth Answer 1: {gt_caption[0]}
Ground-truth Answer 2: {gt_caption[1]}
Ground-truth Answer 3: {gt_caption[2]}
LLM Answer: {pred_caption}
'''
        batch.append(pack)
    return "\n".join(batch)

def pack_pred(samples):
    batch = []
    for sample in samples:
        input_caption, concept, pred_caption, gt_caption = sample
        pack = f'''Input Caption: {input_caption}
Concept to Replace: {concept}
LLM Answer: {pred_caption}
'''
        batch.append(pack)
    return "\n".join(batch)

def forward(batch, instruction, history):
    samples = []
    for data in tqdm(batch):
        input_caption = data["Input Caption"]
        concept = data["Concept"]
        gt_caption = data["Output Caption"]
        caption, history = asyncio.run(send(get_caption, [instruction, input_caption, concept, history]))
        samples.append([input_caption, concept, caption, gt_caption])
    return samples, history

async def get_instruction():
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
    {"role": "user", "content": INSTRUCTION_INIT},
    ])

    content = completion["choices"][0]["message"]["content"]
    history = [{"role": "user", "content": INSTRUCTION_INIT}, {"role": "assistant", "content": content}]
    return content, history

async def optimize(batch, history):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history+[
    {"role": "user", "content": OPTIMIZE_PROMPT.format(batch)},
    ])

    content = completion["choices"][0]["message"]["content"]
    if len(history) >= 6:
        history = history[:2] + history[-2:]
    history = history + [{"role": "user", "content": OPTIMIZE_PROMPT.format(batch)}, {"role": "assistant", "content": content}]
    return content, history

async def get_caption(instruction, input_caption, concept, history):
    tmp = f'''You are working as a data annotator. Complete the task following the instruction.

Task Description: {TASK_DISCRIPTION}

Instruction:
{instruction}

Do not generate other things. Generate answer directly.

Input Caption: {input_caption}
Concept to Remove: {concept}
Output Caption: '''
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history+[
    {"role": "user", "content": tmp},
    ])

    content = completion["choices"][0]["message"]["content"]
    if len(history) >= 6:
        history = history[:2] + history[-2:]
    history = history + [{"role": "user", "content": tmp}, {"role": "assistant", "content": content}]
    return content, history

async def send(func, args=[]):
    while True:
        try:
            task = asyncio.create_task(func(*args))
            await asyncio.wait_for(task, timeout=10)
            result = task.result()
        except openai.error.RateLimitError:
            task.cancel()
            print('Rate Limit, wait 3s & retry')
            time.sleep(3)
            continue
        except asyncio.TimeoutError:
            task.cancel()
            print('Timeout, retry')
            continue
        except KeyboardInterrupt:
            os._exit(0)
        except:
            task.cancel()
            print('Unkown error, retry')
            print(traceback.format_exc())
            time.sleep(3)
            continue
        else:
            break
    return result

with open("training.txt") as f:
    dataset = []
    lines = "".join(f.readlines()).split("\n\n")
    for line in lines:
        tmp = line.split("\n")
        dataset.append({
            "Input Caption": tmp[0].strip()[15:],
            "Concept": tmp[1].strip()[9:],
            "Output Caption": [tmp[2].strip()[17:], tmp[3].strip()[17:], tmp[4].strip()[17:],]
            })

Epoch = 3
Batch_size = 6

raw_dataset = dataset.copy()

writer = open("results.txt", "w")

instruction, history = asyncio.run(send(get_instruction))

for epoch in range(Epoch):
    for start_idx in range(0, len(dataset), Batch_size):
        batch = dataset[start_idx:min(start_idx+Batch_size, len(dataset))]
        samples, history = forward(batch, instruction, history)
        packed_pred = pack_pred(samples)
        packed_gt= pack_gt(samples)
        instruction, history = asyncio.run(send(optimize, ["Example Cases:\n" + packed_gt + "\nLLM Cases:\n" + packed_pred , history]))
    writer.writelines(instruction + "\n")
    writer.writelines("="*20 + "\n")
    writer.writelines(f"Epoch {epoch} finished." + "\n")
    writer.writelines("="*20 + "\n")
    writer.flush()
    random.shuffle(dataset)

writer.close()