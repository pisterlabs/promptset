import os, json, time
from tqdm import tqdm
import numpy as np
import base64
from torch.utils.data import Dataset, DataLoader

from openai import OpenAI, AsyncOpenAI
import asyncio


class FramesDataset(Dataset):
    def __init__(self, frame_dir, data_file):
        self.frame_dir = frame_dir
        self.data_file = data_file
        self.qids = []
        self.vids = []
        self.queries = []
        self.labels = []
        self.read_data()

    def read_data(self):
        with open(self.data_file, "r") as f:
            for line in f:
                data = json.loads(line)
                self.qids.append(data["qid"])
                self.vids.append(data["vid"])
                self.queries.append(data["query"])
                self.labels.append(data["relevant_clip_ids"])

    def read_frames(self, idx):
        frames_path = os.path.join(self.frame_dir, self.vids[idx])
        frames = sorted(os.listdir(frames_path))
        base64Frames = []
        for frame in frames:
            with open(os.path.join(frames_path, frame), "rb") as f:
                base64Frames.append(base64.b64encode(f.read()).decode("utf-8"))
        return base64Frames

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        item = {
            "qid": self.qids[idx],
            "vid": self.vids[idx],
            "query": self.queries[idx],
            "labels": self.labels[idx],
            "frames_dir": os.path.join(self.frame_dir, self.vids[idx]),
        }
        return item


def decode(frames_dir):
    base64Frames = []
    frames = sorted(os.listdir(frames_dir))
    for frame in frames:
        with open(os.path.join(frames_dir, frame), "rb") as f:
            base64Frames.append(base64.b64encode(f.read()).decode("utf-8"))
    return base64Frames

async def wait_test(client, params):
    try:
        #Fake Async Server Call
        # await asyncio.wait_for(gpt.async_request_function(message), timeout=5)
        response = await asyncio.wait_for(client.chat.completions.create(**params), timeout=5)
        return True, response
    except asyncio.TimeoutError:
        print("TimeoutError")
        # sys.exit(1)
        return False, None

def main(client, dataloader, out_dir, log_path, batch_size=4, max_tokens=1000, seed=1013, stop_flag=30):
    prices = 0
    for idx, item in enumerate(tqdm(dataloader)):
        
        if idx >= stop_flag:
            break
        qid = item["qid"].numpy()[0].astype(str)
        vid = item["vid"][0]
        query = item["query"][0]
        labels = [i.item() for i in item['labels']]
        frames_dir = item["frames_dir"][0]
        output_file = os.path.join(out_dir, '{}.json'.format(qid))
        print("======={}_{}=======".format(idx, qid))
        if os.path.exists(output_file):
            print("Exist {}".format(qid))
            continue
        if qid == "3403":
            print("Skip {}".format(qid))
            continue
        
        # print(qid, vid, query, labels, frames)

        frames = decode(frames_dir)
        output = {
            "qid": qid,
            "vid": vid,
            "query": query,
            "relevant_clip_ids": labels,
        }

        responses = {}
        for i in range(len(frames) // batch_size):
            
            batch_frames = frames[i * batch_size : (i + 1) * batch_size]
            # print(len(batch_frames))output,

            prompt = '''These are frames from a video that I want to upload.
            Query: <QUERY>
            Is this video potentially consistent with the query?
            You only need to answer yes or no.
            '''
            
            prompt = prompt.replace("<QUERY>", query)
            # print(prompt)

            PROMPT_MESSAGES = [
                # {
                #     "role": "system",
                #     "content": "You are a helpful assistant.",
                # },
                {
                    "role": "user",
                    "content": [
                        prompt,
                        *map(lambda x: {"image": x, "resize": 534}, batch_frames),
                    ],
                },
            ]
            params = {
                "model": "gpt-4-vision-preview",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200,
            }
            while True:
                done, result = asyncio.run(wait_test(client, params))
                if done:
                    break
                else:
                    print("Re-try the request")
                    continue
            answer = result.choices[0].message.content
            prompt_tokens = result.usage.prompt_tokens
            completion_tokens = result.usage.completion_tokens
            price = (prompt_tokens * 0.01 + completion_tokens * 0.03) / 1000
            prices += price
            
            responses[str(i)] = answer
            print("Answer: ", answer)

            print(
                f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_price={price}"
            )
            log = {
                "uid": qid + "_{}".format(i),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "price": price,
            }
            print("Accumulated price: ", prices)
            # Sleep for 1 second to avoid the API rate limit
            time.sleep(1)
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")
        
        output["answers"] = responses

        print(output)
        
        # Write the response to the output file
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    # Configure the GPT chatbot
    organization = "<Your organization>"
    api_key = "<Your api key.>"
    client = AsyncOpenAI(api_key=api_key, organization=organization)
    # client = OpenAI(api_key=api_key, organization=organization)

    # data_path
    frame_dir = "/home/shinji106/ntu/LLaVA/videos/QVHighlights/qv_frames"
    data_file = "/home/shinji106/ntu/LLaVA/videos/QVHighlights/data/highlight_val_release_100.jsonl"
    out_dir = "./data/QVHighlights/gpt_output_100_2/"
    log_path = "./data/QVHighlights/log/gpt_output_100_2.jsonl"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create Dataset
    dataset = FramesDataset(frame_dir, data_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(len(dataloader))

    main(client, dataloader, out_dir, log_path)
