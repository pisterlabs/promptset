from preprocess import Preprocessor
from params import PreprocessParams, MainParams, BaselineParams, CaptionerParams
from captioner import Captioner
from langchain.llms import OpenAI
import torchvision.transforms as transforms
import os
import torch

from api import *
from incontext import in_context_examples

import numpy as np

from tqdm.rich import tqdm

finetune_dataset_name = "finetune_dataset_blip_400"

if os.path.exists(f"data/{finetune_dataset_name}.pt"):
    print(f"Loading dataset from data/{finetune_dataset_name}.pt")

    flattened_questions, flattened_frames, summaries = torch.load(f"data/{finetune_dataset_name}.pt")

    for i in range(len(flattened_questions)):
        print(f"Question: {flattened_questions[i]}, Summary: {summaries[i]}")

    quit()


# Get 3 frames per video
def load_real_dataset():
    data_json_path = "data/qa_data.json"
    frames_dir_path = "data/frames/"

    preprocessor = Preprocessor(data_json_path, frames_dir_path, PreprocessParams)
    questions, frames, answers, answer_choices = preprocessor.create_dataset()
    return questions, frames, answers, answer_choices


questions, frames, answers, answer_choices = load_real_dataset()

# frames: [ torch.tensor(n_frames, C, H, W) ] -> [ torch.tensor(3, C, H, W) ]


def extract_frames(frames_tensor, n_caption_frames=1):
    """_summary_

    Args:
        frames_tensor: shape = (n_frames, C, H, W)
        n_caption_frames [int]: how many frames in video to return

    Returns:
        chosen_frames torch.tensor: shape = (n_caption_frames, C, H, W)
    """
    if n_caption_frames == 1:
        # Extract the relevant frames given the frame list
        indices = [int(len(frames_tensor) / 2)]
    else:
        length = len(frames_tensor)
        indices = np.linspace(0, length - 1, n_caption_frames + 2, dtype=int)
        indices = indices[1:-1]

    return frames_tensor[indices]


assert len(frames) == len(questions) == len(answers)

captioner = Captioner("blip", CaptionerParams)
CaptionerParams.question_type = CaptionerParams.Configs.Caption

os.environ["OPENAI_API_KEY"] = openai_key_ronak
llm = OpenAI(model_name="gpt-3.5-turbo")


# For each frame, attach it's question and answer
flattened_questions = []
flattened_answers = []
flattened_frames = []
summaries = []
captions = []


for i in tqdm(range(len(frames))):
    video = extract_frames(
        frames[i], n_caption_frames=3
    )  # [ torch.tensor(3, C, H, W) ]
    for frame in video:
        image = transforms.ToPILImage()(frame)
        caption = captioner.caption(image, question=questions[i])

        flattened_questions.append(questions[i])
        flattened_answers.append(answers[i])
        flattened_frames.append(frame)

        prompt = (
            in_context_examples
            + "Original context: "
            + caption
            + "\nQuestion: "
            + questions[i]
            + "\nAnswer: "
            + answers[i]
            + "\nNew context:"
        )

        summary = llm(prompt)
        print("Answer: ", answers[i])
        print(f"Summary: {summary}")
        print("-------------")
        summaries.append(summary)


for summary, caption in zip(summaries, captions):
    print("Caption: ", caption)
    print("Summary: ", summary)

# Save the dataset
dataset = (flattened_questions, flattened_frames, summaries)
torch.save(dataset, f"data/{finetune_dataset_name}.pt")
print(f"Saved to data/{finetune_dataset_name}.pt")
