# Imports
import argparse
import json
import numpy as np
import os
import openai
import random

from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Variables
# OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# OpenAI GPT-3 config
question_id = 1005
n_shot = 16
n_ensemble = 1

# Computing similarity config
encoder = "openai/clip-vit-base-patch16"

# Context files
parent_dir = Path(__file__).resolve().parents[0] / "context" 
random_img_for_clip = parent_dir / "img.jpg"
class_dir = "class"
training_idx_file = parent_dir / class_dir / "train_idx.json"
training_question_features = parent_dir / class_dir / "question_feature.npy"
training_question_file = parent_dir / class_dir / "question.json"
training_answer_file = parent_dir / class_dir / "answer.json"


# Main functions/classes
def process_answer(answer):
    to_be_removed = {""}
    answer_list = answer.split(" ")
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return " ".join(answer_list).lower()


class Type:
  """
  Main question_answering class
  """

  def __init__(self, question_info, question_idx, question_text_embed):
    self.question = question_info
    (
        self.traincontext_answer_dict,
        self.traincontext_question_dict,
    ) = self.load_anno(
        training_answer_file,
        training_question_file,
        None,
    )
    self.train_keys = list(self.traincontext_answer_dict.keys())
    self.load_similarity(question_idx, question_text_embed)
    
  def predict(self):
    _, question_dict = self.load_anno(None, None, self.question)

    key = list(question_dict.keys())[0]
    question = question_dict[key]

    context_key_list = self.get_context_keys(
        key,
        n_shot * n_ensemble,
    )

    # prompt format following GPT-3 QA API
    prompt = "The following is a list of questions and the categories they fall into:\n\n"
    for ni in range(n_shot):
        if context_key_list is None:
            context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
        else:
            context_key = context_key_list[ni]
        while True:  # make sure get context with valid question and answer
            if (
                len(self.traincontext_question_dict[context_key]) != 0
                and len(self.traincontext_answer_dict[context_key][0]) != 0
            ):
                break
            context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]

        prompt += "%s\nCategory: %s\n\n" % (
            self.traincontext_question_dict[context_key],
            self.traincontext_answer_dict[context_key],
        )
    prompt += "%s\Category: " % question

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    return process_answer(response["choices"][0]["text"])

  def get_context_keys(self, key, n):
    # combined with Q-similairty
    lineid = self.valkey2idx[key]
    question_similarity = np.matmul(self.train_feature, self.val_feature.detach().numpy()[lineid, :])
    index = question_similarity.argsort()[-n:][::-1]
    return [self.train_idx[str(x)] for x in index]
    
  def load_similarity(self, question_idx, question_feature):
      val_idx = question_idx
      self.valkey2idx = {}
      for ii in val_idx:
        self.valkey2idx[val_idx[ii]] = int(ii)
      self.train_feature = np.load(training_question_features)
      self.val_feature = question_feature
      self.train_idx = json.load(
          open(
              training_idx_file,
              "r",
          )
      )

  def load_anno(self, answer_anno_file, question_anno_file, questions):
    if answer_anno_file is not None:
        answer_anno = json.load(open(answer_anno_file, "r"))
    if question_anno_file is not None:
        question_anno = json.load(open(question_anno_file, "r"))
    else:
        question_anno = questions

    answer_dict = {}
    if answer_anno_file is not None:
        for sample in answer_anno["answers"]:
            if str(sample["question_id"]) not in answer_dict:
                answer_dict[str(sample["question_id"])] = sample["answer"]
    question_dict = {}
    for sample in question_anno["questions"]:
        if str(sample["question_id"]) not in question_dict:
            question_dict[str(sample["question_id"])] = sample["question"]
    return answer_dict, question_dict


class FindType:
    """
  Main inference class
  """

    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained(encoder)
        self.clip_processor = CLIPProcessor.from_pretrained(encoder)

    def predict(self, question):
        # Generating features
        question_clip_input = self.clip_processor(text=[question], images=Image.open(random_img_for_clip), return_tensors="pt", padding=True)
        question_clip_output = self.clip_model(**question_clip_input)

        # Generating question idxs
        question_idx = {"0": str(question_id)}

        # Answering question
        question_info = {"questions": [{"question": question, "question_id": question_id}]}

        type = Type(
            question_info, question_idx, question_clip_output.text_embeds,
        )  # Have to initialize here because necessary objects need to be generated

        return type.predict().strip()
  

def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    # Answering question
    findtype = FindType()
    answer = findtype.predict(args.question)

    return answer


if __name__ == "__main__":
    main()