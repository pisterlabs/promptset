import json

import cv2
import openai
from torch.nn import functional
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

import elevenlabs

from llm_answers import make_element, respond, ChatResponse


def get_image() -> Image:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Couldn't open the webcam. What a surprise!")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Couldn't grab the photo. Again, what a surprise!")
        exit()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    cap.release()

    return pil_image


def get_vilt_qna_model() -> ViltForQuestionAnswering:
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to("cuda")
    return model


def get_vilt_qna_processor() -> ViltProcessor:
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return processor


def ask_model(image: Image, question: str, processor: ViltProcessor, model: ViltForQuestionAnswering) -> str:
    inputs = processor(image, question, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = functional.softmax(logits, dim=-1)
    values, indices = probabilities.topk(10)
    answer = model.config.id2label[indices[0][0].item()]
    return answer


def yes_no_question(image: Image, question: str, processor: ViltProcessor, model: ViltForQuestionAnswering) -> bool:
    answer = ask_model(image, question, processor, model)
    return "yes" in answer.lower()


def main() -> None:
    processor = get_vilt_qna_processor()
    model = get_vilt_qna_model()

    image = get_image()

    while True:
        question = input("Question: ")

        inputs = processor(image, question, return_tensors="pt").to("cuda")

        outputs = model(**inputs)

        logits = outputs.logits

        # convert logits to probabilities using softmax
        probabilities = functional.softmax(logits, dim=-1)

        # get top 10 values and their indices
        values, indices = probabilities.topk(10)

        # print the top 10 answers along with their probabilities
        for value, idx in zip(values[0], indices[0]):
            print(f"Predicted answer: {model.config.id2label[idx.item()]} with probability {value.item():.2%}")

        image.show()


if __name__ == '__main__':
    main()

