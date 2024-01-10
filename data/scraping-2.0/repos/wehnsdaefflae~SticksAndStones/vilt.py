import json

import openai
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

import elevenlabs

from llm_answers import make_element, respond, ChatResponse


def ask(question: str) -> str:
    print(f"Question: {question}")

    image = Image.open("markwernsdorfer_mgr2.jpg").convert('RGB')

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    print(f"Answer: {answer}")
    return answer


def contains_any(response: str, terms: list) -> bool:
    return any(term in response for term in terms)


def snarky_description(question: str, answer: str, previous: str) -> ChatResponse:
    data = (
        make_element(previous, "PreviousComment") +
        make_element(question, "Question") +
        make_element(answer, "Answer")
    )
    instruction = (
        "Make a witty and snarky two-sentence comment in German about the outfit described above. "
        "Continue seamlessly from where the text in the `<PreviousComment>` tag left off but do not repeat the text."
    )

    # response = respond(instruction, model="gpt-3.5-turbo", data=data, recap=recap, temperature=1.)
    response = respond(instruction, model="gpt-4", data=data, temperature=1.)
    return response


def enquire(question: str, details_dict: dict[str, str]) -> None:
    answer = ask(question)
    details_dict[question] = answer
    previous_comments = details_dict.get("_previous_comments", "")
    response = snarky_description(question, answer, previous_comments)
    audio = elevenlabs.generate(
        text=response.output,
        voice="Bella",
        model="eleven_multilingual_v2"
    )
    elevenlabs.play(audio)
    details_dict["_summary"] = response.summary
    details_dict["_previous_comments"] = previous_comments + " " + response.output
    print(f"Comment: {response.output}\n")


def clothing_details() -> dict[str, str]:
    details = dict()

    # Main Attire Details
    enquire("Is the main attire visible?", details)
    if contains_any(details["Is the main attire visible?"], ["yes", "yep", "yeah"]):
        enquire("What type of clothes are visible?", details)
        if contains_any(details["What type of clothes are visible?"], ["fancy", "formal", "business"]):
            enquire("What material or fabric does the attire look like?", details)
            enquire("Is there a suit or dress visible?", details)
            if contains_any(details["Is there a suit or dress visible?"], ["suit"]):
                enquire("How many pieces does the suit have?", details)
                enquire("What is the color of the suit?", details)
                enquire("Does the suit have any patterns or designs?", details)
            else:
                enquire("How long is the dress?", details)
                enquire("What is the color of the dress?", details)
                enquire("Does the dress have any patterns or designs?", details)
        else:
            enquire("Is there a shirt-pants combo or a single outfit visible?", details)
            if contains_any(details["Is there a shirt-pants combo or a single outfit visible?"], ["combo", "shirt and pants", "shirt-pants"]):
                enquire("What type of shirt or top is visible?", details)
                enquire("What is the color of the shirt or top?", details)
                enquire("What type of pants or skirt is visible?", details)
                enquire("What is the color of the pants or skirt?", details)
            else:
                enquire("What type of single outfit is visible?", details)
                enquire("What is the color of the single outfit?", details)

    # Footwear Details
    enquire("Is any footwear visible in the image?", details)
    if contains_any(details["Is any footwear visible in the image?"], ["yes", "yep", "yeah"]):
        enquire("What type of footwear is visible?", details)
        enquire("What is the color of the footwear?", details)
        enquire("Does the footwear have any patterns or designs?", details)

    # Jewelry and Accessory Details
    enquire("Is any jewelry visible?", details)
    if contains_any(details["Is any jewelry visible?"], ["yes", "yep", "yeah"]):
        enquire("What type of jewelry is visible?", details)
        enquire("What is the color of the jewelry?", details)

    enquire("Is any accessory like a hat, scarf, or belt visible?", details)
    if contains_any(details["Is any accessory like a hat, scarf, or belt visible?"], ["yes", "yep", "yeah"]):
        enquire("What type of accessory is visible?", details)
        enquire("What is the color of the accessory?", details)

    # Bag Details
    enquire("Is there a bag or handbag visible?", details)
    if contains_any(details["Is there a bag or handbag visible?"], ["yes", "yep", "yeah"]):
        enquire("What type of bag is visible?", details)
        enquire("What is the color of the bag?", details)
        enquire("Does the bag have any patterns or designs?", details)

    # Other Details
    enquire("Can you discern the fit of the attire?", details)
    if contains_any(details["Can you discern the fit of the attire?"], ["yes", "yep", "yeah"]):
        enquire("How does the attire fit? Snug, loose, or regular?", details)

    enquire("Can you see the sleeves of the attire?", details)
    if contains_any(details["Can you see the sleeves of the attire?"], ["yes", "yep", "yeah"]):
        enquire("What type of sleeves does it have?", details)

    enquire("Can you determine the neckline of the attire?", details)
    if contains_any(details["Can you determine the neckline of the attire?"], ["yes", "yep", "yeah"]):
        enquire("What kind of neckline does it have?", details)

    enquire("Are there any embellishments like sequins or beads on the attire?", details)
    if contains_any(details["Are there any embellishments like sequins or beads on the attire?"], ["yes", "yep", "yeah"]):
        enquire("Describe the embellishments.", details)

    enquire("Can you spot any pockets on the attire?", details)
    if contains_any(details["Can you spot any pockets on the attire?"], ["yes", "yep", "yeah"]):
        enquire("What type of pockets are visible?", details)

    enquire("Are multiple layers, like a jacket or cardigan, visible over the main attire?", details)
    if contains_any(details["Are multiple layers, like a jacket or cardigan, visible over the main attire?"], ["yes", "yep", "yeah"]):
        enquire("Describe the outer layer.", details)

    enquire("Are any hair accessories visible?", details)
    if contains_any(details["Are any hair accessories visible?"], ["yes", "yep", "yeah"]):
        enquire("What hair accessory is visible?", details)

    enquire("Is a watch visible?", details)
    if contains_any(details["Is a watch visible?"], ["yes", "yep", "yeah"]):
        enquire("Describe the watch.", details)

    return details


def main() -> None:
    openai.api_key_path = "openai_api_key.txt"

    with open("login_info.json", mode="r") as f:
        login_info = json.load(f)

    elevenlabs.set_api_key(login_info["elevenlabs_api"])

    # voices = elevenlabs.voices()
    # https://github.com/elevenlabs/elevenlabs-python
    # https://murf.ai/pricing
    # https://github.com/neonbjb/tortoise-tts

    details = clothing_details()
    print(details)
    # use the above json to generate a snarky description of someone's outfit.


if __name__ == '__main__':
    main()

