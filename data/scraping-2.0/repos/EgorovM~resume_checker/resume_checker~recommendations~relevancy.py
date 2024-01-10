import numpy as np
from openai import OpenAI

from sentence_transformers import CrossEncoder

from resume_checker.recommendations.prompts import resume_prompt, resume_parser
from resume_checker.settings import OPENAI_API_KEY, CROSS_ENCODER_PATH

client = OpenAI(
    api_key=OPENAI_API_KEY,
)
cross_encoder_model = CrossEncoder(CROSS_ENCODER_PATH, max_length=512)


def get_score_and_edits(vacancy_text: str, resume_text: str) -> dict:
    prompt = resume_prompt.format_prompt(vacancy=vacancy_text, resume=resume_text)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": prompt.to_string()},
            {"role": "user", "content": "Оцени предоставлееное резюме пожалуйста"}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content.strip()
    score = 1 / (1 + np.exp(-cross_encoder_model.predict([resume_text, vacancy_text])[1]))

    result: dict = dict(resume_parser.parse(output))
    result.update(score=score)

    return result

