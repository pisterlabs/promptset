from src.config.config import Configuration
import openai
import os

role_prompt = """
You are an expert in medicine with a specific focus on explaining a diagnosis. You know how to analyze a set of sentences to generate an explanation for a medical condition.You will generate an accurate yet constrained explanation based on the information provided.
You will receive the set of diseases and the set of sentences containing relevant information for the explanation. Then you will generate an explanation for why * correct disease * is the most likely disease based on these key reasons. If there is enough information, explain why the other diseases are not the most likely disease.
Avoid any other response more than the explanation itself. If you are not sure about the explanation, you can say that you are not sure.
"""


class ExplanationModule:
    def __init__(self):
        self.config = Configuration()

    def generate_explanation(self, symptoms):
        explanations = []
        for symptom in symptoms:
            explanation = f"The symptom {symptom.name} "
            if symptom.unique_to_correct:
                explanation += "is unique to the correct diagnosis. "
                if symptom.high_occurrence_rate:
                    explanation += "Having a high occurrence rate, it provides strong support for the diagnosis. "
                elif symptom.low_occurrence_rate:
                    explanation += "Even though it has a low occurrence rate, it still provides unique evidence for the diagnosis. "
            elif not symptom.unique_to_correct:
                explanation += "is shared among multiple diagnoses. "
                if symptom.high_occurrence_rate:
                    explanation += "Having a high occurrence rate, it might not be distinctive enough to support the correct diagnosis. "
                elif symptom.low_occurrence_rate:
                    explanation += "Even though it has a low occurrence rate, it might still provide some evidence for the diagnosis, but its impact could be limited due to its shared nature. "
            elif symptom.correct:
                explanation += "is directly linked to the correct diagnosis. "
                if symptom.high_occurrence_rate:
                    explanation += "Having a high occurrence rate, it provides strong evidence for the diagnosis. "
                elif symptom.low_occurrence_rate:
                    explanation += "Even though it has a low occurrence rate, it still provides some evidence, but its impact could be limited. "
            elif not symptom.correct:
                explanation += "is linked to an incorrect diagnosis. "
                if symptom.high_occurrence_rate:
                    explanation += "Having a high occurrence rate, it could lead to confusion or misdiagnosis. "
                elif symptom.low_occurrence_rate:
                    explanation += "Even though it has a low occurrence rate, it might not significantly impact the diagnosis, but it's better to avoid it to prevent confusion. "

            if symptom.present_in_case:
                explanation += "This symptom is present in the clinical case, reinforcing its relevance. "
            else:
                explanation += "This symptom is not present in the clinical case, which may limit its relevance. "

            explanations.append(explanation)
        return explanations

    def generate_simpler_explanation(self, symptoms):
        explanations = []
        for symptom in symptoms:
            if symptom.unique_to_correct:
                if symptom.present_in_case:
                    if symptom.high_occurrence_rate:
                        explanations.append(
                            f"You should consider invoking the reason {symptom.name} since it is a symptom unique to the correct diagnosis, has a high occurrence rate, and it is present in the clinical case."
                        )
                    else:
                        explanations.append(
                            f"You should consider invoking the reason {symptom.name} since it is a symptom unique to the correct diagnosis and it is present in the clinical case."
                        )

            elif (
                not symptom.correct
                and symptom.high_occurrence_rate
                and not symptom.present_in_case
            ):
                explanations.append(
                    f"You should also consider invoking the reason {symptom.name} since it is a symptom with a high occurrence rate for the incorrect disease {symptom.disease}, and it does not appear in the clinical case, supporting discard {symptom.disease} as the correct diagnosis."
                )
            elif (
                symptom.low_occurrence_rate
                and not symptom.unique_to_correct
                and symptom.present_in_case
            ):
                explanations.append(
                    f"You should consider removing the reason {symptom.name} since it is a common symptom alongside all possible diagnosis."
                )
            else:
                explanations.append(
                    f"The symptom {symptom.name} does not meet the criteria for a strong reason in this case."
                )
        return explanations

    def generate_gpt_explanation(self, diseases, correct_disease, information):
        openai.api_key = os.environ["OPENAI_API_KEY"]

        prompt = f"Diseases: {diseases} \n correct_disease: {correct_disease}, \n information: {information} \n Explain why the symptoms suggest the disease is {correct_disease} rather than the others."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        # explanation = response.choices[0].text.strip()
        explanation = response["choices"][0]["message"]["content"].strip()
        return explanation
