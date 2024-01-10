import openai

def generate_distractors_with_openai(paragraph, question, answer, num_distractors=3, api_key='YOUR_API_KEY'):
    client = openai.OpenAI(api_key=api_key)

    prompt = f"Generate {num_distractors} plausible distractors for a question based on the following paragraph: \"{paragraph}\". The question is: \"{question}\" and the correct answer is \"{answer}\". The distractors should be semantically similar but factually incorrect."

    try:
        response = client.completions.create(
            model="text-davinci-003",  
            prompt=prompt,
            max_tokens=150,
            n=1  
        )

        raw_distractors = response.choices[0].text.strip()

        formatted_distractors = [d.strip() for d in raw_distractors.split('\n') if d.strip()]
        return formatted_distractors

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Main function to be called with paragraph, question, and answer
def main(paragraph, question, answer, api_key='YOUR_API_KEY'):
    distractors = generate_distractors_with_openai(paragraph, question, answer, api_key=api_key)
    return distractors

if __name__ == "__main__":
    sample_paragraph = "In this module, we explored a technique used to convert raw data from its numeric or categorical format to a format that can be used when performing modeling. Mathematical models mostly understand data in numeric form, hence the need for transforming raw data. During the transformation process, features should be normalized as needed to meet the assumptions of the mathematical models. Categorical variables can be transformed to numeric format using feature engineering techniques including categorical variable encoding. Bias can be introduced to the data during the feature engineering process. This bias could be conscious or unconscious. There are different kinds of biases and ways to control for these biases. Principal Component Analysis is the first introduction to modeling. It is a way of using modeling techniques for dimensionality reduction in the dataset. Now that we have completed the data understanding phase, we will transition to modeling."
    sample_question = "What is the primary use of Principal Component Analysis?"
    sample_answer = "Dimensionality reduction."


    sample_distractors = main(sample_paragraph, sample_question, sample_answer)
    print("Distractors:", sample_distractors)
