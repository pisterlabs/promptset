import openai

# Get the OpenAI API key by signing up on OpenAI.
openai.api_key = 'YOUR_API_KEY'
class CategorizingService: 
    def categorize_text(self, categories, text_samples):
        labels = []
        text_label_mapping = {}

        # String of categories in which you want to classify the text.
        category_str = ", ".join(map(str, categories))

        # Sample Prompt
        for i in range(len(text_samples)):
            prompt = f"{text_samples[i]}; Classify this sentence as {category_str} in one word."
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content":  f"Classify the following sentence into the category: {category_str}\n{text_samples[i]}"},
                        ],
                max_tokens=50
            )

            label = response.choices[0].message.content.strip(".")
            labels.append(label)
            text_label_mapping[text_samples[i]] = label
        return labels, text_label_mapping

# Example usage
categorizing_service = CategorizingService()
categories = ["Productivity", "Teamwork"]
text_samples = ["She was always a late replier and neglected my messages"]

# Call the method with the desired arguments
category_labels = categorizing_service.categorize_text(categories, text_samples)
print("Category Labels:", category_labels)
