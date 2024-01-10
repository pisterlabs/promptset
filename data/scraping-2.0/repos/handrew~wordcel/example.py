import pandas as pd
from wordcel.featurize import apply_io_bound_function
from wordcel.llm_providers import openai_call


def main():
    """Main function for testing."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "I love this product! It's amazing.",
            "The service was terrible. I'm very disappointed.",
            "The weather today is just perfect.",
            "This movie is fantastic. I highly recommend it.",
            "I had a bad experience with this company's customer support.",
        ],
    }
    df = pd.DataFrame(data)
    print(df)

    def sentiment_classify(text):
        prompt = f"Classify the sentiment of the following text into one of two categories, POS or NEG. Respond in one word only.\n\n{text}"
        return openai_call(prompt, model="gpt-3.5-turbo", max_tokens=32)

    results = apply_io_bound_function(
        df,
        sentiment_classify,
        text_column="text",
        id_column="id",
        num_threads=4,
        cache_folder="cache",
    )
    print(results)
    joined_results = df.join(results.set_index("id"), on="id")
    print()
    print(joined_results)


if __name__ == "__main__":
    main()

