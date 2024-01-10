import src.func_gpt as func
import pandas as pd

# Import openai key
func.set_api_key()

# load data
df = pd.read_csv("data/train_HF_df.csv")


# Test the paraphrase_text function
text_to_paraphrase = df["text"].tolist()

# getting the paraphrased text
paraphrase = func.paraphrase_text_list(text_to_paraphrase)

# append the paraphrased text to the df
df["paraphrased_text"] = paraphrase

# save the df
df.to_csv("data/train_paraphrased_with_gpt4_HF_df_new_prompt.csv")

# Test the df_with_original_and_paraphrased_text function
#df = func.df_with_original_and_paraphrased_text(text_to_paraphrase, paraphrase)

# semantic similarity
#similarity = func.semantic_similarity(df)
#print("semantic similarity calculated")

# print the df
#print(df)