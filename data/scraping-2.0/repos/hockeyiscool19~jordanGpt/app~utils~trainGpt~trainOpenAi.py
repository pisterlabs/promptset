from PUBLIC_VARIABLES import OPENAI_API_KEY
import openai


if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    res = openai.File.create(file=open(
        r"app\utils\data\trainData.jsonl", "r"), purpose='fine-tune', n_epochs=3)
