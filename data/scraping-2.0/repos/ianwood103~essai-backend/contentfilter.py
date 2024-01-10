import openai
import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
list_swears = []
with open("swear_words.txt") as words_file:
        print("successfully opened")
        line = words_file.readline().strip()
        # print(line)
        while line:
            # print(line)
            list_swears.append(line)
            line = words_file.readline().strip()

def is_too_toxic(content_to_classify):
    split_sentence = content_to_classify.split(" ")
    # print(split_sentence)
    overlap = list(filter(lambda word: word in list_swears, split_sentence))
    # print(list(overlap))
    if overlap:
        return True
    # return False
    # exit()
    # handle filter not finding anything
    response = openai.Completion.create(
      engine="content-filter-alpha",
      prompt = "<|endoftext|>"+content_to_classify+"\n--\nLabel:",
      temperature=0,
      max_tokens=1,
      top_p=0,
      logprobs=10
    )
    output_label = response["choices"][0]["text"]
    # This is the probability at which we evaluate that a "2" is likely real
    # vs. should be discarded as a false positive
    toxic_threshold = -0.355

    if output_label == "2":
        # If the model returns "2", return its confidence in 2 or other output-labels
        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # If the model is not sufficiently confident in "2",
        # choose the most probable of "0" or "1"
        # Guaranteed to have a confidence for 2 since this was the selected token.
        if logprobs["2"] < toxic_threshold:
            logprob_0 = logprobs.get("0", None)
            logprob_1 = logprobs.get("1", None)

            # If both "0" and "1" have probabilities, set the output label
            # to whichever is most probable
            if logprob_0 is not None and logprob_1 is not None:
                if logprob_0 >= logprob_1:
                    output_label = "0"
                else:
                    output_label = "1"
            # If only one of them is found, set output label to that one
            elif logprob_0 is not None:
                output_label = "0"
            elif logprob_1 is not None:
                output_label = "1"

            # If neither "0" or "1" are available, stick with "2"
            # by leaving output_label unchanged.

    # if the most probable token is none of "0", "1", or "2"
    # this should be set as unsafe
    if output_label not in ["0", "1", "2"]:
        output_label = "2"
    return output_label not in ["0", "1"]

if __name__ == "__main__":
    print(is_too_toxic("something to"))
    print()
    print(is_too_toxic("piss"))
    print()
    print(is_too_toxic("shit dick"))