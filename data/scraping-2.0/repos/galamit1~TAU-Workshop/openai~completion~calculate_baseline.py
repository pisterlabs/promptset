from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from chat_gpt import chat_with_gpt
import openai

# TODO add api keys:
# openai.organization = ""
# openai.api_key = ""

INITIAL_INSTRUCTIONS = "I'd like you to classify the following tweets if it's a sarcasm or not, print 1 if you think it's sarcasm and 0 if not. the tweets split with \\n, write the results on after the other, for example the output can be 0100011"
INITIAL_INSTRUCTIONS = """I'd like you to classify the following tweets as sarcastic or not sarcastic. The tweet is sarcastic if it uses irony to mock or convey contempt or if the tweet suggests an alternative meaning that differs from the literal meaning of the words in the tweet.
Here is an example of a sarcastic tweet:
"Yea! great product! If you want to lose all your data in a year and be forced to pay to restore ..."
There are 20 tweets that split with \\n
Write the 20 results for each tweet as "Sarcastic" or "Not Sarcastic" by order of the input tweets and separated by commas. Do not print anything else."""

TEST_PATH = "validation_labeled_2000_balanced.csv"

CHAIN_OF_THOUGHTS = True

TWEETS_IN_A_BUNCH = 20
NUMBER_OF_BUNCHES = 50



def get_predictions(ds):
    predictions = []
    for i in range(NUMBER_OF_BUNCHES * 2):
        tweets_done = i * TWEETS_IN_A_BUNCH
        print("classified {} tweets out of {}".format(tweets_done, TWEETS_IN_A_BUNCH * NUMBER_OF_BUNCHES))
        tweets_input = '\n'.join([str(tweet) for tweet in ds["tweets"][tweets_done: tweets_done + TWEETS_IN_A_BUNCH]]) # assume the tweets are without \n
        try:
            response = chat_with_gpt(INITIAL_INSTRUCTIONS, str(tweets_input))
            parsed_response = [int(res.strip() == 'Sarcastic') for res in response.split(',')]
            print(len(parsed_response))
            if len(parsed_response) != TWEETS_IN_A_BUNCH:
                # try again
                response = chat_with_gpt(INITIAL_INSTRUCTIONS, str(tweets_input))
                parsed_response = [int(res.strip() == 'Sarcastic') for res in response.split(',')]
                print(len(parsed_response))
                if len(parsed_response) != TWEETS_IN_A_BUNCH:
                    # skip this bunch
                    continue
            predictions += parsed_response
            print(len(predictions))
            if len(predictions) == TWEETS_IN_A_BUNCH * NUMBER_OF_BUNCHES:
                break
        except Exception as e:  # in case of an error, continue with what we have so far
            print("stopped with an error: " + str(e))
            break
    return predictions


def calculate_base_line(ds):
    y_pred = get_predictions(ds)
    y_true = [i.as_py() for i in ds["class"][:len(y_pred)]]

    # Calculate Precision
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    # Calculate Recall
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    # Calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print("F1-Score:", f1)

    # Calculate Support (this doesn't require scikit-learn)
    # It's simply the count of true instances for each class.
    support = [y_true.count(0), y_true.count(1)]
    print("Support for each class:", support)

    # Generate a classification report (includes precision, recall, and f1-score)
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:\n", class_report)

    # Generate a confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", confusion)


def main():
    ds = load_dataset("csv", data_files=TEST_PATH, sep=",")
    if CHAIN_OF_THOUGHTS:
        with open('sarcasm_reasoning_train_for_chain_of_thoughts2.csv', 'r') as f:
            COT_prompt = f.read()
        print(chat_with_gpt("", COT_prompt))
    calculate_base_line(ds['train'].data)


if __name__ == '__main__':
    main()
