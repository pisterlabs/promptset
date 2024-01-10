from ..items import DomainAnalyitcs
from huggingface_hub.hf_api import HfFolder
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain import PromptTemplate,  LLMChain
import transformers
import torch
import spacy

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

HfFolder.save_token('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')

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

nlp = spacy.load('en_core_web_sm')

pipeline = transformers.pipeline(
    "text-generation",
    device=0,
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_length=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    batch_size=32,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0, 'truncation':True})

class Llama2SentimentPipeline:

    def process_item(self, item, spider):
        item = DomainAnalyitcs(item)

        #item['singlePage'] = True

        # Get sentiment scores for the scraped page
        def split_text_into_parts(text, num_parts):

            # Calculate the approximate size of each part
            part_size = len(text) // num_parts

            # Split the text into equal parts
            text_parts = [text[i:i + part_size] for i in range(0, len(text), part_size)]

            return text_parts

        raw_text = item['raw']

        if item['singlePage']:
            doc = nlp(raw_text)
            phrasesArray = [sent.text.strip() for sent in doc.sents]
        else:
            # Split the text into four equal parts
            phrasesArray = split_text_into_parts(raw_text, 6)

        template = """Classify the text into joy, anger, criticism, fear, sadness, surprise, trust, enthusiasm, confusion, jealousy, calm, axiety, pride, shame, guilt, hope, excitement, gratitude, regret, grief, compassion or neutral. Reply with only one word and nothing else, for example: Joy, Anger, Criticism, Fear, Sadness, Surprise, Trust, Neutral.

                Examples:
                Text: I can't believe you would betray my trust like this, after everything we've been through. Your actions have left me seething with anger and disappointment.
                Sentiment: Anger.

                Text: Winning the championship was an absolute thrill, and I'm overflowing with happiness and excitement!
                Sentiment: Joy.

                Text: {text}
                Sentiment:"""

        template2 = """Classify the text into positive, negative or neutral. Reply with only one word and nothing else: Positive, Negative, Neutral.

                Examples:
                Text: I had a wonderful time with my friends at the beach today.
                Sentiment: Positive.

                Text: I can't believe how much money I lost in the stock market today.
                Sentiment: Negative.

                Text: The weather forecast predicts a chance of rain tomorrow.
                Sentiment: Neutral.

                Text: {text}
                Sentiment:"""

        item['llama2_sentiment'] = []
        item['llama2_posNeg'] = []

        def getTotals(sentiment, posNeg):

            for text in phrasesArray:

                if posNeg == False:
                    prompt = PromptTemplate(template=template, input_variables=["text"])
                else:
                    prompt = PromptTemplate(template=template2, input_variables=["text"])

                llm_chain = LLMChain(prompt=prompt, llm=llm)

                def classify(text):
                    raw_llm_answer = llm_chain.run(text)
                    answer = raw_llm_answer.split('.')[0]
                    return answer.lstrip()

                emotion = classify(text)
                emotionObject = {emotion: {'name': emotion.capitalize(), 'Total': 1}}
                sentiment.append(emotionObject)

            return sentiment

        item['llama2_posNeg'] = getTotals(item['llama2_posNeg'], True)
        item['llama2_sentiment'] = getTotals(item['llama2_sentiment'], False)

        def getCounts(sentiments):
            # Create a dictionary to keep track of the counts for each key
            summed_sentiments = {}

            for item in sentiments:
                key = list(item.keys())[0]
                value = item[key]
                if key in summed_sentiments:
                    summed_sentiments[key]['Total'] += value['Total']
                else:
                    summed_sentiments[key] = value

            return summed_sentiments

        item['llama2_sentiment'] = getCounts(item['llama2_sentiment'])
        item['llama2_posNeg'] = getCounts(item['llama2_posNeg'])

        return item
        

