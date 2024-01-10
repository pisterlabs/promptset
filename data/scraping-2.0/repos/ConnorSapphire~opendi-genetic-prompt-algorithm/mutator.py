# # from transformers import falconModel, falconConfig, AutoTokenizer, T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
# from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
# from langchain import LLMChain, HuggingFaceHub
from template import Template
from utils import clean_prompt, get_model, get_tokenizer
import numpy as np
import pandas as pd
import re
import torch.nn.functional as F
import pprint

class Mutator:
    """Class for extracting and mutating strings from a prompt.
    """
    
    def __init__(self, model = "t5-small", alpha = 0.1, lamda = 1, p = 0.5, threshold = 0.4, max_length = 1024, combine_prob = 0.3):
        """Initialise Mutator class.

        Args:
            model (str, optional): Name of the Large Language Model to use. Defaults to "t5-small".
            alpha (float, optional): Defaults to 0.1.
            lamda (int, optional): Defaults to 1.
            p (float, optional): Probability to use. Defaults to 0.5.
            threshold (float, optional): Defaults to 0.4.
            max_length (int, optional): Defaults to 1024.
            combine_prob (float, optional): Defaults to 0.3.

        Raises:
            ValueError: Raised by unsupported LLMs.
        """
        
        self.history = []
        self.lamda = lamda
        self.prob_select = p
        self.alpha = alpha
        self.threshold = threshold
        self.combine_prob = combine_prob
        self.seed = 0
        self.max_length = max_length
        self.supported_models = ["chatgpt"]
        if model not in self.supported_models:
            raise ValueError("Model not supported. Supported models:", self.supported_models)
        if model == "t5-base":
            checkpoint = "google/flan-t5-base"
        elif model == "t5-large":
            checkpoint = "google/flan-t5-large"
        elif model == "t5-small":
            checkpoint = "google/flan-t5-small"
        elif model == "chatgpt":
            checkpoint = "humarin/chatgpt_paraphraser_on_T5_base"
        elif model == "falcon-7b":
            checkpoint = "tiiuae/falcon-7b"
        self.model = get_model(checkpoint)
        self.tokenizer = get_tokenizer(checkpoint)
        self.encoder = get_model("", encoder = True)

        self.instruction = "You are a professional sentence rephraser. Your job is to take a given sentence, and produce a new sentence that is easier for the language models to understand.\n\nHere are some examples of how rephrasing the sentence can improve the performance of the model."
        self.demos = [
            {
                "question": "So, it is true that Lesley is a great-grandfather of Leroy",
                "answer": "Therefore Leslie is Leroy’s great-grandfather"
            }
        ]

        self.ques_start = "Rephrase the next sentence to enhance the performance of the language model."

    def clean_prompt(self, prompt: str):
        """Cleans the prompt.

        Args:
            prompt (str): Prompt in string format.

        Returns:
            str: Cleaned prompt in string format.
        """
        
        new_prompt = prompt
        new_prompt = re.sub(r"([A-Z])\.", r"\1", new_prompt)
        return new_prompt

    def split_into_sentences(self, text):
        """Use regular expressions to split the prompt into sentences.

        Args:
            text (str): Prompt in string format.

        Returns:
            list[str]: List of sentences from the string given.
        """
        
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov|edu|me)"
        digits = "([0-9])"
        multiple_dots = r'\.{2,}'

        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
        text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]: sentences = sentences[:-1]
        return sentences

    def extract_sentences(self, prompt: str, remove_tags = False):
        """Extract clean sentences from the given prompt. Removing tags if told to.

        Args:
            prompt (str): Prompt in string format.
            remove_tags (bool, optional): Determines whether to remove tags. Defaults to False.

        Returns:
            list[str]: List of clean sentences from the original prompt given.
        """
        
        new_prompt = clean_prompt(prompt)
        if remove_tags:
            new_prompt = new_prompt.replace("Question: ", "")
            new_prompt = new_prompt.replace("Answer: ", "")
        return self.split_into_sentences(new_prompt)

    def clean_sentences(self, sentences):
        """Clean sentences.

        Args:
            sentences (list[str]): List of sentences.

        Returns:
            list[str]: List of cleaned sentences.
        """
        
        sts = sentences.copy()
        sts = [s.strip() for s in sts]
        return sts

    def get_sentences(self, prompt, remove_tags = False):
        """Get cleaned sentences from a prompt. Either removing or keeping tags
        as specified.

        Args:
            prompt (str): Prompt in string format.
            remove_tags (bool, optional): Whether to keep tags in sentences. Defaults to False.

        Returns:
            list[str]: List of cleaned sentences from the prompt.
        """
        
        sentences = self.extract_sentences(prompt, remove_tags = remove_tags)
        return self.clean_sentences(sentences)

    def get_bandit_score(self, encoded_sentence, w, ainv):
        """Calculate score of an encoded sentence.

        Args:
            encoded_sentence: Encoded sentence to calculate the score of.
            w
            ainv

        Returns:
            float: Score of the encoded sentence
        """
        
        return np.dot(encoded_sentence, w) + self.alpha * np.sqrt(np.dot(encoded_sentence, np.dot(ainv, encoded_sentence)))

    def encode_sentences(self, sentences):
        """Encode the given sentences.

        Args:
            sentences (list[str]): List of sentences.

        Returns:
            list[str]: List of encoded sentences.
        """
        
        return self.encoder.encode(sentences)

    def get_demo(self, h):
        if h["score"] > 0:
            return {
                "question": h["before"],
                "answer": h["after"]
            }
        else:
            return {
                "question": h["after"],
                "answer": h["after"]
            }

    def get_sentence_distance(self, a, b):
        """Get the distance between two sentences as a measure of similarity.

        Args:
            a (str): Variant of sentence.
            b (str): Variant of sentence.

        Returns:
            float: Numerical distance between two sentences as a measure of
            similarity.
        """
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_demos(self, sentence):
        if len(self.history) < 1:
            relevant_history = []
        else:
            # print("reld1")
            encoded_sentence = self.encode_sentences(sentence)
            prev_sentences = [h["before"] for h in self.history]
            # print("reld11")
            encoded_prev_sentences = self.encode_sentences(prev_sentences)
            dist = np.array([self.get_sentence_distance(encoded_sentence, s) for s in encoded_prev_sentences])
            # print("reld112")
            indeces = np.where(dist > self.threshold)
            relevant_history = [self.history[i] for i in indeces[0]]
            # print("reld1122")
        demos = self.demos + ([self.get_demo(h) for h in relevant_history])
        # print("reld11111")
        return demos

    def concat_sentences(self, sentences):
        r = ""
        for s in sentences:
            if s.find("Options:") != -1:
                return None
            r = r + " " + s
        return r

    def combine_sentences(self, sentences, orig):
        i = 0
        n = len(sentences)
        s = []
        o = []
        while i < n:
            u = np.random.uniform()
            if u < self.combine_prob:
                r = self.concat_sentences(sentences[i: i + 2])
                ro = self.concat_sentences(orig[i: i + 2])
                if r is not None:
                    s.append(r)
                    o.append(ro)
                    i += 2
            else:
              s.append(sentences[i])
              o.append(orig[i])
              i += 1
        return s, o

    def select_sentence(self, prompt) -> int:
        """Selects a sentence to be mutated.

        Args:
            prompt (str): Prompt in string format.

        Returns:
            int: Index of selected sentence.
        """
        
        sentences_orig = self.get_sentences(prompt)
        sentences = self.get_sentences(prompt, True)
        if len(self.history) < 1:
            flag = False
            while(not flag):
                self.index = np.random.randint(0, len(sentences))
                self.before = sentences[self.index]
                flag = self.before.find("Options: ") == -1
            return self.index

        # flag = False
        # while(not flag):
        prev_sentences = [h["before"] for h in self.history]
        r = np.array([h["score"] for h in self.history])
        h = self.encode_sentences(prev_sentences)
        n = h.shape[1]
        print(h.shape)
        a = np.dot(h.T, h) + self.lamda * np.identity(n)
        # print("a1")
        ainv = np.linalg.inv(a)
        w = np.dot(np.dot(ainv, h.T), r)
        # print("a11")
        scores = np.array([self.get_bandit_score(s, w, ainv) for s in h])
        u = np.random.uniform()
        if u < self.prob_select:
            index = np.where(scores == np.max(scores))[0][0]
            # print("a111")
        else:
            index = np.random.random_integers(0, len(scores))
        self.before = sentences[index]
        self.index = index
            # flag = self.before.find("Options:") == -1
        # print("a1111")
        return index

    def mutate_sentence(self) -> str:
        """Mutates the last selected sentence to be mutated. Stored in
        self.before.

        Returns:
            str: Mutated sentence.
        """
        
        print("Selected:", self.before)
        prompt = "paraphraze: " + self.before
        print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors = "pt").input_ids
        out = self.model.generate(input_ids, max_new_tokens = 1024)
        self.after = self.tokenizer.decode(out[0], skip_special_tokens = True)
        print("Mutated:", self.after)
        return self.after

    def update_history(self, scorediff):
        """Update the history.

        Args:
            scorediff (float): Change in score caused by most recent changes.
        """
        
        self.history.append({
            "before": self.before,
            "after": self.after,
            "score": scorediff
        })


    def update_mutator(self, scorediff):
        """Update the mutator.

        Args:
            scorediff (float): Change in score caused by most recent changes.
        """
        
        self.update_history(scorediff)

    def get_mutated_prompt(self, prompt) -> str:
        """Mutates a given prompt and returns it.

        Args:
            prompt (str): Prompt in string format

        Returns:
            str: Mutated prompt in string format.
        """
        
        select_index = self.select_sentence(prompt)
        mutated_sentence = self.mutate_sentence()
        sentences = self.extract_sentences(prompt)
        if sentences[select_index].find("Question: ") != -1:
            mutated_sentence = "Question: " + mutated_sentence
            mutated_sentence = mutated_sentence.strip(".") + "?"
        elif sentences[select_index].find("Answer: ") != -1:
            mutated_sentence = "Answer: " + mutated_sentence

        sentences[select_index] = mutated_sentence
        return " ".join(sentences)

    def save_history(self, filename):
        """Save history to the given filename and location.

        Args:
            filename (str): Filename to save the history data to.
        """
        
        pd.DataFrame(self.history).to_csv(filename)

def main():
    """Main function.
    """
    # mutator = Mutator("t5-large")
    # instruction = "Answer questions about causal attribution."
    # demos = [
    #     {
    #         "question": "How would a typical person answer each of the following questions about causation?\nFrank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor's body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?\nOptions:\n- Yes\n- No\n", 
    #         "answer": "Let\'s think step by step.\nHere in this question, we are told that \"Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild.\" A typical person would assume that this passage suggests that Frank T. had no intention of shooting and injuring someone and that the bullet accidentally hit the neighbor\'s body; therefore, we conclude that Frank T. did not intentionally hit his neighbor. So the answer is No.\n\n"
    #     }, 
    #     {
    #         "question": "How would a typical person answer each of the following questions about causation?\nFrank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor's body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?\nOptions:\n- Yes\n- No\n", 
    #         "answer": "Let\'s think step by step.\nHere in this question, we are told that \"Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild.\" A typical person would assume that this passage suggests that Frank T. had no intention of shooting and injuring someone and that the bullet accidentally hit the neighbor\'s body; therefore, we conclude that Frank T. did not intentionally hit his neighbor. So the answer is No.\n\n"
    #     }
    # ]
    # question = "
    mutator = Mutator("chatgpt", combine_prob = 0.5)
    context = open("./initial_prompts/causal_judgement").read()
    t = Template("./templates/prompt/context", "./templates/demo/context").get_llm_input(question = question, context = context)
    s = mutator.extract_sentences(t, False)
    pprint(len(s))
    mutator.get_mutated_prompt(t)

if __name__ == "__main__":
    main()