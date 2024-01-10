import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
from tqdm import trange
import pronouncing
import nltk
from random import choice
import io
import re
from collections import Counter

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, temperature=1, num_samples=1, top_k=40, top_p=0.0, repetition_penalty=2.0,
                    device="cuda"):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def generate(inputText=" ", temperature=1, model="gpt2", length=1):
    # model_path = "transformers/examples/" + str(model).lower()
    MODEL_CLASSES = {
        'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
        'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
        'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
        'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
        'xlm': (XLMWithLMHeadModel, XLMTokenizer),
    }
    model_path = str(model).lower()
    model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device)
    context_tokens = tokenizer.encode(inputText, add_special_tokens=False)
    out = sample_sequence(model=model, context=context_tokens, length=length, temperature=temperature)
    out = out[:, len(context_tokens):].tolist()
    for o in out:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
#         text = text[: text.find(None)]
    return text


def replace_dash(text):
    text2 = ""
    weird_chars = ["-", "_", "(", ")", "—", "\"", ":", "[", "]", "“", "’", "‘", "”"]
    for i in text:
        if i in weird_chars:
            text2 += " "
        else:
            text2 += i
    return text2


def list_last_words(path):
    last_words = []
    with io.open(path, "r", encoding="utf-8") as f:
        for i in f:
            words = i.splitlines()
            for j in words:
                j = j.split(" ")
                if len(j[-2]) > 1:
                    last_words.append(replace_dash(j[-2]))
    return last_words


def generate_rhymewords_fromlist(rhyming_structure, possible_wordslist):
    words_rhyming_parts = []
    for word in possible_wordslist:
        word_p = pronouncing.phones_for_word(word.strip())
        for phones in word_p:
            rhyming_part = pronouncing.rhyming_part(phones)
            rhyme_part_word = [rhyming_part, word]
            words_rhyming_parts.append(rhyme_part_word)

    a2_list = []
    b2_list = []

    while len(a2_list) == 0:
        a1 = choice(words_rhyming_parts)
        for item in words_rhyming_parts:
            if item[0] == a1[0] and item[1].lower() != a1[1].lower():
                a2_list.append(item)
    while len(b2_list) == 0:
        b1 = choice(words_rhyming_parts)
        for item in words_rhyming_parts:
            if item[0] == b1[0] and item[1].lower() != b1[1].lower():
                b2_list.append(item)

    a1 = a1[1]
    b1 = b1[1]
    a2 = choice(a2_list)[1]
    b2 = choice(b2_list)[1]
    if rhyming_structure == "AABB":
        return [a1 + " ", a2 + " ", b1 + " ", b2 + " "]
    if rhyming_structure == "ABAB":
        return [a1 + " ", b1 + " ", a2 + " ", b2 + " "]
    if rhyming_structure == "ABBA":
        return [a1 + " ", b1 + " ", b2 + " ", a2 + " "]


def newline_cutoff(text):
    text = replace_dash(text)
    special_chars = ["\n", ".", ";", "!", "?"]
    sentence = ""

    for i in text:
        if i not in special_chars:
            sentence += i
        else:
            break
    if len(sentence.split()) < 3:
        sentence = ""
        temp = 0
        for i in text:
            if i not in special_chars:
                sentence += i
            elif i == ".":
                pass
            else:
                temp += 1
                if temp == 2:
                    break
    return sentence


def reverse(text):
    split = text.split(" ")
    split.reverse()
    text = ""
    for i in split:
        text += i
        text += " "
    return text


def remove_double_space(text):
    no_space = ""
    for i in range(len(text) - 1):
        if i == 0 and text[i] == " ":
            pass
        elif text[i] == " " and text[i + 1] == " ":
            pass
        elif text[i] == " " and text[i + 1] == ",":
            pass
        elif text[i] == " " and text[i + 1] == ":":
            pass
        elif text[i - 1] == "\n" and text[i] == ",":
            pass
        elif text[i - 1] == "\n" and text[i] == " ":
            pass
        else:
            no_space += text[i]
    no_space += text[len(text) - 1]
    return no_space


def poem(rhyme_words, sentiment, temperature):
    t = 0
    # assert len(rhyme_words) == len(sentiment_models)
    sentiment_model = "generic_reverse_v1"
    if sentiment == "positive":
        sentiment_model = "positive_reverse_v1"
    elif sentiment == "negative":
        sentiment_model = "negative_reverse_v1"

    rhyme_4 = rhyme_words[3]
    rhyme_3 = rhyme_words[2]
    rhyme_2 = rhyme_words[1]
    rhyme_1 = rhyme_words[0]

    line_4 = rhyme_4
    line_3 = rhyme_3
    line_2 = rhyme_2
    line_1 = rhyme_1

    excuse = "I couldn't generate a poem for you at this moment, please try again!"

    tries = 3

    ### FIX SHORT GENERATED LINES ###
    big_nono_list = ["þ", "æ", "â", "•", "\\", "/", "�", " "]

    while len(line_4.split()) < 2 or 1 in [c in line_4 for c in big_nono_list]:
        line_4 += generate(inputText=line_4, temperature=temperature, model=sentiment_model, length=25)
        line_4 = newline_cutoff(line_4)
        t += 1
        if t > tries: return excuse
        print(line_4.split(), line_4, len(line_4))
    line_4 = " ".join(line_4.split(" "))
    line_4_ins = line_4 + "\n" + rhyme_3
    t = 0
    while len(line_3.split()) < 2 or 1 in [c in line_3 for c in big_nono_list]:
        line_3 += generate(inputText=line_4_ins, temperature=temperature, model=sentiment_model, length=25)
        line_3 = newline_cutoff(line_3)
        t += 1
        if t > tries: return excuse

        print(line_3.split(), line_3, len(line_3))
    line_3 = " ".join(line_3.split(" "))
    line_3_ins = line_4 + "\n" + line_3 + "\n" + rhyme_2
    t = 0
    while len(line_2.split()) < 2 or 1 in [c in line_2 for c in big_nono_list]:
        line_2 += generate(inputText=line_3_ins, temperature=temperature, model=sentiment_model, length=25)
        line_2 = newline_cutoff(line_2)
        t += 1
        if t > tries: return excuse

        print(line_2.split(), line_2, len(line_2))
    line_2 = " ".join(line_2.split(" "))
    line_2_ins = line_4 + "\n" + line_3 + "\n" + line_2 + "\n" + rhyme_1
    t = 0
    while len(line_1.split()) < 2 or 1 in [c in line_1 for c in big_nono_list]:
        line_1 += generate(inputText=line_2_ins, temperature=temperature, model=sentiment_model, length=25)
        line_1 = newline_cutoff(line_1)
        t += 1
        if t > tries: return excuse

        print(line_1.split(), line_1, len(line_1), "\n\n\n")
    line_1 = " ".join(line_1.split(" "))
    combined_normal = reverse(line_1) + "\n" + reverse(line_2) + "\n" + reverse(line_3) + "\n" + reverse(line_4)

    check_1 = remove_double_space(combined_normal)
    check_2 = remove_double_space(check_1)  # filter out created "\n" + " " occurences
    check_3 = remove_double_space(check_2)  # Better to be safe than sorry

    return check_3.lower()

def GeneratePoem(rhyme_struct, sentiment, temp_in):
    temperature = 0.5 + temp_in * 0.1
    possible_wordslist = list_last_words("poems_final.txt")

    newlist = Counter(possible_wordslist).most_common(3000)
    undesirable_chars_in_lastwords = ["\\", " ", "+", "*", "\"", "\'"]
    k = []
    for i in range(len(newlist)):
        if not [e in newlist[i][0] for e in undesirable_chars_in_lastwords if e in newlist[i][0]]:
            k.append(newlist[i][0])

    rhyme_words = generate_rhymewords_fromlist(rhyme_struct, k)
    poem_raw = poem(rhyme_words, sentiment, temperature).replace(u'\xa0', u' ').replace(u' amp ', u' ').replace(u' s ', ' ')
    poem_final = re.sub(" +", " ", poem_raw)
    return poem_final