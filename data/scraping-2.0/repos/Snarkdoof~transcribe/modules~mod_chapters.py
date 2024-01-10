#!/usr/bin/env python3

import json
import copy
import time
import os

try:
    from sentence_transformers import SentenceTransformer, util
except:
    print("Can't run in this environment")

try:
    import openai
    import numpy as np
    useOpenAI = True
    if os.path.exists(".openai_key"):
        with open(".openai_key", "r") as f:
            openai.api_key = f.read()
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    useOpenAI = False
    print(" **** WARNING *** OpenAI not availble")

"""
pip install keybert
pip install autofaiss sentence-transformers
"""


# from autofaiss import build_index
import numpy as np


ccmodule = {
    "description": "Create chapters and summarize them using OpenAI (optional)",
    "depends": [],
    "provides": [],
    "inputs": {
        "src": "Subtitle file (json)",
        "dst": "Destination file (json), default append _chapters.json to src",
        "min_similarity": "Minimum similarity, 0-1, default 0.4",
        "summarize": "Use OpenAI (GPT) to summarize chapters, default false",
        "lang": "Target language for summaries, default 'no'",
        "openai_cache": "Cache file to avoid running identical prompts multiple times, default /tmp/openai_cache.json",
        "openai_api_key": "OpenAI API Key"
    },
    "outputs": {
        "dst": "Chapter file (json)",
        "freshness": "Freshness file (json) - does a person develop speech?"
    },
    "defaults": {
        "priority": 50,  # Normal
        "runOn": "success"
    },
    "status": {
        "progress": "Progress 0-100%",
        "state": "Current state of processing"
    }
}


DEBUG = False
MAX_CHARS = 7300 

prompts = {
    "summary": {
        "no": "oppsummer teksten på norsk",
        "en": "summarize the text in english"
    },
    "summary_simple": {
        "no": "oppsummer i en setning på norsk så barn skjønner teksten",
        "en": "summarize the text in english for a 6 year old"
    },
    "questions": {
        "no": """Svar på norsk:
1. Hva snakker de om?
2. Hvem er for?
3. Hvem er i mot?
4. Hvem sier mest nytt?
Teksten:
""",
        "en": """Answer in english:
1. What are they discussing?
2. Who is for it?
3. Who is against it?
4. Who adds most to the conversation?
The text:
"""
    }
}
    # "summary_short": "oppsummer teksten kort på norsk",


class GroupModel:
    instance = None

    @staticmethod
    def get(model="NbAiLab/nb-sbert"):
        if not GroupModel.instance:
            GroupModel.instance = SentenceTransformer(model)
        return GroupModel.instance


class GroupSentences:

    def __init__(self, model, cache_file):
        self.sentences = {}
        self.model = model
        self._cache = {}
        self.cache_file = cache_file

        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    self._cache = json.load(f)
                except:
                    print("Invalid cache, removing it")
                    os.path.remove(self.cache_file)

    def add_sentences(self, sentences):
        new_sentences = []
        for sentence in sentences:
            if sentence in self.sentences:
                continue 
            new_sentences.append(sentence)

        # Calculate new embeddings
        embeddings = self.model.encode(new_sentences)
        for idx, embedding in enumerate(embeddings):
            self.sentences[new_sentences[idx]] = embedding

        return len(new_sentences)

    def get_similarity(self, sentence_1, sentence_2):
        # Do we already have these sentences in the 'cache'
        self.add_sentences([sentence_1, sentence_2])
        return util.cos_sim(self.sentences[sentence_1], self.sentences[sentence_2])

    def find_best_match(self, sentence, candidates=[]):
        s = copy.copy(candidates)
        s.append(sentence)
        self.add_sentences(s)

        if not candidates:
            candidates = self.sentences

        scores = []
        # print("Checking", summarized[idx]["text"])
        for candidate in candidates:
            scores.append(util.cos_sim(self.sentences[candiate], self.sentences[sentence]))

    def get_similarity_collect(self, sentence, candidates):
        target_text = "\n".join(candidates)
        return self.get_similarity(sentence, target_text)

    def get_similarity_multi(self, sentence, candidates=[]):
        s = copy.copy(candidates)
        s.append(sentence)
        self.add_sentences(s)

        if not candidates:
            candidates = self.sentences

        scores = []
        # print("Checking", summarized[idx]["text"])
        for candidate in candidates:
            scores.append(util.cos_sim(self.sentences[candiate], self.sentences[sentence]))

        if len(scores) == 0:
            return None, None, None
        return min(scores), sum(scores) / len(scores), max(scores)

    def detect_chapters(self, summarized, max_char_length=0, min_similarity=0.4,
                        cache_file=None):
        """
        If max_char_length is given, we create new chpaters to stay within
        """
        min_sentences = 5  # Each chapter is at least this many sentences
        # min_similarity = 0.30  # Works for GMN  

        min_sentence_length = 6  # Ignore sentences that are shorter than this many words
        switch_after = 8  # Switch chapter after x following sentences that are different
        current_chapter = []
        num_new = 0
        startts = 0
        first_idx_new_chapter = 0
        new_chapter = []
        chapters = []

        for idx, item in enumerate(summarized):
            sentence = item["text"]
            if len(sentence) < min_sentence_length:
                print( "*** Dropping too short", sentence)
                continue

            if len(current_chapter) == 0:
                current_chapter.append(sentence)
                continue

            # How similar is this sentences to the current chapter?
            score = self.get_similarity_collect(sentence, current_chapter)

            if DEBUG:
                print("%.2f: %s" % (score, sentence))

            # if max(scores) < min_similarity:
            if max_char_length and len(" ".join(current_chapter)) + len(sentence) > max_char_length:
                print(len(chapters) + 1, "Chapter would be to long, faking score 0")
                score = 0

            if score < min_similarity:  #  and max_score < (min_similarity * 1.5):
                new_chapter.append(sentence)
                if num_new == 0:
                    first_idx_new_chapter = idx
                num_new += 1
                if DEBUG:
                    print("   Suspect nr", num_new, "chapter idx", first_idx_new_chapter)
                if num_new >= switch_after or score == 0:
                    if DEBUG:
                        print("**** Changing subjects ****")
                        print("    Chapter from", startts, "to", summarized[first_idx_new_chapter - 1]["end"])

                    chapters.append({"start": startts, "end": summarized[first_idx_new_chapter - 1]["end"]})

                    startts = summarized[first_idx_new_chapter - 1]["end"]  # Go back to the first non-hit
                    current_chapter = new_chapter
                    new_chapter = []
                    num_new = 0
            else:
                num_new = 0
                # Part of the chapter
                new_chapter = []
                current_chapter.append(sentence)


        # There probably is a chapter here now too
        if len(current_chapter) > 0:
            chapters.append({"start": startts, "end": summarized[-1]["end"]})


        # Summarize the chapters
        print("Collect chapters")
        for chapter in chapters:
            s = []
            puretext = ""
            for item in summarized:
                if item["start"] < chapter["end"] and item["end"] > chapter["start"]:
                    s.append("%s: %s" % (item["who"], item["text"]))
                    puretext += item["text"] + " "
            chapter["text"] = "\n".join(s)
        return chapters

    # TODO: Double check to see if any of the adjoined chapters are about the same
    # thing?

    def summarize(self, item, only_summary=False, reprocess=False,
                  model="text-davinci-003", lang="no"):
        """
        text-davinci-003 is the best, slowest and most expensive
        text-curie-001 - doesn't seem to do much
        """
        sourcetext = item["text"][:MAX_CHARS]

        ret = {}
        # Normal 
        for key in prompts:
            if not reprocess and key in item:
                ret[key] = item[key]
                continue  # Already have this one

            print("    ", key)
            prompt = '%s: "%s"' % (prompts[key][lang], sourcetext)

            if key == "summary_simple":
                prompt = '%s: "%s"' % (prompts[key][lang], ret["summary"]["text"])

            for i in range(4):
                try:
                    if prompt not in self._cache:
                        response = openai.Completion.create(model=model,
                            prompt=prompt, temperature=0, max_tokens=1000)

                        t = response["choices"][0]["text"].replace("\n\n", "")
                        ret[key] = {
                            "text": t, 
                            "quality": int(100 * float(self.get_similarity(t, sourcetext)))
                        }
                        self._cache[prompt] = ret[key]

                        if self.cache_file:
                            with open(self.cache_file, "w") as f:
                                json.dump(self._cache, f)
                        
                    else:
                        ret[key] = self._cache[prompt]

                    print("Prompt:", prompt)
                    print("response", response)
                    # TODO: Check if we got the full message, otherwise try to get the rest.
                    # if response["choices"][0]["reason"] == "length":
                    #    print("Woops, we should fix this")
                    #    raise SystemExit(-1)
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Exception, retrying in 10 seconds")
                    time.sleep(10)
        return ret

    def recurse_summary(self, summaries, is_recursive=False, lang="no"):
        """
        Create a single summary of summaries - if the summaries are too long
        put together, it will do it recursively until only one is present
        """
        print("Recurse summary of %d summaries" % len(summaries))
        ret = []
        text = ""
        def __summarize(text, start, end, type="summary"):
            print("Summarize from", start, end, type)
            prompt = "%s: %s" % (prompts[type][lang], text)
            for i in range(4):
                try:
                    response = openai.Completion.create(model="text-davinci-003",
                        prompt=prompt, temperature=0, max_tokens=1000)

                    # TODO: If the reason for stopping is not that it's done, we will get the rest of the result on the next call to openai
                    s = response["choices"][0]["text"].replace("\n\n", "")                    
                    return {
                        "start": start,
                        "end": end,
                        "summary": s,
                        "quality": self.get_similarity(s, text)
                    }
                except Exception as e:
                    print("Oops, retrying", e)
                    time.sleep(10)

        start = summaries[0]["start"]
        end = summaries[0]["end"]
        for summary in summaries:
            if len(text) + len(summary["summary"]) > MAX_CHARS:
                print("  ", start, end)
                ret.append(__summarize(text, start, end))
                text = ""
                start = summary["start"]

            text += summary["summary"] + "\n"
            end = summary["end"]

        end = summaries[-1]["end"]
        if (text):
            print("  -- last one")
            ret.append(__summarize(text, start, end))

        if len(ret) > 1:
            return self.recurse_summary(ret, is_recursive=True)

        if len(ret) == 0:
            print("  **** ERROR: No summary at all!")
            return {}

        r = []
        if is_recursive:
            r = summaries

        start = ret[0]["start"]
        end = ret[0]["end"]
        r.append({
            "start": start,
            "end": end,
            "summary": ret[0]["summary"],
            "summary_short": __summarize(ret[0]["summary"], start, end, "summary_short")["summary"],
            "summary_simple": __summarize(ret[0]["summary"], start, end, "summary_simple")["summary"]
        })
        return r

    def collect_subtitles(self, subs):
        """
        Summarize subtitles pr. person. Basically just
        group the texts if they have the same speaker.

        Returns equivalent to subs, with start, end and text tags collected
        """

        summarized = []
        for item in subs:
            if len(summarized) == 0 or item["who"] != summarized[-1]["who"]:
                summarized.append(copy.copy(item))
                continue

            # Same user still
            summarized[-1]["text"] += " " + item["text"]
            summarized[-1]["end"] = item["end"]

        return summarized

    def detect_new_info(self, sentences, chapters):
        """
        Go through the sentences and check for each person if the sentence
        they speak in each chapter bring anything new or just rehash the same
        old stuff
        """

        freshness = {}
        people_sentences = {}
        for sentence in sentences:

            who = sentence["who"]
            if who not in freshness:
                freshness[who] = []
            if who not in people_sentences:
                people_sentences[who] = [sentence["text"]]
                continue

            # What's the difference from earlier statements
            score = self.get_similarity_collect(sentence["text"], people_sentences[who])
            freshness[who].append({
                "start": sentence["start"],
                "end": sentence["end"],
                "score": int(100 * (1 - float(score))),
                "who": who,
                "text": sentence["text"]
            })
            people_sentences[who].append(sentence["text"])

        if 0:
        #for chapter in chapters:
            people_sentences = {}
            for sentence in sentences:
                if sentence["start"] > chapter["end"] or sentence["end"] < chapter["start"]:
                    continue  # Not part of this chapter

                who = sentence["who"]
                if who not in freshness:
                    freshness[who] = []
                if who not in people_sentences:
                    people_sentences[who] = [sentence["text"]]
                    continue

                # What's the difference from earlier statements
                score = self.get_similarity_collect(sentence["text"], people_sentences[who])
                freshness[who].append({
                    "start": sentence["start"],
                    "end": sentence["end"],
                    "score": int(100 * (1 - float(score))),
                    "who": who,
                    "text": sentence["text"]
                })
                people_sentences[who].append(sentence["text"])

        return freshness

    def add_speakers(self, sentences, chapters):
        """
        Go through sentences and add speakers to the chapter in the format
        {"speaker": num_characters}
        """

        for chapter in chapters:
            speakers = {}
            for sentence in sentences:
                if sentence["who"] == None or sentence["who"] == "null":
                    continue

                if sentence["start"] > chapter["end"] or \
                   sentence["end"] < chapter["start"]:
                   continue

                if sentence["who"] not in speakers:
                    speakers[sentence["who"]] = len(sentence["text"])
                else:                    
                    speakers[sentence["who"]] += len(sentence["text"])

            chapter["speaking"] = speakers

        return chapters

def process_task(cc, task, stop_event):

    args = task["args"]

    src = args["src"]
    dst = args.get("dst", None)
    min_similarity = args.get("min_similarity", 0.4)
    summarize = args.get("summarize", False)
    lang = args.get("lang", "no")
    if not lang:
        lang = "no"  # might be "" or None, we don't want that

    openai_cache = args.get("openai_cache", "/tmp/openai_cache.json")
    if useOpenAI:
        openai.api_key = args.get("openai_api_key", None)

    if not dst:
        dst = os.path.splitext(src)[0] + "_chapters.json"

    dst_dir = os.path.split(dst)[0]
    bn = os.path.splitext(os.path.basename(src))[0]
    dst_freshness = os.path.join(dst_dir, bn + "_freshness.json")

    if summarize:
        if not useOpenAI:
            raise Exception("Missing openai, use pip install on worker nodes")
        if not openai.api_key:
            raise Exception("Missing OpenAI API Key")

    if lang not in prompts["summary"]:
        raise Exception("Unsupported language '%s', Supported languages:" % lang,
                        list(prompts["summary"].keys()))

    # Do actual work
    cc.status["progress"] = 0
    with open(src, "r") as f:
        source = json.load(f)

    groupie = GroupSentences(GroupModel.get(), cache_file=openai_cache)
    cc.status["progress"] = 15
    collected = groupie.collect_subtitles(source)
    sentences = [item["text"] for item in collected]
    groupie.add_sentences(sentences)
    cc.status["progress"] = 30

    # We detecth chapters and stay within the max limit for OpenAI with a little margin for
    # the added "name:" bits
    chapters = groupie.detect_chapters(collected, max_char_length=MAX_CHARS * 0.80,
                                       min_similarity=min_similarity,
                                       cache_file=openai_cache)
    cc.status["progress"] = 40

    groupie.add_speakers(source, chapters)
    cc.status["progress"] = 50

    # Try to detect how "new" each person's sentences are
    freshness = groupie.detect_new_info(collected, chapters)

    if summarize:
        cc.status["progress"] = 60
        percent_pr_chapter = 30.0 / len(chapters)
        for idx, chapter in enumerate(chapters):
            chapter["summary_quality"] = {}
            summaries = groupie.summarize(chapter, lang=lang)
            chapter.update(summaries)
            cc.status["progress"] = 60 + ((idx + 1) * percent_pr_chapter)

    cc.status["progress"] = 90

    try:
        # Go through the summaries of different people, summarize how active each speaker is?
        for i, chapter in enumerate(chapters):
            print("*** CHAPTER %d ****" % i, chapter["start"], "-", chapter["end"])
            # print(chapter["summary_short"])
            print(" --------------------------- ")
    finally:
        if dst:
            with open(dst, "w") as f:
                json.dump(chapters, f, indent=" ")
        if summarize and freshness:
            with open(dst_freshness, "w") as f:
                json.dump(freshness, f, indent=" ")

    cc.status["progress"] = 100

    return 100, {"dst": dst, "freshness": dst_freshness}


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", dest="src", help="Source file", required=True)

    parser.add_argument("-o", "--output", dest="dst",
                        help="Destination file")

    parser.add_argument("--similarity", dest="min_similarity",
                        help="Minimum similarity for sentences of a chapter, 0-1, " + \
                             "higher numbers = more similar, 0.4 is default", 
                        default=0.40)

    parser.add_argument("--maxlen", dest="maxlen", help="Max length of chapter in minutes", 
                        default=15)

    parser.add_argument("--summarize", dest="summarize", help="Use GPT-3 to summarize", 
                        action="store_true", default=False)

    parser.add_argument("--lang", dest="lang", help="Target language for summaries", 
                        default="no")

    options = parser.parse_args()

    if options.lang not in prompts["summary"]:
        raise SystemExit("Supported languages:", list(prompts["summary"].keys()))

    if 1 and useOpenAI and not openai.api_key:
        raise Exception("Missing OpenAI key, use OPEN_API_KEY environment variable or put it in .openai_key")

    options.min_similarity = float(options.min_similarity)
    options.maxlen = float(options.maxlen) * 60

    with open(options.src, "r") as f:
        source = json.load(f)

    print("Loading model")
    loaded_model = SentenceTransformer("NbAiLab/nb-sbert")
    groupie = GroupSentences(loaded_model, options)

    print("Loading sentences")
    collected = groupie.collect_subtitles(source)
    print("   %d sentences turned into %d collected sentences" % (len(source), len(collected)))
    sentences = [item["text"] for item in collected]

    print("Encoding sentences")
    groupie.add_sentences(sentences)

    print("Detecting chapters")

    # We detecth chapters and stay within the max limit for OpenAI with a little margin for
    # the added "name:" bits
    chapters = groupie.detect_chapters(collected, max_char_length=MAX_CHARS * 0.80)

    print("    %d chapters" % len(chapters))

    print("Adding speakers")
    groupie.add_speakers(source, chapters)


    for i, chapter in enumerate(chapters):
        if len(chapter["text"]) > MAX_CHARS:
            print("  --- Chapter %d too long: %d" % (i, len(chapter["text"])))

    # Try to detect how "new" each person's sentences are
    print("Detecting freshness")
    freshness = groupie.detect_new_info(collected, chapters)

    if options.summarize:
        print("Summarizing chapters")
        for idx, chapter in enumerate(chapters):
            print("   Chapter", idx)
            chapter["summary_quality"] = {}
            summaries = groupie.summarize(chapter)
            chapter.update(summaries)

    try:
        if 1:
            # Go through the summaries of different people, summarize how active each speaker is?
            print("Summarizing chapter")
            for i, chapter in enumerate(chapters):
                print("*** CHAPTER %d ****" % i, chapter["start"], "-", chapter["end"])
                # print(chapter["summary_short"])
                print(" --------------------------- ")
    finally:
        if options.dst:
            print("Saving to", options.dst)
            with open(options.dst, "w") as f:
                json.dump(chapters, f, indent=" ")

            with open("/tmp/freshtest.json", "w") as f:
                json.dump(freshness, f, indent=" ")
