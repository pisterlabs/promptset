"""
Compute automated topic coherence using gensim.
"""
import argparse
import os
import pickle
import sys
import time
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

sys.path.append("../")
from utils import str_to_bool


def main():
    parser = argparse.ArgumentParser(description="Compute coherence from topics text file")
    parser.add_argument("--topics_path", type=str, required=True, help="path to topics file")
    parser.add_argument("--vocab_path", type=str, required=True, help="path to vocab pickle file")
    parser.add_argument(
        "--texts_dir", type=str, required=True, help="path to tokenized text dir (uses all files inside)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        nargs="+",
        required=True,
        help="list of topn words in topics to use (averaged to compute agg. coherence)",
    )
    parser.add_argument("--to_lower", type=str_to_bool, default=False, help="whether to ignore casing")
    parser.add_argument("--coherence_type", type=str, default="c_npmi", help="coherence type")
    parser.add_argument("--window_size", type=int, default=10, help="window size for coherence")
    parser.add_argument("--out_name", type=str, default=None, help="name to append to output file")
    args = parser.parse_args()

    print("*" * 10)
    print(args)
    print("*" * 10)

    with open(args.topics_path, "r") as f:
        topics = [line.strip().split() for line in f.readlines()]

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    coherence_texts = []
    fnames = []

    for fname in os.listdir(args.texts_dir):
        if fname.endswith(".txt"):
            fnames.append(fname)

            with open(os.path.join(args.texts_dir, fname), "r") as f:
                coherence_texts.extend([line.strip().split() for line in f.readlines()])

    print(f"Loaded {len(coherence_texts)} coherence texts from {len(fnames)} files: {fnames}")

    if args.to_lower:
        print("Lowercasing topic, coherence texts, vocab...")
        topics = [[word.lower() for word in topic] for topic in topics]
        coherence_texts = [[word.lower() for word in text] for text in coherence_texts]
        token2id = dict([(k, i) for i, k in enumerate(set([k.lower() for k in vocab["TM_vocab"]]))])
    else:
        token2id = dict([(k, i) for i, k in enumerate(vocab["TM_vocab"])])

    coherence_dict = Dictionary()
    coherence_dict.token2id = token2id
    coherence_dict.id2token = {v: k for k, v in coherence_dict.token2id.items()}

    print("Computing coherences...")
    all_coherences = [[] for _ in range(len(topics))]
    start = time.time()

    for top_n in args.top_n:
        cm = CoherenceModel(
            topics=topics,
            dictionary=coherence_dict,
            texts=coherence_texts,
            coherence=args.coherence_type,
            window_size=args.window_size,
            topn=top_n,
        )

        coherences = np.round(cm.get_coherence_per_topic(), decimals=4)

        for i, coherence in enumerate(coherences):
            all_coherences[i].append(coherence)

    fname_suffix = "" if args.out_name is None or len(args.out_name) == 0 else f"_{args.out_name}"
    out_path = os.path.join(os.path.dirname(args.topics_path), f"coherences{fname_suffix}.txt")

    print(f"Finished in {time.time() - start:.2f}s.")
    print(f"Saving coherences to {out_path}.")

    def print_and_write(s, f):
        print(s)
        f.write(s + "\n")

    with open(out_path, "w") as f:
        f.write("*" * 10 + "\n")
        f.write(str(args) + "\n")
        f.write("*" * 10 + "\n")
        f.write(f"Coherence texts: {fnames}\n\n")

        for idx, (coherences, topic) in enumerate(zip(all_coherences, topics)):
            print_and_write(f"[{idx}] {coherences} {' '.join(topic)}", f)

        if len(args.top_n) > 1:
            mean_coherences_each_n = np.round(np.mean(all_coherences, axis=0), decimals=4)
            print_and_write(f"\nMean coherences: {mean_coherences_each_n}", f)

        agg_coherence = np.round(np.mean(all_coherences), decimals=4)
        print_and_write(f"\nAggregate coherence: {agg_coherence}", f)


if __name__ == "__main__":
    main()
