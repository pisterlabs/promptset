import time
from collections import Counter

import gensim.corpora as corpora
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud


def _clean_stop_words(df, list_stop_words):
    df = df.apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in (list_stop_words)]
        )
    )
    return df


def compute_bag_of_words(df):
    # split each tweet sentence into words
    text_tokens = [[text for text in tweet.split()] for tweet in df]
    # create a dictionary
    dico_words = corpora.Dictionary(text_tokens)
    # Filter too common or rare words
    dico_words.filter_extremes(no_below=10, no_above=0.95)
    # compute the frequency of each word in the dictionary
    doc_term_matrix = [dico_words.doc2bow(rev) for rev in text_tokens]

    return (text_tokens, dico_words, doc_term_matrix)


def compute_lda(dictionary, corpus, texts, num_topics):
    start = time.time()
    LDA = LdaModel
    # Build LDA model
    lda_model = LDA(
        corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        alpha="auto",
        passes=20,
        per_word_topics=True,
    )

    end = time.time()
    delta = (end - start) / 60
    print(f"=== LDA model with {num_topics} topics took : {delta:.2} minutes")

    topics = []
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} -> Words: {}".format(idx, topic))
    topics.append(topic)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"=== Coherence Score: {coherence_lda:.2}")
    # 0.33063023037515266 baseline
    return lda_model, coherence_lda


def _graph_nb_topics(start, limit, step, coherence_values, alias):
    x = range(start, limit, step)

    fig = plt.figure()
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title(f"Coherence values for {alias}", size=18)
    plt.legend("Coherence values", loc="best")
    plt.show()
    fig.savefig(f"output/img/search_nbtopics_{alias}.png", dpi=200)


def search_nb_topics(dictionary, corpus, texts, start, limit, step, alias):
    coherence_values = []
    for num_topics in range(start, limit, step):
        # todo: transform into a function
        model = compute_lda(
            dictionary=dictionary, corpus=corpus, texts=texts, num_topics=num_topics
        )
        coherence_values.append(model[1])

    # show graph
    _graph_nb_topics(start, limit, step, coherence_values, alias)

    best_values = [
        index
        for index, value in enumerate(coherence_values)
        if value == max(coherence_values)
    ]

    return best_values[0] + start


# Assign a topic to each tweet
# TODO: too many loops, takes too long
def _format_topics_sentences(lda_model, corpus, texts):

    sent_topics_df = pd.DataFrame()

    start = time.time()
    # Get main topic in each document
    for i, row_list in enumerate(lda_model[corpus]):
        # per_word_topics: list of topics, sorted in descending order of most likely topics for each word
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            # dominant topic
            if j == 0:
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break
    end = time.time()
    delta = (end - start) / 60
    print(f"Format topics took {delta:.2} minutes")
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # format
    sent_topics_df = sent_topics_df.reset_index()
    sent_topics_df.columns = [
        "document_no",
        "dominant_topic",
        "topic_perc_contrib",
        "keywords",
        "tokens",
    ]

    return sent_topics_df


def graph_topics(lda_result, df_dominant_topics, alias):
    topics = lda_result.show_topics(formatted=False, num_words=20)
    data_flat = [w for w_list in df_dominant_topics["tokens"] for w in w_list]
    counter = Counter(data_flat)

    all_ = pd.DataFrame()
    for i, topic in topics:
        for word, weight in topic:
            tmp = pd.DataFrame(
                [[word, i, weight, counter[word]]],
                columns=["keyword", "topic", "weight", "word_count"],
            )
            all_ = pd.concat([all_, tmp], axis=0)

    # Plot Word Count and Weights of Topic Keywords
    ncols = 2
    nrows = len(all_["topic"].unique()) // ncols + (
        len(all_["topic"].unique()) % ncols > 0
    )
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"{alias} : Word counts and weights for each topic")
    # fig, axes = plt.subplots(nrows, ncols, figsize=(20,15), sharey=True, dpi=100)
    # cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, topic in enumerate(all_["topic"].unique()):
        ax = plt.subplot(nrows, ncols, i + 1)

        ax.bar(
            x="keyword",
            height="word_count",
            data=all_.loc[all_["topic"] == topic, :],
            # color=cols[i],
            width=0.5,
            alpha=0.5,
        )
        ax_twin = ax.twinx()
        ax_twin.scatter(
            x="keyword",
            y="weight",
            data=all_.loc[all_["topic"] == topic, :],
            color="C7",
            label="Weights",
        )

        ax.set_ylabel("Word Count",)
        # color=cols[i])
        ax_twin.set_ylim(0, (all_["weight"].max() + 0.02))
        ax.set_ylim(0, all_["word_count"].max() + 100)
        ax.set_title(
            f"Topic: {topic}", fontsize=16  # color=cols[i],
        )
        ax.tick_params(axis="y", left=False)
        ax.set_xticklabels(
            all_.loc[all_["topic"] == topic, "keyword"],
            rotation=30,
            horizontalalignment="right",
        )

        ax.legend(loc="upper left", fontsize=10)
        ax_twin.legend(loc="upper right", fontsize=10)
        fig.tight_layout()

    fig.savefig(f"output/img/{alias}.png", dpi=200)


def compute_word_cloud(df, text, alias):
    color_list = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(
        # background_color='white',
        width=2500,
        height=1800,
        max_words=200,
        colormap="Oranges",
        color_func=lambda *args, **kwargs: color_list[i],
        prefer_horizontal=1.0,
    )

    ncols = 2
    nrows = len(df["dominant_topic"].unique()) // ncols + (
        len(df["dominant_topic"].unique()) % ncols > 0
    )
    fig = plt.figure(figsize=(20, 20))
    # plt.subplots_adjust(hspace=0.9)
    fig.suptitle(f"World Cloud for {text}", fontsize=18, y=0.95)
    plt.tight_layout()

    # for i, ax in enumerate(axes.flatten()):
    for i, variable in enumerate(df["dominant_topic"].unique()):
        # add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, i + 1)
        # fig.add_subplot(ax)
        cloud.generate(str(df[df["dominant_topic"] == i][text].values))
        plt.gca().imshow(cloud, interpolation="bilinear")
        plt.gca().set_title("Topic " + str(i), fontdict=dict(size=16))
        plt.gca().axis("off")
    nb_topics = df["dominant_topic"].max() + 1
    plt.savefig(f"output/img/world_cloud_{alias}_{nb_topics}.png")
