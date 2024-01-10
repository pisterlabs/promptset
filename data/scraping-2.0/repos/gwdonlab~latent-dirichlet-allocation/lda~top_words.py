import os, json, pickle
from ogm.trainer import TextTrainer
from gensim.models import CoherenceModel
import argparse as ap

# Experiment parameters obtained by CLI args
argparser = ap.ArgumentParser()
argparser.add_argument(
    "--n_topics",
    help="Number of topics to search coherences from",
    type=int,
    required=True,
)
argparser.add_argument("--experiment_config", help="Path to experiment JSON file", required=True)
argparser.add_argument(
    "--model_num",
    help="Specific model to get keywords from; if not specified, will find the highest-coherence model",
    required=False,
    type=int,
)
argparser.add_argument(
    "--word_cloud",
    action="store_true",
    help="If set, will create a wordcloud of keywords",
)
argparser.add_argument(
    "--wordcloud_wordcount",
    required=False,
    type=int,
    help="Number of keywords to place in wordcloud",
    default=12,
)
argparser.add_argument(
    "--dump_wordcloud_data",
    action="store_true",
    help="If set, will dump word/weight data from wordcloud into pkl files",
)
argparser.add_argument(
    "--wordcloud_mask",
    help="Path to B/W PNG file that will be used as the shape for the wordcloud",
    required=False,
    default="shape.png",
)
argparser.add_argument(
    "--wordcloud_noshow", help="Don't bother displaying wordclouds", action="store_true"
)
argparser.add_argument(
    "--ldavis",
    action="store_true",
    help="If set, will generate an LDAvis HTML doc for the given topic model",
)
args = argparser.parse_args()


def main():

    # Determine experiment identifier based on config file
    with open(args.experiment_config, "r") as infile:
        setup_dict = json.load(infile)

    experiment_name = setup_dict["name"]

    # Loop through the metadata and find the model with the highest coherence if user didn't specify
    if args.model_num is None:
        filename = (
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(args.n_topics)
            + "topics"
            + "/metadata.json"
        )

        info = {}
        with open(filename, "r") as json_file:
            info = json.load(json_file)

        best_coherence = -1.0
        main_path = None
        for key in info.keys():
            if "model" in key and float(info[key]["coherence"]) > best_coherence:
                best_coherence = float(info[key]["coherence"])
                main_path = info[key]["path"]

        print("Best coherence: " + str(best_coherence))

    else:
        main_path = (
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(args.n_topics)
            + "topics/model_"
            + str(args.model_num)
        )

    model_path = main_path + "/lda.model"
    print("Loading model from: " + model_path)

    trainer = TextTrainer()
    trainer.load_model("lda", model_path)
    try:
        cm = CoherenceModel.load(main_path + "/coherence.model")
        topic_coherences = cm.get_coherence_per_topic()
        print(topic_coherences)
    except FileNotFoundError:
        print("There was no coherence model saved for this topic model.")
        topic_coherences = None

    i = 0
    for topic_id, topic in trainer.model.print_topics(-1):
        print("* Topic: " + str(topic_id) + " \n\t* Words: " + topic)
        if topic_coherences is not None:
            print("\t* Per-topic coherence:", topic_coherences[i])
            i += 1

    if args.word_cloud:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        from wordcloud import WordCloud

        for topic_id, topic_words in trainer.model.print_topics(
            num_topics=-1, num_words=args.wordcloud_wordcount
        ):
            # Data for plotting wordcloud as a bar chart
            x_axis = []
            y_axis = []

            cloud_words = ""
            for weight_word_pair in topic_words.split(" + "):
                pairing = weight_word_pair.split("*")
                adjusted_weight = int(float(pairing[0]) * 1000)
                cleaned_word = pairing[1].replace('"', "")
                for i in range(adjusted_weight):
                    cloud_words += cleaned_word + " "

                x_axis.append(cleaned_word)
                y_axis.append(pairing[0])

            if args.dump_wordcloud_data:
                with open("axes_" + str(topic_id) + ".pkl", "wb") as outfile:
                    pickle.dump([x_axis, y_axis], outfile)

            if not args.wordcloud_noshow:
                cloud_shape = np.array(Image.open(args.wordcloud_mask))
                wordcloud = WordCloud(
                    width=800,
                    height=800,
                    background_color="white",
                    min_font_size=10,
                    collocations=False,
                    mask=cloud_shape,
                ).generate(cloud_words)

                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)

    if args.ldavis:
        from pyLDAvis.gensim_models import prepare
        from pyLDAvis import save_html

        data_path = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]
        trainer.parse_file(data_path)
        if "time_filter" in setup_dict:
            trainer.filter_within_time_range(
                col=setup_dict["time_filter"]["time_key"],
                data_format=setup_dict["time_filter"]["data_format"],
                input_format=setup_dict["time_filter"]["arg_format"],
                start=setup_dict["time_filter"]["start"],
                end=setup_dict["time_filter"]["end"],
            )

        if "attribute_filters" in setup_dict:
            for attr_filter in setup_dict["attribute_filters"]:
                trainer.filter_data(attr_filter["filter_key"], set(attr_filter["filter_vals"]))

        if "replace_before_stemming" in setup_dict:
            trainer.replace_words(setup_dict["text_key"], setup_dict["replace_before_stemming"])

        if "remove_before_stemming" in setup_dict:
            trainer.remove_words(setup_dict["text_key"], set(setup_dict["remove_before_stemming"]))

        trainer.lemmatize_stem_words(setup_dict["text_key"])

        if "replace_after_stemming" in setup_dict:
            trainer.replace_words(setup_dict["text_key"], setup_dict["replace_after_stemming"])

        if "remove_after_stemming" in setup_dict:
            trainer.remove_words(setup_dict["text_key"], set(setup_dict["remove_after_stemming"]))

        trainer.make_dict_and_corpus(setup_dict["text_key"])
        data = prepare(trainer.model, trainer.corpus, trainer.dictionary)
        save_html(data, setup_dict["name"] + "_" + str(args.n_topics) + ".html")


if __name__ == "__main__":
    main()
