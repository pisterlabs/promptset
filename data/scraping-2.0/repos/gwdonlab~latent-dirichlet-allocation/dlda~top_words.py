import os, json, random, csv
import argparse as ap
from ogm.trainer import TextTrainer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from labellines import labelLines


def get_args():
    argparser = ap.ArgumentParser()

    argparser.add_argument("n_topics", help="Number of topics to analyze results for", type=int)
    argparser.add_argument(
        "--experiment_config", help="Path to experiment's JSON file", required=True
    )
    argparser.add_argument(
        "--show_plot",
        help="Display a plot of individual topics' coherence scores",
        action="store_true",
    )
    argparser.add_argument(
        "--print_only_topic",
        help="Print only the keywords for this topic number",
        type=int,
    )
    argparser.add_argument(
        "--plot_keywords_topic",
        help="Pass an int to show bar plots for keywords in a particular topic",
        type=int,
    )
    argparser.add_argument(
        "--shuffle_colors",
        help="Randomly generate plot colors for topic plot rather than use matplotlib's colors",
        action="store_true",
    )
    argparser.add_argument(
        "--plot_keywords_frame",
        help="Pass an int to only show a keyword bar plot for that time frame (int should index an array of time frames)",
        type=int,
    )
    argparser.add_argument(
        "--bar_color",
        help="Matplotlib-accepted color for the bar plots of topic keywords",
        default="#1f77b4",
    )
    argparser.add_argument(
        "--lock_barplot_yaxis",
        help="Set upper limit on bar plot's y-axis",
        type=float,
    )
    argparser.add_argument(
        "--plot_title",
        help="Title for the individual topic coherence plot",
        default="Coherence Scores of Individual Topics",
    )
    argparser.add_argument(
        "--lock_yaxis",
        help="Set this flag to force the y-axis to be [0, 1]",
        action="store_true",
    )
    argparser.add_argument(
        "--label_lines",
        action="store_true",
        help="Set this flag to put labels directly on the lines instead of a plot legend. This behavior is automatic for n_topics > 10",
    )
    argparser.add_argument(
        "--remove_from_label",
        nargs="*",
        help="List of strings to remove from x-axis plot labels",
    )
    argparser.add_argument(
        "--write_axes", help="Write the axes of all plots to CSV files", action="store_true"
    )
    args = argparser.parse_args()
    with open(args.experiment_config, "r") as infile:
        input_dict = json.load(infile)

    return input_dict, args


def main(setup_dict, args):
    n_topics = args.n_topics
    remove_from_label = args.remove_from_label
    only_topic = args.print_only_topic
    experiment_name = setup_dict["name"]

    main_path = os.getenv("MODEL_DIR") + "/" + experiment_name + "/" + str(n_topics) + "topics"

    # Load metadata file
    with open(main_path + "/metadata.json", "r") as json_file:
        info = json.load(json_file)

    # Load model
    model_path = main_path + "/ldaseq.model"
    print("Loading model from: " + model_path)
    trainer = TextTrainer()
    trainer.load_model("ldaseq", model_path)

    # We will be keeping track of each individual topic's coherence for a plot later
    individual_coherences = [[0] * (len(info.keys()) - 1) for x in range(n_topics)]
    time_frame_labels = []

    # If a specific topic was specified, alert user about change in behavior
    if only_topic is not None:
        print("KEYWORDS FOR TOPIC", only_topic)

    # For each topic, print the topic's top words and coherence
    for i in range(len(info.keys()) - 1):
        if only_topic is None:
            print("<details>")
            print("<summary> Click to expand time frame " + str(i) + " </summary>\n")
            print("Average coherence for time frame:", info["time_" + str(i)]["coherence"])

        cm = CoherenceModel.load(info["time_" + str(i)]["coherence_savepath"])
        topic_coherences = cm.get_coherence_per_topic()
        this_label = info["time_" + str(i)]["start_time"]
        print("\nTime period start date:", this_label)
        if remove_from_label is not None:
            for item in remove_from_label:
                this_label = this_label.replace(item, "")
        time_frame_labels.append(this_label)

        # Loop through individual topics in this time slice
        j = 0
        for topic in trainer.model.print_topics(time=i, top_terms=10):
            # If a specific topic was specified, only print that
            if only_topic is not None:
                if j == only_topic:
                    print(topic)

            # Otherwise, print everything in Markdown bullet format
            else:
                print("* Topic: " + str(j) + " \n  * Words:", topic)
                print("  * Per-topic coherence:", topic_coherences[j])

            # If plotting a bar graph of this topic at this time frame
            # OR plotting a bar graph of this topic at all time frames
            if (args.plot_keywords_topic == j and args.plot_keywords_frame == i) or (
                args.plot_keywords_topic == j and args.plot_keywords_frame is None
            ):
                x_axis = []
                y_axis = []
                for word, weight in topic:
                    x_axis.append(word)
                    y_axis.append(weight)

                plt.bar(x_axis, y_axis, color=args.bar_color)

                if args.lock_barplot_yaxis is not None:
                    plt.ylim((0, args.lock_barplot_yaxis))

                plt.xticks(rotation=30)
                plt.title("Topic " + str(j) + " Word Probabilities, Time Frame " + str(i))
                plt.show()

                if args.write_axes:
                    with open(
                        "topic_" + str(j) + "_timeframe_" + str(i) + "_wordprobs.csv", "w"
                    ) as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(zip(x_axis, y_axis))

            individual_coherences[j][i] = topic_coherences[j]
            j += 1

        if only_topic is None:
            print("</details>\n")

    print("Average coherence:", info["aggregated"]["avg_coherence"])

    if args.show_plot:
        ticks = [i for i in range(len(time_frame_labels))]
        for i in range(n_topics):
            if args.shuffle_colors:
                r = lambda: random.randint(0, 255)
                plt.plot(
                    ticks,
                    individual_coherences[i],
                    label="Topic " + str(i),
                    color="#%02X%02X%02X" % (r(), r(), r()),
                )
            else:
                plt.plot(ticks, individual_coherences[i], label="Topic " + str(i))

        plt.title(args.plot_title)
        plt.xlabel("Start of time frame")
        plt.ylabel("Coherence score ($C_v$)")
        plt.xticks(rotation="vertical", labels=time_frame_labels, ticks=ticks)

        if args.lock_yaxis:
            plt.ylim(ymax=1, ymin=0)

        if args.label_lines or n_topics > 10:
            labelLines(plt.gca().get_lines())
        else:
            plt.legend()

        plt.show()
        if args.write_axes:
            with open("coherences_plot.csv", "w") as outfile:
                individual_coherences.insert(0, ticks)
                writer = csv.writer(outfile)
                writer.writerows(zip(*individual_coherences))


if __name__ == "__main__":
    in_dict, args = get_args()
    main(in_dict, args)
