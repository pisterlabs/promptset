from utils.args import add_args
from utils.plotting_utils import vconcat_resize
from utils.misc_utils import authenticate, seed_function
import logging
import os
import pandas as pd
import numpy as np
import inflect
import openai
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import matplotlib as mpl
import seaborn as sns
import argparse
import json
from scipy.optimize import linear_sum_assignment, linprog
from PIL import Image
import cv2
import zipfile
from utils.templates import PROFICIENCY_METRICS, LABEL_KEY, TASK_INSTRUCTIONS

inflect_engine = inflect.engine()

font_size = 26

# Create a Matplotlib Font object from our `.ttf` file
font = font_manager.FontEntry(fname=str("fonts/Roboto-Regular.ttf"), name="roboto")
# Register this object with Matplotlib's ttf list
font_manager.fontManager.ttflist.append(font)

rc = {}
rc["font.family"] = "roboto"
rcParams.update(rc)

PROFICIENCY_FILTER_THRESHOLD = {}
PROFICIENCY_FILTER_THRESHOLD["mbpp"] = 1
PROFICIENCY_FILTER_THRESHOLD["knkarthick_dialogsum"] = 0.25
PROFICIENCY_FILTER_THRESHOLD["mmlu_biology"] = 1

CLASSIFICATION_TASKS = ["mmlu_biology", "medmcqa"]

# pretty labels

PRETTY_LABELS = {}
PRETTY_LABELS["mbpp"] = {}
PRETTY_LABELS["mbpp"]["subtask"] = {
    "Parse natural language description": "Parse Description",
    "Understand test cases": "Understand Test Cases",
    "Handle data types and structures": "Handle Data Types",
    "Implement mathematical operations": "Implement Math Operations",
    "Handling loops and conditionals": "Handle Loops/If-Else",
    "Manage variable assignments and data manipulation": "Variable Assignments",
    "Implement algorithmic operations": "Implement Algorithms",
    "Handle exception and error cases": "Handle Exceptions & Errors",
    "Optimize for efficiency and readability": "Optimize for Efficiency",
    "Validate against test cases": "Validate Against Test Cases",
    "Generate Python syntax": "Generate Python Syntax",
    "Manipulate arrays and lists": "Manipulate Arrays & Lists",
    "Handle edge cases or special scenarios": "Handle Edge Cases",
    "Extract and store arrays from function parameters": "Extract & Store Arrays",
}
PRETTY_LABELS["mbpp"]["domain"] = {
    "Mathematical Operations": "Mathematical Operations",
    "String Manipulation": "String Manip.",
    "List Manipulation": "List Manip.",
    "Conditional Statements": "Conditional Statements",
    "Data Processing": "Data Processing",
    "Sorting": "Sorting",
    "Number Manipulation": "Number Manip.",
    "Tuple Manipulation": "Tuple Manip.",
    "Boolean Operations": "Bool Operations",
    "Geometric Calculations": "Geometric Calculations",
    "Text Pattern Matching": "Text Pattern Matching",
    "Array Manipulation": "Array Manip.",
    "File Handling": "File Handling",
    "Data Validation": "Data Validation",
    "Sequence Analysis": "Sequence Analysis",
}

PRETTY_LABELS["knkarthick_dialogsum"] = {}
PRETTY_LABELS["knkarthick_dialogsum"]["subtask"] = {
    "Identifying the participants in the conversation": "Identify the participants",
    "Understanding the topic of discussion": "Understand the topic",
    "Extracting key information or important details": "Extract key information",
    "Summarizing the conversation concisely": "Summarize concisely",
    "Recognizing the roles and relationships of the speakers": "Recognize roles",
    "Comprehending specific statements or questions": "Comprehend specific statements",
    "Interpreting instructions or suggestions": "Interpret instructions/suggestions",
    "Identifying requests for information or clarification": "Identify requests for information",
    "Extracting important information and questions": "Extract important questions",
    "Understanding the conversational context": "Understand conversational context",
    "Recognizing the main topic of conversation": "Recognize main topic",
    "Noting suggestions, recommendations, or solutions proposed": "Note suggestions/solution",
    "Extracting information about language proficiency or qualifications": "Extract language proficiency",
    "Recognizing and interpreting emotions": "Recognize & interpret emotions",
    "Extracting relevant details": "Extract relevant details",
}


PRETTY_LABELS["knkarthick_dialogsum"]["domain"] = {
    "Dating and relationships": "Dating & relationships",
    "Outdoor activities and sports": "Outdoor activities",
    "Career and job interviews": "Career & job interviews",
    "Food and restaurant ordering": "Food & restaurant ordering",
    "Environmental issues and pollution": "Environmental issues",
    "Social interactions and personal relationships": "Social interactions",
    "Leisure and recreation": "Leisure & recreation",
    "Employment and professional skills": "Employment & professional skills",
    "Food and hospitality": "Food & hospitality",
    "Environmental conservation and sustainability": "Environmental sustainability",
    "Movie preferences and plans": "Movie preferences & plans",
    "Sports and live events": "Sports & live events",
    "Fashion and clothing choices": "Fashion & clothing choices",
    "Education": "Education",
    "Work environment": "Work environment",
}

PRETTY_LABELS["mmlu_biology"] = {}
PRETTY_LABELS["mmlu_biology"]["subtask"] = {
    "Understanding and interpreting clinical information": "Interpret clinical info",
    "Identifying and categorizing symptoms, conditions, and diseases": "Identify symptoms",
    "Analyzing and processing medical test results": "Analyze medical tests",
    "Recommending appropriate treatments and interventions based on patient-specific factors": "Recommend appropriate treatment",
    "Providing accurate and relevant information to healthcare professionals and patients": "Provide acc. info",
    "Understanding and interpreting multiple choice questions": "Interpret multiple-choice ques.",
    "Analyzing and selecting the correct answer choice": "Analyze answer choice",
    "Recognizing key terms and concepts in clinical biology": "Recognize concepts",
    "Identifying patterns and relationships between questions and answers": "Identify patterns b/w Ques. and Ans.",
    "Retaining and applying knowledge from example data to new questions and answers": "Apply knowledge",
    "Understanding and classifying pH levels": "Understanding and classifying pH levels",
    "Providing information and reminders about medication administration and potential side effects": "Providing information and reminders about medication administration and potential side effects",
    "Suggesting the appropriate size of cannula for specific medical interventions such as blood transfusions": "Suggesting the appropriate size of cannula for specific medical interventions",
    "Applying domain-specific knowledge to select the most appropriate answer choice": "Apply domain-specific knowledge",
    "Identifying potential drug interactions": "Identify potential drug interactions",
}
PRETTY_LABELS["mmlu_biology"]["domain"] = {
    "Cell Biology": "Cell Biology",
    "Neurology": "Neurology",
    "Biochemistry": "Biochemistry",
    "Physiology": "Physiology",
    "Pharmacology": "Pharmacology",
    "Clinical biology": "Clinical biology",
    "Diagnostic tests": "Diagnostic tests",
    "Treatment options": "Treatment options",
    "Anatomy and physiology": "Anatomy & physiology",
    "Medical procedures and interventions": "Medical procedures",
    "Genetics and heredity": "Genetics and heredity",
    "Dermatology": "Dermatology",
    "Urology": "Urology",
    "Respiratory medicine": "Respiratory medicine",
    "Wound healing and surgery": "Wound healing & surgery",
}


def get_dataset_assignment_LP(
    args,
    all_category_elements_importance_scores,
    ground_truth_scores,
    categories,
    max_assignments_per_data_point=2,
    slack=0.1,
):
    assignments = {}
    for category in categories:
        # filter based on category
        category_gt_scores = ground_truth_scores[
            ground_truth_scores["category_type"] == category
        ]
        # if we don't have a complete graph (scores for some missing categories), do a join and assign a -1 score to missing categories
        # Fill missing values with default score
        default_score = -1
        category_gt_scores["score"] = category_gt_scores["score"].fillna(default_score)
        category_gt_scores_pivoted = category_gt_scores.pivot(
            index="id", columns="category", values="score"
        ).fillna(default_score)
        category_gt_scores_np = category_gt_scores_pivoted.values
        num_data_points, num_category_elements = category_gt_scores_np.shape
        # duplicate the columns based on the importance scores
        category_elements_importance_scores = all_category_elements_importance_scores[
            category
        ]
        # align the importance scores categories with the columns of the gt scores
        category_elements_importance_scores = (
            category_elements_importance_scores.reindex(
                category_gt_scores_pivoted.columns
            )
        )
        category_elements_importance_scores_np = (
            category_elements_importance_scores.values
        )
        category_elements_importance_scores_np = (
            category_elements_importance_scores_np
            / np.sum(category_elements_importance_scores_np)
        )
        num_slots_per_category_element = np.floor(
            category_elements_importance_scores_np * num_data_points
        ).astype(int)
        # the number of slots might not add up to the number of data points
        # distrbute the remaining slots randomly
        num_slots_remaining = num_data_points - np.sum(num_slots_per_category_element)
        if num_slots_remaining > 0:
            num_slots_per_category_element = (
                num_slots_per_category_element
                + np.random.multinomial(
                    num_slots_remaining,
                    np.ones(num_category_elements) / num_category_elements,
                )
            )
        num_slots_per_category_element = (
            num_slots_per_category_element * max_assignments_per_data_point
        )
        # add some slack
        assert slack >= 0 and slack <= 1
        num_slots_per_category_element_ub = num_slots_per_category_element + np.floor(
            slack * num_slots_per_category_element
        )
        num_slots_per_category_element_lb = num_slots_per_category_element - np.floor(
            slack * num_slots_per_category_element
        )

        # construct the linear program
        # decision variables
        # x_ij = 1 if category element j is assigned to data point i
        # x_ij = 0 otherwise
        # objective function
        # max sum_i sum_j x_ij * score_ij
        # i = [1, num_data_points]
        # j = [1, num_category_elements]
        # constraints
        # sum_j x_ij = 2 for all i
        # sum_i x_ij = num_slots_per_category_element[j] * (1 +- slack) for all j
        # flexible solver
        # x_ij =  {0,1}
        # score_ij = [1,5]
        num_category_elements = category_gt_scores_np.shape[1]
        num_data_points = category_gt_scores_np.shape[0]
        # cost vector
        c = category_gt_scores_np.flatten()
        A = np.zeros(
            (
                num_data_points + num_category_elements + num_category_elements,
                num_data_points * num_category_elements,
            )
        )
        b = np.zeros(num_data_points + num_category_elements + num_category_elements)
        # constraint 1
        for i in range(num_data_points):
            A[i, i * num_category_elements : (i + 1) * num_category_elements] = 1
            b[i] = max_assignments_per_data_point
        # constraint 2 -- upper bound
        for j in range(num_category_elements):
            A[num_data_points + j, j::num_category_elements] = 1
            b[num_data_points + j] = num_slots_per_category_element_ub[j]
        # constraint 2 -- lower bound
        for j in range(num_category_elements):
            A[
                num_data_points + num_category_elements + j, j::num_category_elements
            ] = -1
            b[
                num_data_points + num_category_elements + j
            ] = -num_slots_per_category_element_lb[j]
        # solve the linear program
        res = linprog(-c, A_ub=A, b_ub=b, bounds=(0, 1), integrality=1)
        # get the assignments
        reshaped_assignments = res.x.reshape(num_data_points, num_category_elements)
        assert np.all(
            np.logical_or(reshaped_assignments == 0, reshaped_assignments == 1)
        )
        assignment = {}
        for j in range(num_category_elements):
            non_zeros_data_points = np.nonzero(reshaped_assignments[:, j] == 1)
            assignment[
                category_gt_scores_pivoted.columns[j]
            ] = category_gt_scores_pivoted.index[non_zeros_data_points].tolist()
            assert (
                len(assignment[category_gt_scores_pivoted.columns[j]])
                <= num_slots_per_category_element_ub[j]
            )
            assert (
                len(assignment[category_gt_scores_pivoted.columns[j]])
                >= num_slots_per_category_element_lb[j]
            )
            print(
                "Number of assignments for category element",
                category_gt_scores_pivoted.columns[j],
                f"{len(assignment[category_gt_scores_pivoted.columns[j]])} {num_slots_per_category_element_ub[j]} {num_slots_per_category_element_lb[j]}",
            )
            assignments[category] = assignment
    return assignments


def preprocessing(args, all_scores_generations, all_scores_gt, proficiency_scores):
    categories = args.categories.split(",")
    # rename the columns to category
    for category in categories:
        category_df = all_scores_generations[category]
        category_df.rename(columns={category: "category"}, inplace=True)
        category_df["category_type"] = category
        category_df_gt = all_scores_gt[category]
        category_df_gt.rename(columns={category: "category"}, inplace=True)
        category_df_gt["category_type"] = category
    # merge all categories into a single dataframe
    all_scores_generations_merged = pd.concat(all_scores_generations.values())
    all_scores_gt_merged = pd.concat(all_scores_gt.values())

    # assert no empty generations, and assert scores in range [0,5]
    assert np.all(all_scores_generations_merged["generation"] != "")
    assert np.all(all_scores_gt_merged["generation"] != "")
    assert np.all(all_scores_generations_merged["score"] >= 0)
    assert np.all(all_scores_generations_merged["score"] <= 5)
    assert np.all(all_scores_gt_merged["score"] >= 0)
    assert np.all(all_scores_gt_merged["score"] <= 5)

    assert len(all_scores_generations_merged) == len(all_scores_gt_merged)

    # assert number of unique ids in generations and gt are the same
    # assert number of unique ids in proficiency scores and gt are the same
    assert np.all(
        np.unique(all_scores_generations_merged["id"].values)
        == np.unique(all_scores_gt_merged["id"].values)
    )
    assert np.all(
        np.unique(all_scores_generations_merged["id"].values)
        == np.unique(proficiency_scores.index.values)
    )
    pruned_category_elements = {}
    for category in categories:
        category_df_gt = all_scores_gt_merged[
            all_scores_gt_merged["category_type"] == category
        ]
        # find mean score for different elements in the category with pd groupby
        grouped_category_df_gt = category_df_gt.groupby(category_df_gt["category"])
        scores_per_category_type = grouped_category_df_gt["score"].mean()
        top_10_category_elements = scores_per_category_type.sort_values(ascending=False)
        top_10_category_elements = top_10_category_elements[:10]
        top_10_category_elements.index = top_10_category_elements.index.str.split(
            ":"
        ).str.get(0)
        pruned_category_elements[category] = top_10_category_elements
    # prune the generation score to only contain the top 10 category elements
    pruned_generation_scores = []
    for category in categories:
        category_df = all_scores_generations_merged[
            all_scores_generations_merged["category_type"] == category
        ]
        category_df["category"] = category_df["category"].str.split(":").str.get(0)
        category_df = category_df[
            category_df["category"].isin(pruned_category_elements[category].index)
        ]
        pruned_generation_scores.append(category_df)
    # pruned GT scores as well
    pruned_gt_scores = []
    for category in categories:
        category_df = all_scores_gt_merged[
            all_scores_gt_merged["category_type"] == category
        ]
        category_df["category"] = category_df["category"].str.split(":").str.get(0)
        category_df = category_df[
            category_df["category"].isin(pruned_category_elements[category].index)
        ]
        pruned_gt_scores.append(category_df)
    # merge all categories into a single dataframe
    all_scores_gt_merged_pruned = pd.concat(pruned_gt_scores)
    all_scores_generations_merged_pruned = pd.concat(pruned_generation_scores)
    return (
        all_scores_generations_merged_pruned,
        all_scores_gt_merged_pruned,
        pruned_category_elements,
    )


def get_gt_breakdown(args, all_scores_gt):
    categories = args.categories.split(",")
    fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(40, 9))
    # visualize the scores for these category elements
    for i, category in enumerate(categories):
        category_df_gt = all_scores_gt[all_scores_gt["category_type"] == category]
        # find mean score for different elements in the category with pd groupby
        grouped_category_df_gt = category_df_gt.groupby(category_df_gt["category"])
        scores_per_category_type = grouped_category_df_gt["score"].mean()
        # setting title etc.
        scores_per_category_type = scores_per_category_type.sort_values(ascending=True)
        qualitative_colors = sns.color_palette("husl", 10)

        sns.set_theme(style="white")
        sns.set_palette(qualitative_colors)
        sns.set_style("white")
        labels = scores_per_category_type.index
        labels_split = []
        for label in labels:
            label = label.strip()
            if args.pretty_plot:
                label = PRETTY_LABELS[args.task_name][category][label]
            else:
                label_words = label.split()
                label = "\n".join(
                    [
                        " ".join(label_words[: len(label_words) // 2]),
                        " ".join(label_words[len(label_words) // 2 :]),
                    ]
                )
            labels_split.append(label)

        axes[i].pie(
            x=scores_per_category_type.values,
            labels=labels_split,
            colors=qualitative_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": font_size},
            pctdistance=0.80,
            explode=[0.05] * len(scores_per_category_type),
        )
        # add labels
        axes[i].set_title(
            f"{inflect_engine.plural_noun(category.capitalize())}",
            fontsize=1.5 * font_size,
        )
        hole = plt.Circle((0, 0), 0.65, facecolor="white")
        axes[i].add_patch(hole)
        # save the scores for each category element
        scores_per_category_type.to_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"gt_scores_per_category_element_{category}.csv",
            )
        )

    fig.suptitle("Prior over categories", fontsize=2 * font_size)
    plt.tight_layout(h_pad=2, w_pad=2, pad=2)
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "gt_breakdown.pdf"),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "gt_breakdown.png"),
        dpi=300,
        transparent=True,
    )


def get_correlation_breakdown(
    args, all_scores_generations, all_scores_gt, proficiency_scores
):
    # initialize the reportcard plot
    categories = args.categories.split(",")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    # visualize the scores for these category elements
    color_dict = {"subtask": ["#ffc8c8", "#ff5858"], "domain": ["#bbdefb", "#2196f3"]}
    for i, category in enumerate(["subtask"]):
        generation_correlations = {}

        category_df_gt = all_scores_gt[all_scores_gt["category_type"] == category]
        category_df_generations = all_scores_generations[
            all_scores_generations["category_type"] == category
        ]
        # iterate over the category elements
        # for each category element, find the correlation between the gt and generation scores
        for category_element in category_df_gt["category"].unique():
            # filter based on category element
            category_element_df_gt = category_df_gt[
                category_df_gt["category"] == category_element
            ]
            category_element_df_generations = category_df_generations[
                category_df_generations["category"] == category_element
            ]
            # sort both the dataframes based on the id
            category_element_df_gt = category_element_df_gt.sort_values(by="id")
            category_element_df_generations = (
                category_element_df_generations.sort_values(by="id")
            )
            # group by id to get proficiency score for each generation
            category_element_df_generations = category_element_df_generations.join(
                proficiency_scores,
                on="id",
                how="inner",
                rsuffix="_proficiency",
            )
            filter_index = (
                category_element_df_generations["proficiency_score"]
                >= PROFICIENCY_FILTER_THRESHOLD[args.task_name]
            )
            category_element_df_generations = category_element_df_generations[
                filter_index
            ]
            category_element_df_gt = category_element_df_gt[
                category_element_df_gt["id"].isin(
                    category_element_df_generations["id"].values
                )
            ]
            assert np.all(
                category_element_df_gt["id"].values
                == category_element_df_generations["id"].values
            )
            # filter based on proficiency scores

            num_intersection = np.sum(
                np.abs(
                    category_element_df_gt["score"].values
                    - category_element_df_generations["score"].values
                )
                >= 2
            )
            correlation = num_intersection / len(category_element_df_gt)
            generation_correlations[category_element] = correlation
        # plot the scores for each category element
        generation_correlations = pd.DataFrame.from_dict(
            generation_correlations, orient="index"
        )
        generation_correlations.columns = ["score"]
        # sort based on score before plotting
        generation_correlations = generation_correlations.sort_values(
            by="score", ascending=True
        )
        if args.pretty_plot:
            # remove the two rows in the middle of the data frame
            middle_index = len(generation_correlations) // 2
            generation_correlations = pd.concat(
                [
                    generation_correlations.iloc[: middle_index - 1],
                    generation_correlations.iloc[middle_index + 1 :],
                ]
            )

        labels = generation_correlations.index
        labels_split = []
        for label in labels:
            label = label.strip()
            if args.pretty_plot:
                label = PRETTY_LABELS[args.task_name][category][label]
            else:
                label_words = label.split()
                label = "\n".join(
                    [
                        " ".join(label_words[: len(label_words) // 2]),
                        " ".join(label_words[len(label_words) // 2 :]),
                    ]
                )
            labels_split.append(label)

        # find mean score for different elements in the category with pd groupby
        # rotate xticks
        qualitative_colors = sns.color_palette("Set2", 10)
        sns.set_theme(style="white")
        sns.set_palette(qualitative_colors)
        sns.set_style("white")

        colours = color_dict[category]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "colour_map", colours, N=256
        )
        norm = mpl.colors.Normalize(
            generation_correlations["score"].min(),
            generation_correlations["score"].max(),
        )  # linearly normalizes data into the [0.0, 1.0] interval

        bar_plot = sns.barplot(
            ax=axes,
            x=labels_split,
            y=generation_correlations["score"],
            palette=cmap(norm(generation_correlations["score"])),
            linewidth=2,
        )
        # add labels
        axes.tick_params(axis="y", labelsize=font_size)
        axes.set_ylabel("Distance", fontsize=font_size, labelpad=20)
        axes.set_ylim(
            max(generation_correlations["score"].min() - 0.05, 0),
            # min(generation_correlations["score"].max(), 1),
            1,
        )
        xlabels = axes.get_xticklabels()
        axes.spines[["right", "top", "left"]].set_visible(False)
        axes.spines["bottom"].set_linewidth(1.5)
        axes.spines["bottom"].set_color("grey")

        # loop through bars and add annotations
        for j, bar in enumerate(bar_plot.patches):
            # Get the x-coordinate of the bar
            x = bar.get_x()
            # Get the y-coordinate of the bar
            y = bar.get_y()
            # add the text
            axes.text(
                x=x + bar.get_width() / 2,
                y=y + bar.get_height() + 0.01,
                s=xlabels[j].get_text(),
                ha="center",
                va="bottom",
                fontsize=font_size,
                rotation=90,
                multialignment="left",
            )

        axes.set_xticklabels([])
        # save the scores for each category element
        generation_correlations.to_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"generation_correlations_{category}.csv",
            )
        )
    fig.suptitle(
        "Alignment between usage of skills",
        fontsize=1.5 * font_size,
    )
    plt.tight_layout(h_pad=2, w_pad=2, pad=2)
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "correlation_breakdown.pdf"),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "correlation_breakdown.png"),
        dpi=300,
        transparent=True,
    )


def get_proficiency_breakdown(args, all_scores_gt, proficiency_scores):
    # initialize the reportcard plot
    # get the LP assignments
    categories = args.categories.split(",")
    fig, axes = plt.subplots(ncols=len(categories), figsize=(30, 18), sharey=True)
    categories_importance_scores = {}
    for category in categories:
        category_df_gt = all_scores_gt[all_scores_gt["category_type"] == category]
        # find mean score for different elements in the category with pd groupby
        grouped_category_df_gt = category_df_gt.groupby(category_df_gt["category"])
        scores_per_category_type = grouped_category_df_gt["score"].mean()
        categories_importance_scores[category] = scores_per_category_type
    dataset_assignments = get_dataset_assignment_LP(
        args,
        categories_importance_scores,
        all_scores_gt,
        categories,
        max_assignments_per_data_point=2,
    )
    color_dict = {"subtask": ["#F4D941", "#EC8235"], "domain": ["#bbdefb", "#2196f3"]}
    # now we have the assignments for each category element
    for i, category in enumerate(categories):
        cur_assignment = dataset_assignments[category]
        generation_scores_with_assignments = {}
        qualitative_samples = pd.DataFrame()
        for category_element in cur_assignment:
            # filter based on assignments
            # the index of proficiency scores is the id
            cur_proficiency_scores = proficiency_scores[
                proficiency_scores.index.isin(cur_assignment[category_element])
            ]
            generation_scores_with_assignments[category_element] = [
                cur_proficiency_scores["proficiency_score"].mean(),
                len(cur_proficiency_scores),
            ]

            # output some qualitative samples
            # get the top 3 and bottom 3 generations for each category element
            top_generations = cur_proficiency_scores.sort_values(
                by="proficiency_score", ascending=False
            )[:3]
            top_generations["category_element"] = category_element
            bottom_generations = cur_proficiency_scores.sort_values(
                by="proficiency_score", ascending=True
            )[:3]
            bottom_generations["category_element"] = category_element
            qualitative_samples = pd.concat(
                [qualitative_samples, top_generations, bottom_generations]
            )

            # average the scores for each category element given the assignments
        # plot the scores for each category element
        generation_scores_with_assignments_df = pd.DataFrame.from_dict(
            generation_scores_with_assignments, orient="index"
        )
        generation_scores_with_assignments_df.columns = ["score", "num_samples"]
        # sort based on score before plotting
        generation_scores_with_assignments_df = (
            generation_scores_with_assignments_df.sort_values(
                by="score", ascending=True
            )
        )

        generation_scores_with_assignments_df.to_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"generation_scores_with_assignments_{category}.csv",
            )
        )
        if args.pretty_plot:
            # remove the two rows in the middle of the data frame
            middle_index = len(generation_scores_with_assignments_df) // 2
            generation_scores_with_assignments_df = pd.concat(
                [
                    generation_scores_with_assignments_df.iloc[: middle_index - 1],
                    generation_scores_with_assignments_df.iloc[middle_index + 1 :],
                ]
            )

        labels = generation_scores_with_assignments_df.index
        labels_split = []
        for label in labels:
            label = label.strip()
            if args.pretty_plot:
                label = PRETTY_LABELS[args.task_name][category][label]
            else:
                label_words = label.split()
                label = "\n".join(
                    [
                        " ".join(label_words[: len(label_words) // 2]),
                        " ".join(label_words[len(label_words) // 2 :]),
                    ]
                )
            labels_split.append(label)

        qualitative_colors = sns.color_palette("Set2", 10)
        sns.set_theme(style="white")
        sns.set_palette(qualitative_colors)
        sns.set_style("white")

        colours = color_dict[category]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "colour_map", colours, N=256
        )
        norm = mpl.colors.Normalize(
            generation_scores_with_assignments_df["score"].min(),
            generation_scores_with_assignments_df["score"].max(),
        )  # linearly normalizes data into the [0.0, 1.0] interval

        sns.barplot(
            ax=axes[i],
            x=labels_split,
            y=generation_scores_with_assignments_df["score"],
            capsize=10,
            errwidth=10,
            palette=cmap(norm(generation_scores_with_assignments_df["score"])),
        )
        # add labels
        axes[i].tick_params(axis="y", labelsize=font_size * 1.3)
        axes[i].set_title(
            f"{inflect_engine.plural_noun(category.capitalize())}",
            fontsize=font_size * 1.9,
        )
        ylabel = args.proficiency_metric
        axes[i].set_ylabel(ylabel, ha="center", fontsize=font_size * 1.4, labelpad=20)
        axes[i].set_ylim(
            max(generation_scores_with_assignments_df["score"].min() - 0.1, 0),
            min(generation_scores_with_assignments_df["score"].max() + 0.1, 1),
        )
        xlabels = axes[i].get_xticklabels()
        axes[i].set_xticklabels(
            xlabels,
            rotation=90,
            ha="center",
            fontsize=font_size * 1.6,
            multialignment="right",
        )

        axes[i].spines[["right", "top", "left"]].set_visible(False)
        axes[i].spines["bottom"].set_linewidth(1.5)
        axes[i].spines["bottom"].set_color("grey")

        # store the qualitative samples
        qualitative_samples.to_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"qualitative_samples_{category}.csv",
            )
        )
        # dump the LP assignments
        index_2_category_element = {}
        for category_element in cur_assignment:
            for index in cur_assignment[category_element]:
                if index not in index_2_category_element:
                    index_2_category_element[index] = []
                index_2_category_element[index].append(category_element)
        index_2_category_element_df = pd.DataFrame.from_dict(
            index_2_category_element, orient="index"
        )
        index_2_category_element_df = index_2_category_element_df.join(
            proficiency_scores_df, how="inner"
        )
        index_2_category_element_df = index_2_category_element_df.sort_index()
        index_2_category_element_df.to_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"index_2_category_element_{category}.csv",
            )
        )
    fig.suptitle("Proficiency by category", fontsize=2 * font_size, font="roboto")
    plt.tight_layout(h_pad=2, w_pad=2, pad=2)
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "proficiency_breakdown.pdf"),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "proficiency_breakdown.png"),
        dpi=300,
        transparent=True,
    )


def get_nl_summary(args):
    # show the list of categories and category elements
    # list the breakdown of the category elements as json
    # list the breakdown of the proficiency scores as json
    # list the breakdown of the correlation scores as json if they exist
    # prompt the model to generate a NL summary
    task_instruction = TASK_INSTRUCTIONS[args.task_name]
    categories = args.categories.split(",")
    gt_breakdown = {}
    proficiency_scores = {}
    correlation_scores = {}
    category_elements = {}
    for category in categories:
        # load the ground truth breakdown
        gt_breakdown[category] = pd.read_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"gt_scores_per_category_element_{category}.csv",
            ),
        )
        proficiency_scores[category] = pd.read_csv(
            os.path.join(
                args.input_dir_generation_scores,
                f"generation_scores_with_assignments_{category}.csv",
            ),
        )
        if os.path.exists(
            os.path.join(
                args.input_dir_generation_scores,
                f"generation_correlations_{category}.csv",
            ),
        ):
            correlation_scores[category] = pd.read_csv(
                os.path.join(
                    args.input_dir_generation_scores,
                    f"generation_correlations_{category}.csv",
                ),
            )
        category_elements[category] = (
            gt_breakdown[category]["category"].unique().tolist()
        )
        # prepare the dataframes for the NL summary
        proficiency_scores[category] = proficiency_scores[category].set_index(
            "Unnamed: 0"
        )
        proficiency_scores[category] = proficiency_scores[category]["score"].to_dict()
        gt_breakdown[category] = gt_breakdown[category].set_index("category")
        gt_breakdown[category] = gt_breakdown[category]["score"].to_dict()
        if category in correlation_scores:
            correlation_scores[category] = correlation_scores[category].set_index(
                "Unnamed: 0"
            )
            correlation_scores[category] = correlation_scores[category][
                "score"
            ].to_dict()
    # compose request to LLM
    task_instruction_message = f"A machine learning model is tasked with the following task: \n f{task_instruction}"
    category_list = {
        category: "\n".join(category_elements[category]) for category in categories
    }
    category_message = [
        f"These are the {inflect_engine.plural_noun(category)} for the task:\n {category_list[category]}"
        for category in categories
    ]
    category_message = "\n\n".join(category_message)
    gt_breakdown_message = [
        f"In the evaluation data, these are the importance scores of the {inflect_engine.plural_noun(category)}:\n {json.dumps(gt_breakdown[category])}"
        for category in categories
    ]
    gt_breakdown_message = "\n\n".join(gt_breakdown_message)
    proficiency_scores_message = [
        f"The following scores show how well the model performs on the {inflect_engine.plural_noun(category)}: {json.dumps(proficiency_scores[category])}"
        for category in categories
    ]
    proficiency_scores_message = "\n\n".join(proficiency_scores_message)
    if args.task_name not in CLASSIFICATION_TASKS:
        correlation_scores_message = [
            f"The following distance demonstrates how much the {inflect_engine.plural_noun(category)} are actually used for generating the output when they are requried to generate the input. Therefore, a low distance implies that the model is utilizing the category when it needs to: {json.dumps(correlation_scores[category])}. [Important] Lower distance implies the {category} is leveraged when it needs to be used."
            for category in ["subtask"]
        ]
        correlation_scores_message = "\n\n".join(correlation_scores_message)
    else:
        correlation_scores_message = ""
    summarization_message = "Given the above information, please write a brief summary highlighting important information. Please be precise and concise but please be comprehensive."
    system_prompt = "Given a holistic picture of the performance of a machine learning model, you are asked to summarize the model's overall performance."
    try:
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": task_instruction_message,
                },
                {
                    "role": "user",
                    "content": category_message,
                },
                {
                    "role": "user",
                    "content": gt_breakdown_message,
                },
                {
                    "role": "user",
                    "content": proficiency_scores_message,
                },
                {
                    "role": "user",
                    "content": correlation_scores_message,
                },
                {
                    "role": "user",
                    "content": summarization_message,
                },
            ],
            temperature=args.temperature,
            max_tokens=1700,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
        )
    except:
        print(
            "exception encountered while creating pruned set of categories, skipping this iteration"
        )
        return
    if (
        "error" in response
        or "choices" not in response
        or len(response["choices"]) == 0
    ):
        return
    response_text = response["choices"][0]["message"]["content"]
    # save the response
    with open(
        os.path.join(args.input_dir_generation_scores, "NLsummary.txt"), "w"
    ) as f:
        f.write(response_text)
    print(response_text)
    # save the response as an image
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.text(
        0.5,
        0.5,
        response_text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=0.9 * font_size,
        wrap=True,
        font="roboto",
        multialignment="left",
        backgroundcolor="lavender",
    )
    ax.axis("off")
    plt.tight_layout(h_pad=2, w_pad=2, pad=2)
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "NLsummary.pdf"),
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        os.path.join(args.input_dir_generation_scores, "NLsummary.png"),
        dpi=300,
        transparent=True,
    )


def get_reportcard(args, all_scores_generations, all_scores_gt, proficiency_scores):
    # initialize the reportcard plot
    categories = args.categories.split(",")
    # preprocessing
    all_scores_generations, all_scores_gt, pruned_category_elements = preprocessing(
        args, all_scores_generations, all_scores_gt, proficiency_scores
    )
    # get the gt breakdown
    get_gt_breakdown(args, all_scores_gt)
    # get the proficiency scores for each category element
    get_proficiency_breakdown(args, all_scores_gt, proficiency_scores)
    # get correlation between gt and generation scores for each category elements
    if args.task_name not in CLASSIFICATION_TASKS:
        get_correlation_breakdown(
            args, all_scores_generations, all_scores_gt, proficiency_scores
        )
    # get the NL summary
    get_nl_summary(args)
    # concatenate the different images into a single reportcard
    # load images with cv2
    all_images = []
    gt_breakdown_image = cv2.imread(
        os.path.join(args.input_dir_generation_scores, "gt_breakdown.png")
    )
    proficiency_breakdown_image = cv2.imread(
        os.path.join(args.input_dir_generation_scores, "proficiency_breakdown.png")
    )
    all_images.append(gt_breakdown_image)
    all_images.append(proficiency_breakdown_image)
    if args.task_name not in CLASSIFICATION_TASKS:
        correlation_breakdown_image = cv2.imread(
            os.path.join(args.input_dir_generation_scores, "correlation_breakdown.png")
        )
        correlation_image_dummy = (
            np.ones(
                (
                    correlation_breakdown_image.shape[0],
                    correlation_breakdown_image.shape[1] // 2,
                    3,
                )
            ).astype(np.uint8)
            * 255
        )
        # horizontal concatenation of correlation breakdown and dummy image
        correlation_breakdown_image = np.concatenate(
            (
                correlation_image_dummy,
                correlation_breakdown_image,
                correlation_image_dummy,
            ),
            axis=1,
        )

        all_images.append(correlation_breakdown_image)

    nl_summary = cv2.imread(
        os.path.join(args.input_dir_generation_scores, "NLsummary.png")
    )
    all_images.append(nl_summary)
    img_resize = vconcat_resize(all_images)
    cv2.imwrite(
        os.path.join(args.input_dir_generation_scores, "reportcard.png"), img_resize
    )
    # convert the reportcard to pdf
    img = Image.open(
        os.path.join(args.input_dir_generation_scores, "reportcard.png")
    ).convert("RGB")
    img.save(
        os.path.join(args.input_dir_generation_scores, "reportcard.pdf"),
        save_all=True,
    )

    # dump a zip file with all the data

    zipf = zipfile.ZipFile(
        os.path.join(args.input_dir_generation_scores, "reportcard.zip"), "w"
    )
    with zipf:
        zipf.write(
            os.path.join(args.input_dir_generation_scores, "reportcard.pdf"),
            "dashboard.pdf",
        )
        zipf.write(
            os.path.join(args.input_dir_generation_scores, "gt_breakdown.pdf"),
            "prior_over_categories.pdf",
        )
        zipf.write(
            os.path.join(args.input_dir_generation_scores, "proficiency_breakdown.pdf"),
            "proficiency_over_categories.pdf",
        )
        if args.task_name not in CLASSIFICATION_TASKS:
            zipf.write(
                os.path.join(
                    args.input_dir_generation_scores, "correlation_breakdown.pdf"
                ),
                "distance_bw_GT_and_Output.pdf",
            )
        zipf.write(
            os.path.join(args.input_dir_generation_scores, "NLsummary.pdf"),
            "summary.pdf",
        )
        zipf.write(
            os.path.join(args.input_dir_generation_scores, "NLsummary.txt"),
            "summary.txt",
        )
        # also add the data files
        for category in categories:
            zipf.write(
                os.path.join(
                    args.input_dir_generation_scores,
                    f"gt_scores_per_category_element_{category}.csv",
                ),
                f"prior_over_category_elements_{category}.csv",
            )
            zipf.write(
                os.path.join(
                    args.input_dir_generation_scores,
                    f"generation_scores_with_assignments_{category}.csv",
                ),
                f"proficiency_over_category_elements_{category}.csv",
            )
            if args.task_name not in CLASSIFICATION_TASKS:
                if category == "subtask":
                    zipf.write(
                        os.path.join(
                            args.input_dir_generation_scores,
                            f"generation_correlations_{category}.csv",
                        ),
                        f"distance_bw_GT_and_Output_{category}.csv",
                    )


if __name__ == "__main__":
    # get the model generation prompts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        type=str,
        default="subtask,domain",
        help="Categories to use for the reportcard",
    )
    parser.add_argument(
        "--input_dir_generation_scores",
        type=str,
        default="",
        help="Input directory for finding the generation scores",
    )
    parser.add_argument(
        "--generation_file",
        type=str,
        default="",
        help="Input file for finding the generations",
    )
    parser.add_argument(
        "--proficiency_metric",
        type=str,
        default="",
        help="Proficiency metric to use for the reportcard",
    )
    parser.add_argument(
        "--input_dir_gt_scores",
        type=str,
        default="",
        help="Input directory for finding the ground truth scores",
    )
    parser.add_argument(
        "--pretty_plot",
        action="store_true",
        help="Whether to use pretty plots or not",
    )
    parser = add_args(parser)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=args.logging_level)
    # Random seed
    seed_function(args)
    api_key = authenticate(args)
    all_scores_generations = {}
    all_scores_gt = {}
    # load both the generation scores and the ground truth scores
    for category in args.categories.split(","):
        # get the generation score for this category
        generation_score_file = os.path.join(
            args.input_dir_generation_scores, f"{category}_scores.csv"
        )
        generation_scores = pd.read_csv(generation_score_file)
        # get the ground truth score for this category
        gt_score_file = os.path.join(args.input_dir_gt_scores, f"{category}_scores.csv")
        gt_scores = pd.read_csv(gt_score_file)
        # add to the dictionary
        all_scores_generations[category] = generation_scores
        all_scores_gt[category] = gt_scores
    # load the proficiency scores from the generation jsonl file
    proficiency_scores = {}
    proficiency_metric = ""
    with open(args.generation_file) as f:
        assert (
            args.proficiency_metric in PROFICIENCY_METRICS[args.task_name]
        ), "Proficiency metric not supported"
        proficiency_metric = f"generation_{args.proficiency_metric}"
        for line in f:
            line = json.loads(line)
            proficiency_scores[line["id"]] = line
    # convert the proficiency scores to a dataframe
    proficiency_scores_df = pd.DataFrame.from_dict(proficiency_scores, orient="index")
    # rename the column
    proficiency_scores_df = proficiency_scores_df.rename(
        columns={proficiency_metric: "proficiency_score"}
    )
    # get the reportcard
    get_reportcard(args, all_scores_generations, all_scores_gt, proficiency_scores_df)
