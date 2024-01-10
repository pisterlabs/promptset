"""
model_sketch_book
---
Classes and functions to support model prototyping.
"""

import numpy as np
import pandas as pd
import altair as alt

import ipywidgets as widgets
from ipywidgets import fixed, interact_manual

import altair as alt
from nltk.corpus import wordnet
import openai

from .msb_enums import (
    InputType,
    OutputType,
    ModelType,
    SketchSortMode,
)
from .SketchBook import SketchBook
from .Sketch import Sketch
from .Concept import (
    ImageConcept,
    GPTTextConcept,
    LogicalConcept,
    KeywordConcept,
)
from .helper_functions import *

# Set up IPyWidgets
widgets.interact_manual.opts["manual_name"] = "See results"


# Function to create a sketchbook
# Currently unused
def create_model_sketchbook(
    goal,
    schema,
    credentials=None,
):
    if credentials is None:
        set_openai_credentials()
    else:
        openai.organization = credentials["organization"]
        openai.api_key = credentials["api_key"]
    sketchbook = SketchBook(goal, schema)
    return sketchbook


def show_concepts(sb):
    concepts = sb.concepts
    for concept_id, c in concepts.items():
        print(
            f"Concept ID: {concept_id}\n\tConcept term: {c.concept_term}\n\tInput field: {c.input_field}\n\tOutput type: {c.output_type.name}"
        )
        if concept_id in sb.notes:
            print("\tNotes:")
            print(f"\t\t{sb.notes[concept_id]}")


def show_sketches(sb):
    sketches = sb.sketches
    for sketch_id, s in sketches.items():
        concepts = [s._readable_concept(c_id) for c_id in s.concepts]
        print(
            f"Sketch ID: {sketch_id}\n\tModel type: {s.model_type.name}\n\tOutput type: {s.output_type.name}"
        )
        print("\tConcepts:")
        for concept_str in concepts:
            print(f"\t\t{concept_str}")
        if sketch_id in sb.notes:
            print("\tNotes:")
            print(f"\t\t{sb.notes[sketch_id]}")


# Function to call to create concept from widget
def create_concept_model_widget(
    sb,
    concept_term,
    input_field,
    output_type,
):
    _, df = create_concept_model_internal(
        sb,
        concept_term,
        input_field,
        output_type,
        is_testing=False,
        is_styled=True,
    )
    return df


# Internal function to create concept (from direct call, widget, or web app)
def create_concept_model_internal(
    sb,
    concept_term,
    input_field,
    output_type,
    is_testing=False,
    is_styled=True,
):
    # Create the concept
    print("Creating concept...")
    input_type = sb.schema[input_field]
    if input_type == InputType.Image or input_type == InputType.ImageLocal:
        concept = ImageConcept(sb, concept_term, input_field, output_type)
    elif input_type == InputType.Text:
        concept = GPTTextConcept(sb, concept_term, input_field, output_type)
    elif input_type == InputType.Number:
        raise Exception(f"The {input_type} input type has not yet been implemented.")

    # Run the concept on data
    print("Running concept on data...")
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")
    _ = concept.run(dataset_id, is_testing)

    # Visualize the concept results
    print("Visualizing concept results...")
    df = concept.visualize(dataset_id, is_styled)

    return concept, df


# Function to show a widget for creating an image or text concept model
def create_concept_model(sb):
    concept_term = widgets.Text(
        value=None,
        placeholder="Enter concept term (max: 1-2 words)",
        description="Concept term: ",
        disabled=False,
        style=dict(description_width="initial"),
    )

    input_field_options = [""] + [
        input_field
        for input_field, input_type in sb.schema.items()
        if input_type is not InputType.GroundTruth
    ]
    input_field = widgets.Dropdown(
        options=input_field_options,
        value="",
        description="Input field: ",
        disabled=False,
    )

    # Output type shown dynamically depending on input type
    output_type_options = [
        ("continuous scores (CLIP)", OutputType.Continuous),
        ("binary labels (GPT-3)", OutputType.Binary),
    ]
    output_type = widgets.RadioButtons(
        options=output_type_options,
        value=OutputType.Continuous,
        description="Output type: ",
        disabled=False,
    )
    output_type.layout.visibility = "hidden"  # Initially hidden

    def update_binarize_cbox(*args):
        if input_field.value == "":
            # Hide output options
            output_type.layout.visibility = "hidden"
        elif (sb.schema[input_field.value] == InputType.Image) or (
            sb.schema[input_field.value] == InputType.ImageLocal
        ):
            # Hide output options
            output_type.layout.visibility = "hidden"
        else:
            # Show output options if image field is selected
            output_type.layout.visibility = "visible"

    input_field.observe(update_binarize_cbox)

    interact_manual(
        create_concept_model_widget,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        concept_term=concept_term,
        input_field=input_field,
        output_type=output_type,
    )


# Function to create a logical concept model
def create_logical_concept_model_internal(
    sb,
    subconcept_ids=[],
    operator="AND",
):
    concept = LogicalConcept(
        sb=sb,
        concept_term=None,
        input_field=None,
        output_type=OutputType.Binary,
        subconcept_ids=subconcept_ids,
        operator=operator,
    )

    # Run the concept on data
    print("Running concept on data...")
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")
    _ = concept.run(dataset_id, is_testing=False)

    # Visualize the concept results
    print("Visualizing concept results...")
    df = concept.visualize(dataset_id, is_styled=True)

    return df


# Function to show a widget for creating a logical concept model
def create_logical_concept_model(sb):
    concept_ids_select = get_available_concept_widget(
        sb,
        only_binary=True,
    )

    operator = widgets.RadioButtons(
        options=["AND", "OR"],
        value="AND",
        description="Operator: ",
        disabled=False,
    )

    interact_manual(
        create_logical_concept_model_internal,
        sb=fixed(sb),
        subconcept_ids=concept_ids_select,
        operator=operator,
    )


# Function to create a keyword concept model
def create_keyword_concept_model_internal(
    sb,
    input_field,
    keywords_str,
    case_sensitive,
):
    keywords = keywords_str.split(",")
    keywords = [k.strip() for k in keywords]
    concept = KeywordConcept(
        sb=sb,
        concept_term=None,
        input_field=input_field,
        output_type=OutputType.Binary,
        keywords=keywords,
        case_sensitive=case_sensitive,
    )

    # Run the concept on data
    print("Running concept on data...")
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")
    _ = concept.run(dataset_id, is_testing=False)

    # Visualize the concept results
    print("Visualizing concept results...")
    df = concept.visualize(dataset_id, is_styled=True)

    return df


# Function to show a widget for creating a keyword concept model
def create_keyword_concept_model(sb):
    text_input_fields = [
        input_field
        for input_field, input_type in sb.schema.items()
        if input_type == InputType.Text
    ]
    input_field_options = [""] + text_input_fields
    input_field = widgets.Dropdown(
        options=input_field_options,
        value="",
        description="Input field: ",
        disabled=False,
    )

    keywords = widgets.Textarea(
        value="",
        placeholder="Enter a comma-separated list of keywords",
        description="Keywords: ",
        disabled=False,
    )

    case_sensitive = widgets.Checkbox(
        value=True, description="Case-sensitive?", disabled=False
    )

    interact_manual(
        create_keyword_concept_model_internal,
        sb=fixed(sb),
        input_field=input_field,
        keywords_str=keywords,
        case_sensitive=case_sensitive,
    )


# Function to call to create sketch from widget
def create_sketch_model_widget(
    sb,
    concepts,
    model_type,
    sort_mode,
):
    _, df = create_sketch_model_internal(
        sb,
        concepts,
        model_type,
        sort_mode,
    )
    return df


# Function to create a sketch model
def create_sketch_model_internal(
    sb,
    concepts,
    model_type,
    sort_mode,
):
    # Create the sketch
    print("Creating sketch...")
    output_type = OutputType.Continuous  # TEMP
    sketch = Sketch(sb, concepts, model_type, sort_mode, output_type)

    # Train the sketch on data
    print("Training sketch on data...")
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")
    sketch.train(dataset_id)

    # Visualize the sketch results
    print("Visualizing sketch results...")
    df = sketch.visualize(dataset_id)
    return sketch, df


def get_available_concepts(sb, only_binary=False):
    available_concept_ids = [
        (
            f"{c_id}: {c.concept_term}, input={c.input_field}, output={c.output_type.name.lower()}",
            c_id,
        )
        for c_id, c in sb.concepts.items()
        if (len(c.cached_datasets) > 0) and (not only_binary or c._has_binary_output())
    ]
    return available_concept_ids


def get_available_concept_widget(
    sb, allow_multiple=True, only_binary=False, label="Concept IDs: "
):
    available_concept_ids = get_available_concepts(sb, only_binary)
    if allow_multiple:
        concept_ids_select = widgets.SelectMultiple(
            options=available_concept_ids,
            value=[],
            rows=10,
            description=label,
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style=dict(description_width="initial"),
        )
    else:
        concept_ids_select = widgets.Select(
            options=available_concept_ids,
            value=None,
            rows=10,
            description=label,
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style=dict(description_width="initial"),
        )
    return concept_ids_select


# Function to show a widget for creating a sketch model
def create_sketch_model(sb):
    concept_ids_select = get_available_concept_widget(sb)

    model_type_options = [
        ("Linear Regression", ModelType.LinearRegression),
        ("Logistic Regression", ModelType.LogisticRegression),
        ("MLP", ModelType.MLP),
        ("Decision Tree", ModelType.DecisionTree),
        ("Random Forest", ModelType.RandomForest),
        # ("Zero Shot", ModelType.ZeroShot)
    ]
    model_type_dropdown = widgets.Dropdown(
        options=model_type_options,
        value=ModelType.LinearRegression,
        description="Aggregator: ",
        disabled=False,
        style=dict(description_width="initial"),
    )
    sort_options = [
        ("Sketch Prediction", SketchSortMode.SketchPred),
        ("Ground Truth", SketchSortMode.GroundTruth),
        ("Diff (Sketch Pred - Ground Truth)", SketchSortMode.Diff),
        ("Abs Val of Diff", SketchSortMode.AbsDiff),
    ]
    sort_mode_dropdown = widgets.Dropdown(
        options=sort_options,
        value=SketchSortMode.SketchPred,
        description="Sort order: ",
        disabled=False,
        style=dict(description_width="initial"),
    )

    interact_manual(
        create_sketch_model_widget,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        concepts=concept_ids_select,
        model_type=model_type_dropdown,
        sort_mode=sort_mode_dropdown,
    )


def get_available_sketches(sb):
    available_sketch_ids = [
        (
            f"{s_id}: model_type={s.model_type.name}, output={s.output_type.name.lower()}",
            s_id,
        )
        for s_id, s in sb.sketches.items()
        if (s.model is not None)
    ]
    return available_sketch_ids


def get_available_sketch_widget(sb, allow_multiple=True):
    available_sketch_ids = get_available_sketches(sb)
    if allow_multiple:
        sketch_ids_select = widgets.SelectMultiple(
            options=available_sketch_ids,
            value=[],
            rows=10,
            description="Sketch IDs: ",
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style=dict(description_width="initial"),
        )
    else:
        sketch_ids_select = widgets.Select(
            options=available_sketch_ids,
            value=None,
            rows=10,
            description="Sketch ID: ",
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style=dict(description_width="initial"),
        )
    return sketch_ids_select


def get_available_dataset_widget(sb, only_labeled=False):
    available_dataset_ids = [
        (f"{d_id}: length={len(d.df)}, labeled={d.labeled}", d_id)
        for d_id, d in sb.datasets.items()
        if (not only_labeled or d.labeled)
    ]
    dataset_id_select = widgets.Select(
        options=available_dataset_ids,
        value=None,
        rows=5,
        description="Dataset ID: ",
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style=dict(description_width="initial"),
    )
    return dataset_id_select


def test_sketch_internal(sb, sketch_id, dataset_id):
    sketch = sb.get_sketch(sketch_id)
    print("Evaluating sketch on data...")
    _ = sketch.eval(dataset_id)

    # Visualize the sketch results
    print("Visualizing sketch results...")
    df = sketch.visualize(dataset_id)
    return df


# Function to test a trained sketch model on a specified dataset
def test_sketch(sb):
    # Sketch select
    sketch_id_select = get_available_sketch_widget(sb, allow_multiple=False)

    # Dataset select
    dataset_id_select = get_available_dataset_widget(sb)
    interact_manual(
        test_sketch_internal,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        sketch_id=sketch_id_select,
        dataset_id=dataset_id_select,
    )


# Function to generate a single bar chart plot of a specified metric
# - df: The dataframe containing each sketch's performance metrics in a row with columns representing different metrics
# - perf_col: The name of the dataframe column containing the current metric to plot
# - perf_readable_name: The name of the metric to use in the chart title
# - sort: The preferred ordering of the sketches on the x-axis
def get_perf_plot(df, perf_col, perf_readable_name, sort):
    chart = (
        alt.Chart(df.sample(df.shape[0]))
        .mark_bar()
        .encode(
            y=alt.Y(
                f"{perf_col}:Q",
                title=perf_readable_name,
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("Sketch:N", sort=sort),
            x=alt.X(
                "Sketch:O",
                title="",
                axis=alt.Axis(labelAngle=-45),
                sort=sort,
            ),
        )
    )
    return chart


# Function to aggregate all performance metric results and display them in a set of side-by-side charts.
# - all_perf: The dictionary containing all performance metric values. Keys are sketch names, and values are dictionaries which have keys of metric names and values of perf metric values.
# - sort: The preferred ordering of the sketches on the x-axis; if populated, should be a list of the sketch names. (Defaults to "x", which in Altair just sorts the values in ascending alphabetical order)
def get_performance_summary(all_perf, sort="x"):
    rows = []
    for sketch_id, v in all_perf.items():
        # Add performance result
        mae = v["mae"]
        acc = v["acc"]
        f1 = v["f1"]
        prec = v["prec"]
        recall = v["recall"]
        row = [sketch_id, mae, acc, f1, prec, recall]
        rows.append(row)

    res_df = pd.DataFrame(
        rows, columns=["Sketch", "MAE", "Accuracy", "F1", "Precision", "Recall"]
    )

    # Generate plots
    mae_plot = get_perf_plot(res_df, "MAE", "Mean Absolute Error", sort)
    f1_plot = get_perf_plot(res_df, "F1", "F1 score", sort)
    acc_plot = get_perf_plot(res_df, "Accuracy", "Accuracy", sort)
    prec_plot = get_perf_plot(res_df, "Precision", "Precision", sort)
    recall_plot = get_perf_plot(res_df, "Recall", "Recall", sort)

    return alt.concat(mae_plot, f1_plot, acc_plot, prec_plot, recall_plot)


def compare_sketches_internal(sb, sketch_ids, dataset_id):
    # Check that the dataset is labeled
    dataset = sb.datasets[dataset_id]
    assert (
        dataset.labeled
    ), f"Dataset `{dataset_id}` is not labeled. Only results on labeled datasets can be compared by the `compare_sketches()` function."
    # Check that every sketch already has a trained model
    sketches = {}
    for sketch_id in sketch_ids:
        sketch = sb.get_sketch(sketch_id)
        assert (
            sketch.model is not None
        ), f"Sketch `{sketch.id}` does not have a trained model. Please run the `train()` function on a training dataset before using this sketch in a comparison."
        sketches[sketch_id] = sketch

    # Generate performance results for each sketch model on this dataset
    all_perf = {}
    for sketch_id, sketch in sketches.items():
        _ = sketch.eval(dataset_id)
        all_perf[sketch_id] = sketch.cached_perf[dataset_id]

    # Combine performance results in performance plot
    chart = get_performance_summary(all_perf, sort=sketch_ids)
    return chart


# Function to compare two sketch models' performance on a specified dataset
# The dataset must be labeled to be eligible for comparison
def compare_sketches(sb):
    # Sketch select
    sketch_ids_select = get_available_sketch_widget(sb, allow_multiple=True)

    # Dataset select
    dataset_id_select = get_available_dataset_widget(sb, only_labeled=True)
    interact_manual(
        compare_sketches_internal,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        sketch_ids=sketch_ids_select,
        dataset_id=dataset_id_select,
    )


# Function to set concept tuning parameters and re-run concept results
def tune_concept_internal(
    sb, concept_id, should_threshold, threshold, normalize, should_calib, calib
):
    # Handle params and run + visualize model with those params
    tune_params = {
        "threshold": (threshold if should_threshold else None),
        "normalize": normalize,
        "calib": (calib if should_calib else None),
    }
    concept = sb.get_concept(concept_id)

    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")

    # Visualize the concept results after tuning params have been set
    print("Visualizing concept results...")
    concept.tune_params = tune_params
    df = concept.visualize(dataset_id, is_styled=True)
    return df


# Handles conditionally displaying/hiding a widget based on whether the checkbox_widget is checked/unchecked.
def create_conditional_widget(checkbox_widget, widget_to_show):
    widget_to_show.layout.display = "none"  # Initially hidden

    def update_widget_vis(*args):
        if checkbox_widget.value:
            # Show widget
            widget_to_show.layout.display = "inherit"
        else:
            # Hide widget
            widget_to_show.layout.display = "none"

    checkbox_widget.observe(update_widget_vis)


# Function to show a widget for tuning an existing concept
def tune_concept(sb):
    concept_id_select = get_available_concept_widget(sb, allow_multiple=False)
    concept_id_select.layout.visiblity = "visible"

    normalize = widgets.Checkbox(value=False, description="Normalize?", disabled=False)

    threshold = widgets.Checkbox(value=False, description="Threshold?", disabled=False)

    calib = widgets.Checkbox(value=False, description="Calibrate?", disabled=False)

    threshold_val = widgets.FloatSlider(
        min=0, max=1, value=0.5, step=0.01, description="Threshold: "
    )

    calib_vals = widgets.FloatRangeSlider(
        min=0, max=1, value=[0, 1], step=0.01, description="Calibration: "
    )

    create_conditional_widget(threshold, threshold_val)
    create_conditional_widget(calib, calib_vals)

    interact_manual(
        tune_concept_internal,
        sb=fixed(sb),
        concept_id=concept_id_select,
        should_threshold=threshold,
        threshold=threshold_val,
        normalize=normalize,
        should_calib=calib,
        calib=calib_vals,
    )


# Split the data to achieve balance in overall rating buckets between the train and test sets. In each bucket, examples will be consistently split between the train and test set according to the specified fraction.
# - df_in: The dataset to be split
# - ground_truth: The name of the dataframe column containing the overall ground-truth ratings for examples
# - n_buckets: The number of quantile buckets to create to divide the examples (by their overall ratings)
# - train_frac: The proportion of examples that should be assigned to the training set (the remainder will be assigned to the test set)
def split_balance_overall_rating(df_in, ground_truth: str, n_buckets=5, train_frac=0.5):
    # Stratify into four score buckets
    # For each score bucket, randomly assign to train or test
    df = df_in.copy()
    n_distinct_vals = len(df[ground_truth].unique().tolist())
    if n_buckets >= n_distinct_vals:
        print(
            f"There are only {n_distinct_vals} distinct '{ground_truth}' values, so we have changed to {n_distinct_vals - 1} buckets instead of {n_buckets}"
        )
        n_buckets = n_distinct_vals - 1
    df["bin"] = pd.qcut(df[ground_truth], q=n_buckets, duplicates="drop")
    bins = df["bin"].unique().tolist()

    train = pd.DataFrame([], columns=df.columns)  # empty df with same columns
    test = pd.DataFrame([], columns=df.columns)  # empty df with same columns

    for bin_val in bins:
        df_filt = df[df["bin"] == bin_val]  # filter to current bin
        # Sample half to train and half to test
        df_filt_train = df_filt.sample(frac=train_frac)  # Sample half for train set
        df_filt_test = df_filt.drop(df_filt_train.index)  # Assign the rest to test set
        train = pd.concat([train, df_filt_train])
        test = pd.concat([test, df_filt_test])

    train = train.drop(columns=["bin"])
    test = test.drop(columns=["bin"])

    # Plot resulting distribution
    chart_train = (
        alt.Chart(train)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="count"),
            y=alt.Y(f"{ground_truth}:N"),
        )
        .properties(title="Train distribution")
    )

    chart_test = (
        alt.Chart(test)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="count"),
            y=alt.Y(f"{ground_truth}:N"),
        )
        .properties(title="Test distribution")
    )

    chart = chart_train | chart_test

    return train, test, chart


# Get similar concepts to a concept term based on nltk's wordnet synonym set
# - concept_term: The term that should characterize this concept
def get_similar_concepts(concept_term):
    synonyms = []
    for syn in wordnet.synsets(concept_term):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return set(synonyms)


def compare_concepts_to_gt_internal(sb, concept_ids):
    # Get ground truth data
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")

    # Get concept predictions
    df = sb._get_concept_preds_df(concept_ids, dataset_id)

    # Get input fields
    input_fields = list(sb.schema.keys())
    df = sb.style_input_fields(df, dataset_id, input_fields)
    binary_fields, contin_fields = sb.sep_concept_col_types(concept_ids)

    # Get concept correlations
    correlations = sb._get_concept_gt_correlations(concept_ids, dataset_id)
    df, diff_cols = sb._get_concept_gt_diffs(df, concept_ids, dataset_id)

    # Get overall correlation summary
    print("Pearson correlation coefficients:")
    highest_corr = 0
    highest_corr_concept = None
    for c_id in concept_ids:
        c = sb.concepts[c_id]
        concept_term = c.concept_term
        corr = np.round(correlations[c_id], 3)
        print(f"\tConcept {c_id} ({concept_term}): correlation={corr}")
        if abs(corr) > highest_corr:
            highest_corr = abs(corr)
            highest_corr_concept = c

    print(
        f"\nRecommended concept: {highest_corr_concept.id} ({highest_corr_concept.concept_term}), correlation={highest_corr}"
    )

    # Only show input fields, ground truth, concepts, diffs
    cols_to_show = input_fields + binary_fields + contin_fields + diff_cols
    df = df[cols_to_show]
    df = sb.style_output_fields(df, binary_fields, contin_fields, diff_fields=diff_cols)
    return df


def compare_concepts_to_gt(sb):
    # Concept selector
    concept_ids_select = get_available_concept_widget(sb)

    interact_manual(
        compare_concepts_to_gt_internal,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        concept_ids=concept_ids_select,
    )


def compare_two_concepts_internal(sb, concept_id1, concept_id2):
    # Get ground truth data
    dataset_id = sb.default_dataset_id
    if dataset_id is None:
        raise Exception("Please specify a default dataset")

    # Get concept predictions
    concept_ids = [concept_id1, concept_id2]
    df = sb._get_concept_preds_df(concept_ids, dataset_id)

    # Get input fields
    input_fields = list(sb.schema.keys())
    df = sb.style_input_fields(df, dataset_id, input_fields)
    binary_fields, contin_fields = sb.sep_concept_col_types(concept_ids)

    # Get concept correlations
    correlation = np.corrcoef(df[concept_id1], df[concept_id2])[0, 1]
    diff_col = f"diff ({concept_id1} - {concept_id2})"
    df[diff_col] = df[concept_id1] - df[concept_id2]

    # Calculate correlation; show diff between concept scores
    print(f"Pearson correlation coefficient: {correlation}")

    # Only show input fields, ground truth, concepts, diffs
    cols_to_show = input_fields + binary_fields + contin_fields + [diff_col]
    df = df[cols_to_show]
    df = sb.style_output_fields(
        df, binary_fields, contin_fields, diff_fields=[diff_col]
    )
    return df


def compare_two_concepts(sb):
    concept_id1_select = get_available_concept_widget(
        sb, allow_multiple=False, label="Concept ID 1: "
    )
    concept_id2_select = get_available_concept_widget(
        sb, allow_multiple=False, label="Concept ID 2: "
    )

    interact_manual(
        compare_two_concepts_internal,
        # `fixed` indicates not to create a widget for this argument
        sb=fixed(sb),
        concept_id1=concept_id1_select,
        concept_id2=concept_id2_select,
    )


def take_note_internal(
    sb,
    internal_id,
    notes=None,
):
    if notes is not None:
        sb.notes[internal_id] = notes
    print("Saved!")


def take_note(sb):
    available_concept_ids = get_available_concepts(sb)
    available_sketch_ids = get_available_sketches(sb)

    eligible_ids = available_concept_ids + available_sketch_ids
    internal_id_select = widgets.Select(
        options=eligible_ids,
        value=None,
        rows=10,
        description="Concept or Sketch ID: ",
        disabled=False,
        style=dict(description_width="initial"),
        layout={"width": "max-content"},
    )
    notes_entry = widgets.Textarea(
        value=None,
        placeholder="Enter your notes on the concept or sketch",
        description="Scratch notes: ",
        disabled=False,
        rows=3,
        layout=widgets.Layout(width="50%"),
        style=dict(description_width="initial"),
    )

    save_interact_manual = interact_manual.options(manual_name="Save")
    save_interact_manual(
        take_note_internal,
        sb=fixed(sb),  # `fixed` indicates not to create a widget for this argument
        internal_id=internal_id_select,
        notes=notes_entry,
    )
