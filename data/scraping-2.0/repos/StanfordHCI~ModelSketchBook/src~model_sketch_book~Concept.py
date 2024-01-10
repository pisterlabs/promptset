# CONCEPT

import math
import numpy as np
import re
import torch
import open_clip
import random
import openai

from .msb_enums import InputType, OutputType, idMode
from .SketchBook import SketchBook
from .Result import Result
from .helper_functions import *

from typing import Tuple, Dict

# SETUP
# Set up CLIP
# Load model and preprocess function
CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="laion400m_e32"
)

DEFAULT_TUNE_PARAMS = {
    "threshold": None,
    "normalize": False,
    "calib": None,
}


# Class definition of a concept
# Parameters:
# - concept_term: term that describes or captures the concept
# - input_field: name of column with desired input to concept's model
# - output_type: OutputType enum representing desired output format
# - tune_params: dictionary specifying parameters for tuning (thresholding, normalizing, or calibrating)
# Attributes:
# - id: unique identifier string
# - input_type: InputType enum of desired input's format
# - cached_datasets: list of Dataset object ids that have cached Results
class Concept:
    def __init__(
        self,
        sb: SketchBook,
        concept_term: str,
        input_field: str,
        output_type: OutputType,
        # Optional tuning parameters
        tune_params: Dict[str, any] = DEFAULT_TUNE_PARAMS,
    ):
        self.sb = sb
        self.id = create_id(sb, idMode.Concept)
        self.concept_term = concept_term
        self.input_field = input_field
        self.output_type = output_type
        self.tune_params = tune_params

        self.input_type = sb.schema[input_field] if input_field is not None else None
        self.cached_datasets = set()

        self.sb.add_concept(self, self.id)  # add concept to sketchbook

    def get_concept_key(self):
        return (self.concept_term, self.input_field, self.output_type)

    # Internal helper function to calculate the predictions for a given concept
    def get_preds(
        self,
    ):
        raise Exception(
            f"get_preds() for concept type '{self.__class__}' has not been implemented yet"
        )

    def fetch_from_cache(self, dataset_id):
        # Fetch previously cached results if they exist
        dataset = self.sb.datasets[dataset_id]
        concept_key = self.get_concept_key()
        if concept_key in dataset.cached_concept_res:
            res = dataset.cached_concept_res[concept_key]
            return res
        else:
            return None

    def add_to_cache(self, dataset_id, res):
        # Adds Result to cache for the Dataset if it's not already there
        dataset = self.sb.datasets[dataset_id]
        concept_key = self.get_concept_key()
        if concept_key not in dataset.cached_concept_res:
            dataset.cached_concept_res[concept_key] = res
        self.cached_datasets.add(dataset_id)

    def get_rand_preds(self, dataset_id):
        dataset = self.sb.datasets[dataset_id]
        item_ids = dataset.get_item_ids()
        if self.output_type == OutputType.Continuous:
            preds = {item_id: random.uniform(0, 1) for item_id in item_ids}
        elif self.output_type == OutputType.Binary:
            preds = {item_id: random.choice([True, False]) for item_id in item_ids}
        else:
            raise Exception(f"Output type {self.output_type} not supported.")

        res = Result(self.id, preds)
        return res

    # Main function to apply the concept to a specified dataset
    # Will fetch cached results (if available) or calculate new predictions
    # Includes the logic to perform tuning based on the latest tuning settings (threshold, calibration, normalization)
    def run(
        self,
        dataset_id,
        is_testing=False,
        debug=False,
    ):
        # Fetch predictions (cached if available, random if requested)
        cached_concept_res = self.fetch_from_cache(dataset_id)
        if cached_concept_res is not None:
            if debug:
                print("Fetched from cache")
            res = cached_concept_res
        elif is_testing:
            # Testing mode that bypasses actual concept-specific logic to fetch concept predictions
            res = self.get_rand_preds(dataset_id)
        else:
            # Otherwise, calculate new predictions for this concept on the dataset (the standard case)
            res = self.get_preds(dataset_id)
        self.add_to_cache(dataset_id, res)

        # Apply tuning
        res = self.tune(res)
        return res

    # Helper functions to set tuning parameters
    def _set_threshold(self, threshold: float):
        # threshold: threshold for binarization (values < threshold = False; values >= threshold = True)
        self.tune_params["threshold"] = threshold

    def _get_threshold(self):
        return self.tune_params["threshold"]

    def _set_normalize(self, normalize: bool):
        # normalize: whether to normalize the predictions between 0 and 1 based on the observed minimum and maximum prediction values
        self.tune_params["normalize"] = normalize

    def _get_normalize(self):
        return self.tune_params["normalize"]

    def _set_calib(self, calib: Tuple[float, float]):
        # calib: the pair of values within which to calibrate predictions, as specified by a (calib_min, calib_max) tuple.
        # - calib_min: the minimum value prediction for calibration; all values <= this value will be set to 0.
        # - calib_max: the maximum value prediction for calibration; all values >= this value will be set to 1.
        self.tune_params["calib"] = calib

    def _get_calib(self):
        return self.tune_params["calib"]

    # Sets all tuning parameters back to their default, unset states
    def _clear_tuning(self):
        self.tune_params = DEFAULT_TUNE_PARAMS

    # Applies tuning on concept results (thresholding, normalization, and/or calibration). Tuning is only allowed for concepts with Continuous output types.
    def tune(self, res):
        threshold = self._get_threshold()
        normalize = self._get_normalize()
        calib = self._get_calib()
        has_tune_params = threshold or normalize or calib
        if has_tune_params and self.output_type != OutputType.Continuous:
            raise Warning(
                "Tuning is only supported for concepts with Continuous output types."
            )

        if calib and normalize:
            raise Warning(
                "Both calibration and normalization have been requested; please only apply one of these options at a time. Here, we have applied only normalization."
            )
        if normalize:
            res = self._tune_normalize(res)
        elif calib:
            res = self._tune_calibrate(res, calib)

        if threshold:
            res = self._tune_threshold(res, threshold)

        return res

    # Applies threshold on predictions to binarize scores.
    # Acts on provided Results object
    # Returns Results object with adjusted predictions.
    def _tune_threshold(self, res, threshold):
        preds = res.preds

        def apply_thresh(pred):
            return pred >= threshold

        tuned_preds = {item_id: apply_thresh(pred) for item_id, pred in preds.items()}
        new_res = Result(self.id, tuned_preds)
        return new_res

    # Normalizes the predictions so that min and max map to 0 and 1, respectively.
    # Acts on provided Results object
    # Returns Results object with adjusted predictions.
    def _tune_normalize(self, res):
        preds = res.preds
        # Prep normalization parameters
        pred_vals = list(preds.values())
        pred_min = np.min(pred_vals)
        pred_max = np.max(pred_vals)
        pred_range = pred_max - pred_min

        def apply_norm(pred):
            return (pred - pred_min) / pred_range

        tuned_preds = {item_id: apply_norm(pred) for item_id, pred in preds.items()}
        new_res = Result(self.id, tuned_preds)
        return new_res

    # Calibrate predictions so that provided calib values (calib_min and calib_max) map to 0 and 1, respectively.
    # Acts on provided Results object
    # Returns Results object with adjusted predictions.
    def _tune_calibrate(self, res, calib):
        calib_min, calib_max = calib
        preds = res.preds
        # Prep calibration parameters
        calib_range = calib_max - calib_min

        def apply_calib(pred):
            if pred <= calib_min:
                return 0.0
            elif pred >= calib_max:
                return 1.0
            return (pred - calib_min) / calib_range

        tuned_preds = {item_id: apply_calib(pred) for item_id, pred in preds.items()}
        new_res = Result(self.id, tuned_preds)
        return new_res

    # visualize() helper functions

    def _has_binary_output(self):
        return (self.output_type == OutputType.Binary) or (
            self._get_threshold() is not None
        )

    # Color cells based on concept score
    def _style_concept_score_col(self, df, col_name, output_type):
        if self._has_binary_output():
            df = df.style.apply(color_t_f, subset=[col_name])
        elif output_type == OutputType.Continuous:
            df = df.style.apply(color_magnitude, subset=[col_name]).set_table_styles(
                [
                    dict(
                        selector="th",
                        props=[("min-width", "100px"), ("word-break", "break-word")],
                    )
                ]
            )
        else:
            raise Exception(f'Output type "{output_type}" is not supported')

        df = df.set_sticky(axis=1)  # sticky header
        return df

    # Prepare concept dataframe with input, concept scores, and ground-truth ratings
    def _get_vis_df(self, dataset_id, show_input_field=True, concept_score_col=None):
        # Fetch predictions
        res = self.run(dataset_id)
        preds = res.preds

        # Join with dataset to form df
        if concept_score_col is None:
            concept_score_col = self.concept_term
        df = self.sb.join_preds(dataset_id, preds, concept_score_col)
        ground_truth_col = self.sb.datasets[dataset_id].ground_truth

        # Filter columns to show: input, concept score, gt ratings
        if show_input_field:
            cols_to_show = [
                "msb_item_id",
                self.input_field,
                concept_score_col,
                ground_truth_col,
            ]
        else:
            cols_to_show = ["msb_item_id", concept_score_col, ground_truth_col]
        df = df[cols_to_show]

        # Sort by concept score
        # TODO: allow change to sorting
        df = df.sort_values(by=concept_score_col, ascending=False)

        return df

    # Returns a visualization of the concept scores in dataframe form
    def visualize(
        self,
        dataset_id,
        is_styled,
    ):
        pass


# Subclass definition of an image concept
# - rescale_max_width: maximum width to use when resizing input image
class ImageConcept(Concept):
    def __init__(
        self,
        *args,
        rescale_max_width: int = 200,
        **kwargs,
    ):
        self.rescale_max_width = rescale_max_width
        super().__init__(*args, **kwargs)

        if (self.input_type != InputType.Image) and (
            self.input_type != InputType.ImageLocal
        ):
            raise Exception(
                f"The provided input field `{self.input_field}` is of type `{self.input_type}`; this concept only accepts image inputs, so please select an image input field."
            )

        if self.output_type is OutputType.Binary:
            raise Exception(
                "The binary output type is not supported for ImageConcept. Please set the `threshold` attribute to apply a threshold on the ImageConcept predictions if you would like binary outputs."
            )

    def get_preds(
        self,
        dataset_id,
    ):
        dataset = self.sb.datasets[dataset_id]

        # Load images
        if (self.input_field) not in dataset.cached_images:
            # Load and cache images
            self.sb._cache_image_field(dataset, self.input_field, self.input_type)
        # Load cached images
        images = dataset.cached_images[self.input_field]

        # Calculate predictions
        preds = {}
        for item_id, img in images.items():
            similarity = image_text_similarity(
                CLIP_MODEL, CLIP_PREPROCESS, img, self.concept_term
            ).item()
            preds[item_id] = similarity

        res = Result(self.id, preds)
        return res

    def visualize(self, dataset_id, is_styled):
        df = self._get_vis_df(dataset_id)

        # Render images
        item_ids = df["msb_item_id"].tolist()
        images_html = self.sb.datasets[dataset_id].cached_images_html[self.input_field]
        df[self.input_field] = [images_html[item_id] for item_id in item_ids]
        df = df.drop(columns=["msb_item_id"])

        if is_styled:
            df = self._style_concept_score_col(df, self.concept_term, self.output_type)
        return df


# Subclass definition of a GPT text concept
# - item_name: item name (e.g. tweet, comment) to be used in prompt
# - truncate_items: whether or not to truncate items that are too long
# - chunk_size: how many items to include in each prompt
class GPTTextConcept(Concept):
    def __init__(
        self,
        *args,
        item_name: str = "item",
        truncate_items: bool = True,
        chunk_size: int = 10,
        **kwargs,
    ):
        self.item_name = item_name
        self.truncate_items = truncate_items
        self.chunk_size = chunk_size
        super().__init__(*args, **kwargs)

        if self.input_type != InputType.Text:
            raise Exception(
                f"The provided input field `{self.input_field}` is of type `{self.input_type}`; this concept only accepts text inputs, so please select an text input field."
            )

    def get_preds(
        self,
        dataset_id,
    ):
        dataset = self.sb.datasets[dataset_id]
        if self.output_type == OutputType.Binary:
            preds = get_text_labels_binary(dataset.df, self)
        elif self.output_type == OutputType.Continuous:
            preds = get_text_labels_contin(dataset.df, self)
        else:
            raise Exception(
                f"Output type '{self.output_type}' has not been implemented yet"
            )

        res = Result(self.id, preds)
        return res

    def visualize(self, dataset_id, is_styled):
        df = self._get_vis_df(dataset_id)
        df = df.drop(columns=["msb_item_id"])

        if is_styled:
            df = self._style_concept_score_col(df, self.concept_term, self.output_type)
        return df


# Subclass definition of a logical concept
# - subconcept_ids: list of concept IDs upon which to apply logical operator
# - operator: logical operator to apply
# - subconcepts: list of the concepts upon which to apply logical operator
class LogicalConcept(Concept):
    def __init__(
        self,
        *args,
        subconcept_ids: list,
        operator: str,
        **kwargs,
    ):
        self.subconcept_ids = subconcept_ids
        self.operator = operator

        super().__init__(*args, **kwargs)
        self.subconcepts = [
            self.sb.concepts[c_id] for c_id in self.subconcept_ids
        ]  # Fetch underlying concepts with specified IDs

        # Modify concept term to be of the format "subconcept1 AND subconcept2 AND subconcept3" for display in dataframes and widgets
        self.concept_term = (f" {self.operator} ").join(self.subconcept_ids)

    def get_subconcept_preds(self, dataset_id):
        dataset = self.sb.datasets[dataset_id]
        subconcept_res = [c.run(dataset_id) for c in self.subconcepts]
        subconcept_preds = [res.preds for res in subconcept_res]

        # key = item_id, value = list of all predictions for that item for the different subconcepts
        subconcept_pred_all = {item_id: [] for item_id in dataset.get_item_ids()}
        for subconcept_pred_dict in subconcept_preds:
            for item_id, pred in subconcept_pred_dict.items():
                subconcept_pred_all[item_id].append(pred)

        return subconcept_preds, subconcept_pred_all

    def get_preds(
        self,
        dataset_id,
    ):
        # Get subconcept predictions
        _, preds_all = self.get_subconcept_preds(dataset_id)

        # Aggregate subconcept predictions based on the operator
        if self.operator == "AND":
            preds_combo = {
                item_id: np.all(pred_arr) for item_id, pred_arr in preds_all.items()
            }
        elif self.operator == "OR":
            preds_combo = {
                item_id: np.any(pred_arr) for item_id, pred_arr in preds_all.items()
            }

        res = Result(self.id, preds_combo)
        return res

    def visualize(self, dataset_id, is_styled):
        df = self._get_vis_df(dataset_id, show_input_field=False)

        # Prepare combined df with both preds and logical pred
        subconcept_preds, _ = self.get_subconcept_preds(dataset_id)
        for subconcept_id, cur_preds in zip(self.subconcept_ids, subconcept_preds):
            cur_df = self.sb.join_preds(dataset_id, cur_preds, subconcept_id)
            cur_df = cur_df[["msb_item_id", subconcept_id]]
            df = df.merge(cur_df, on="msb_item_id")

        df = df.drop(columns=["msb_item_id"])

        if is_styled:
            df = self._style_concept_score_col(df, self.concept_term, self.output_type)
        return df


# Subclass definition of a keyword concept
# - keywords: list of keywords to use in concept's keyword search
# - case_sensitive: whether to treat the keywords as case-sensitive or not
class KeywordConcept(Concept):
    def __init__(
        self,
        *args,
        keywords: list,
        case_sensitive: bool,
        **kwargs,
    ):
        self.keywords = keywords
        self.case_sensitive = case_sensitive

        super().__init__(*args, **kwargs)

        # Only allow on text fields
        if self.input_type != InputType.Text:
            raise Exception(
                f"The KeywordConcept can only be applied to Text input fields. The selected input field '{self.input_field}' is of type '{self.input_type}'."
            )

        self.concept_term = f"{self.id} keywords"

    def _check_keywords(self, cur_str):
        cur_str = str(cur_str)
        for keyword in self.keywords:
            if self.case_sensitive and (keyword in cur_str):
                return True
            elif not self.case_sensitive and (keyword.lower() in cur_str.lower()):
                return True
        return False

    def get_preds(
        self,
        dataset_id,
    ):
        dataset = self.sb.datasets[dataset_id]
        df = dataset.df
        items = df[self.input_field].tolist()
        item_ids = df["msb_item_id"].tolist()

        # Check input field against all keywords
        preds = {}
        for idx, cur_item in enumerate(items):
            item_id = item_ids[idx]
            pred = self._check_keywords(cur_item)
            preds[item_id] = pred

        res = Result(self.id, preds)
        return res

    def visualize(self, dataset_id, is_styled):
        df = self._get_vis_df(dataset_id)
        df = df.drop(columns=["msb_item_id"])

        if is_styled:
            df = self._style_concept_score_col(df, self.concept_term, self.output_type)
        return df


# ---------------------
# IMAGE CONCEPT HELPERS


# Gets the similarity between the provided image and the provided text
# - clip_model: The CLIP model to use
# - preprocess: The CLIP image pre-processing function to use
# - image: The image to compare against the concept term (loaded in memory)
# - text: The concept term against which the image will be compared
def image_text_similarity(
    clip_model,
    preprocess,
    image,
    text,
):
    img = preprocess(image).unsqueeze(0)
    text = open_clip.tokenize([text])

    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

    return similarity


# --------------------
# TEXT CONCEPT HELPERS
# Binary (GPT): Returns binary labels for whether all text examples either align with the concept term.
# 1 = aligns with concept term, 0 = does not align with concept term. Under the hood, the two classes
# specified to GPT-3 in the prompt are '<concept term>' and 'not <concept term>'
# - df: The dataframe containing text examples
# - concept: The concept (with the relevant input_field and concept_term) against which to evaluate the text examples
def get_text_labels_binary(
    df,
    concept: Concept,
):
    concept_term = concept.concept_term

    predictions = create_and_run_prompt(
        df=df,
        concept_term=concept_term,
        item_type=concept.item_name,
        item_type_col=concept.input_field,
        model="text-davinci-002",
        chunk_size=concept.chunk_size,
        truncate_item=concept.truncate_items,
    )

    return predictions


def get_prompt(
    x_arr, concept_term, max_example_length, item_type="text", truncate_item=True
):
    anchor1 = concept_term
    anchor2 = f"not {concept_term}"

    if len(x_arr) > 1:
        # Batch
        intro = f"Decide whether these {item_type}s are '{anchor1}' or '{anchor2}'.\n\n"
        outro = f"\n\n{item_type} results:"
        x_arr_str = [f"{j + 1}: '{comment.strip()}'" for j, comment in enumerate(x_arr)]
    else:
        # Individual example
        x = x_arr[0]
        intro = f"Decide whether this {item_type} is '{anchor1}' or '{anchor2}'.\n\n"
        outro = f"\n\n{item_type} result:"
        x_arr_str = [f"{item_type}: {x}"]

    if truncate_item:
        x_arr_str = [x[:max_example_length] for x in x_arr_str]
    else:
        raise Exception(
            "Items have too many tokens. Please decrease chunk_size, set truncate_item to true or edit item texts in your own way"
        )

    examples = "\n".join(x_arr_str)
    cur_prompt = intro + examples + outro

    return cur_prompt


def calc_max_example_length(max_tokens, chunk_size):
    max_total_example_tokens = (
        2048 - max_tokens - 10
    )  # max model tokens - max response tokens - tokens in intro & outro
    max_total_example_length = (
        3 * max_total_example_tokens
    )  # assume each token is 3 characters long on average
    max_example_length = int(max_total_example_length // chunk_size)
    return max_example_length


# Function to create a GPT-3 prompt, drive calls to GPT-3 or mock function and return predictions
# - df: The dataframe containing the text examples
# - anchor1: First anchor word to use in the prompt, antithetical to anchor2
# - anchor2: Second anchor word to use in the prompt, antithetical to anchor1
# - item_type: The type of content being classified (ex: tweet, comment)
# - item_type_col: The name of the column containing the content being classified
# - model: The GPT-3 model type to use
# - max_tokens: The maximum number of tokens for examples
# - chunk_size: The number of examples to include in each call to GPT-3
# - truncate_item: Whether or not to truncate examples
def create_and_run_prompt(
    df,
    concept_term,
    item_type,
    item_type_col,
    model="text-ada-001",
    max_tokens=200,
    chunk_size=20.0,
    truncate_item=True,
):
    # Models: text-ada-001, text-babbage-001, text-curie-001, text-davinci-002
    chunk_sizes_to_try = []
    min_chunk_size = 1.0
    chunk_sizes_to_try = generate_chunk_sizes(
        chunk_sizes_to_try, min_chunk_size, chunk_size
    )

    comments = df[item_type_col].tolist()
    item_ids = df["msb_item_id"].tolist()

    for chunk_size_to_try in chunk_sizes_to_try:
        chunk_size = chunk_size_to_try
        n_chunks = math.ceil(len(df) / chunk_size)
        max_example_length = calc_max_example_length(max_tokens, chunk_size)
        predictions = {}
        for i in range(n_chunks):
            start_i = int(i * chunk_size)
            end_i = min(int(start_i + chunk_size), len(comments))

            cur_comments = comments[start_i:end_i]
            cur_comments = [
                emoji_pattern.sub(r"", str(c)) for c in cur_comments
            ]  # remove emojis
            cur_item_ids = item_ids[start_i:end_i]

            cur_prompt = get_prompt(
                cur_comments, concept_term, max_example_length, item_type, truncate_item
            )

            try:
                cur_pred, cur_results_isolated = get_gpt_response(
                    model, cur_prompt, max_tokens, concept_term
                )
            except:
                raise Exception(f"Failed to get GPT response.")
            if len(cur_comments) != len(cur_pred):
                break

            for i, cur_pred in enumerate(cur_pred):
                predictions[cur_item_ids[i]] = cur_pred

        if len(df) == len(predictions):
            break
        elif chunk_size_to_try == min_chunk_size:
            # case where using a chunk size of 1.0 doesn't work
            raise Exception(
                f"Unable to create concept model for '{concept_term}'. Please try using a different concept term."
            )

    return predictions


# Function to generate the list of potential chunk sizes to try for GPT-3 prompt
# Returns a list of the chunk sizes to try (floats)
def generate_chunk_sizes(
    chunk_sizes_to_try,
    min_chunk_size,
    chunk_size,
):
    curr_size = chunk_size
    while curr_size > 0:
        chunk_sizes_to_try.append(curr_size)
        curr_size -= 5

    if min_chunk_size not in chunk_sizes_to_try:
        chunk_sizes_to_try.append(min_chunk_size)

    return chunk_sizes_to_try


# Pattern to handle removing emojis before GPT-3 processing
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # chinese char
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    re.UNICODE,
)

def parse_gpt_response(results, concept_term, debug=False):
    results_arr = results.strip().split("\n")
    
    if ":" in results_arr[0]:
        cur_results_isolated = [res.split(":")[1] for res in results_arr]
    elif "-" in results_arr[0]:
        cur_results_isolated = [res.split("-")[1] for res in results_arr]
    elif "1." in results_arr[0]:
        cur_results_isolated = [res.split(".")[1] for res in results_arr]
    else:
        raise Exception("Unexpected output format from GPT")

    cur_results_isolated = [res.lower().strip() for res in cur_results_isolated]
    cur_pred = [(res == concept_term) for res in cur_results_isolated]

    if debug:
        print(
            f"cur_pred = {cur_pred}\ncur_results_isolated = {cur_results_isolated}\n\n"
        )

    return cur_pred, cur_results_isolated

# Function to make a GPT-3 call and sanitize the output
# - model: The GPT-3 model type to use
# - cur_prompt: The current prompt for GPT-3
# - max_tokens: The maximum number of tokens for examples
# - anchor1: First anchor word used in the prompt
def get_gpt_response(
    model,
    cur_prompt,
    max_tokens,
    concept_term,
    is_batched=True,
    debug=False,
):
    response = openai.Completion.create(
        model=model,
        prompt=cur_prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )

    results = response.choices[0].text

    # Special handling for non-batched case
    if not is_batched:
        res = results.lower().strip()
        cur_pred = res == concept_term
        return [cur_pred], [res]

    # Handle default batched case
    if debug:
        print(cur_prompt)
        print(results)
    return parse_gpt_response(results, concept_term, debug)


# Continuous (CLIP): Returns continuous 0-1 scores for the extent to which all text examples align with the concept term.
# 1 = high similarity, 0 = low similarity. Under the hood, we are using CLIP to encode the input text and compare similarity
# with the concept term.
# - df: The dataframe containing the text examples
# - concept: The concept (with the relevant input_field and concept_term) against which to evaluate the text examples
def get_text_labels_contin(
    df,
    concept: Concept,
    aggregation_method="max",
):
    in_col = concept.input_field
    concept_term = concept.concept_term
    predictions = {}

    in_vals = df[in_col].tolist()
    item_ids = df["msb_item_id"].tolist()
    for idx, in_val in enumerate(in_vals):
        in_val_words = str(in_val).split()
        chunk_size = concept.chunk_size
        similarities = [
            text_text_similarity(
                CLIP_MODEL, " ".join(in_val_words[i : i + chunk_size]), concept_term
            )
            for i in range(0, len(in_val_words), chunk_size)
        ]
        if aggregation_method == "mean":
            similarity = np.mean(similarities)
        elif aggregation_method == "max":
            similarity = np.max(similarities)
        else:
            raise Exception(f"Aggregation method {aggregation_method} is not valid")

        predictions[item_ids[idx]] = similarity

    return predictions


# Gets the similarity between the provided text items
# - clip_model: The CLIP model to use
# - text_in: The image to compare against the concept term (loaded in memory)
# - text_ref: The concept term against which the image will be compared
def text_text_similarity(
    clip_model,
    text_in,
    text_ref,
):
    text_in = open_clip.tokenize(text_in)
    text_ref = open_clip.tokenize([text_ref])

    with torch.no_grad():
        text_in_features = clip_model.encode_text(text_in)
        text_ref_features = clip_model.encode_text(text_ref)
        text_in_features /= text_in_features.norm(dim=-1, keepdim=True)
        text_ref_features /= text_ref_features.norm(dim=-1, keepdim=True)

        similarity = text_in_features @ text_ref_features.T
    # print(similarity, similarity[0], similarity[0][0].item())
    # sim_pos = similarity[0][0].item()
    # sim_neg = similarity[0][1].item()
    # return np.max([0., (sim_pos - sim_neg)])

    return similarity[0][0].item()
