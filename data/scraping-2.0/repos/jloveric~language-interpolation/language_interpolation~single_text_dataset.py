import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Callable, Union, Any, Dict
import gutenbergpy.textget
from torch import Tensor
import logging
from multiprocessing import Pool
from functools import partial
import itertools
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter

logger = logging.getLogger(__name__)


def ascii_to_float(ascii_tensor: torch.Tensor):
    return (ascii_tensor - 64 + 0.5) / 64


def float_to_ascii(float_tensor: torch.Tensor):
    return ((float_tensor + 1.0) * 64 - 0.5).int()


def print_lines(text, size=100):
    short = text[:size].replace("\n", " ")
    logger.info(f"{short}")


def create_full_paths(root_dir: str, filenames: List[str] = None):
    """
    Construct global path from a list of local paths and a root directory.
    Args :
        root_dir : The directory containing the files
        filenames : A list of filenames within the root
    """
    if filenames is not None:
        full_paths = [f"{root_dir}/{path}" for path in filenames]
        return full_paths

    return None


def unify_ids(specific_ids: List[int], id_range: List[int]):
    """
    Create a single list from ids specified as a list and those
    specified as a range.
    Args :
        specific_ids : [1,2,10,20,100]
        range : [1,10] all values between 1 and 10 including 1 and 10
    """
    ids = []
    if specific_ids is not None:
        ids = specific_ids

    if id_range is not None:
        expand_ids = list(
            range(id_range[0], id_range[1] + 1)
        )  # User expects inclusive range
        ids.extend(expand_ids)

    return ids


def encode_input_from_text(text_in: str, features: int = 0) -> Tuple[torch.tensor, str]:
    """
    Convert a string to input that the network can take.  Take the last "features" number
    of characters and convert to numbers.  Return those numbers as the network input, also
    return the raw_features (the text used to create the numbers).
    Args :
        text_in : input string.
        features : number of input features.
    Returns :
        tensor encoding, text used to create encoding.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    raw_sample = text[-(features):]
    encoding = [ord(val) for val in raw_sample]
    return torch.tensor(encoding), raw_sample


def decode_output_to_text(
    encoding: torch.tensor, topk: int = 1
) -> Tuple[torch.tensor, str]:
    """
    Takes an output from the network and converts to text.
    Args :
        encoding : Tensor of size 128 for each ascii character
        topk : The number of maximum values to report back
    Returns :
        Tuple of topk values and corresponding topk indices and list containing
        actual ascii values.
    """
    probabilities = torch.nn.Softmax(dim=0)(encoding)

    ascii_codes = torch.topk(probabilities, k=topk, dim=0)
    ascii_values = [
        chr(val).encode("ascii", "ignore").decode("ascii") for val in ascii_codes[1]
    ]

    return ascii_codes[0], ascii_codes[1], ascii_values


def generate_dataset(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset and convert to ordinal values for each character.
    This approach needs to be used for the neural network based approach.

    This is memory inefficient as it accumulates the datasets as a moving
    window with 1 character.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        n_feature = [ord(val) for val in text[i : (i + features)]]
        feature_list.append(n_feature)
        n_target = [ord(val) for val in text[(i + features) : (i + features + targets)]]
        target_list.append(n_target)

    return torch.tensor(feature_list), torch.tensor(target_list)


def dataset_centered(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a centered dataset as integers
    """

    f_left = features // 2
    f_right = features - f_left
    t_left = targets // 2
    t_right = targets - t_left

    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        feature_set = (
            text[i : (i + f_left)] + text[(f_left + 1) : (f_left + 1 + f_right)]
        )

        n_feature = [ord(val) for val in feature_set]
        feature_list.append(n_feature)
        n_target = [
            ord(val) for val in text[(i + f_left - t_left) : (i + f_left + t_right)]
        ]
        target_list.append(n_target)

    return torch.tensor(feature_list), torch.tensor(target_list)


def generate_dataset_char(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset as characters for use in random forest approaches.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        n_feature = [ord(val) for val in text[i : (i + features)]]
        feature_list.append(n_feature)
        n_target = [ord(val) for val in text[(i + features) : (i + features + targets)]]
        target_list.append(n_target)

    return feature_list, target_list


def dataset_centered_char(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a centered dataset as char
    """

    f_left = features // 2
    f_right = features - f_left
    t_left = targets // 2
    t_right = targets - t_left

    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)
    final = len(text) - (targets + features)
    feature_list = []
    target_list = []
    for i in range(final):
        feature_set = (
            text[i : (i + f_left)] + text[(f_left + 1) : (f_left + 1 + f_right)]
        )
        n_feature = [ord(val) for val in feature_set]
        feature_list.append(n_feature)
        n_target = [
            ord(val) for val in text[(i + f_left - t_left) : (i + f_left + t_right)]
        ]
        target_list.append(n_target)

    return feature_list, target_list


def generate_flat_dataset(
    text_in: str, features: int, targets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset based on sentences. Just return the
    text as ascii character codes (0-n). Features and targets
    are ignored.
    """
    text = text_in.encode("ascii", "ignore").decode("ascii")
    print_lines(text)

    text_as_ord = [ord(val) for val in text]

    return torch.tensor(text_as_ord), torch.tensor([])


dataset_registry = {
    "sequence": generate_dataset,
    "centered": dataset_centered,
    "sequence_of_char": generate_dataset_char,
    "centered_char": dataset_centered_char,
    "flat": generate_flat_dataset,
}


def dataset_from_file(
    filename: str,
    features: int,
    targets: int,
    max_size: int = -1,
    dataset_generator=generate_dataset,
):
    with open(filename, "r") as f:
        return dataset_generator(
            text_in=f.read()[0:max_size], features=features, targets=targets
        )


def dataset_from_gutenberg(
    gutenberg_id: int,
    features: int,
    targets: int,
    max_size: int = -1,
    dataset_generator: Callable[[str, int, int], Tuple[Any, Any]] = generate_dataset,
) -> Union[Tuple[Tensor, Tensor], Any]:
    """
    Create a dataset from a book in project gutenberg https://www.gutenberg.org/
    Args :
        gutenberg_id : integer id of the book
        features : number of input features to use (number of characters)
        targets: number of targets to use (number of characters)
        datset_generator: formats the resulting dataset
    """
    raw_book = gutenbergpy.textget.get_text_by_id(gutenberg_id)
    clean_book = gutenbergpy.textget.strip_headers(raw_book)
    clean_book = clean_book.decode()

    return dataset_generator(
        text_in=clean_book[0:max_size], features=features, targets=targets
    )


def dataset_sequential(
    filenames: List[str] = None,
    gutenberg_ids: List[int] = None,
    text: str = None,
    features: int = 10,
    targets: int = 1,
    max_size: int = -1,
    dataset_generator: Callable[[str, int, int], Tuple[Any, Any]] = generate_dataset,
    num_workers: int = 0,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Load the datasets from different sources, but keep the different sources
    as lists so that they can be considered specific sequences.  This is needed
    when deriving new datasets from embeddings, which are meant to remain in sequence.

    Args :
        filenames : List of filenames to load data from
        features : Number of input features (characters)
        targets : Number of output features (characters)
        max_size : Set the maximum number of characters to read from file.  Defaults
        to -1 which is to read everything.
        dataset_generator: A function that converts text into a tuple of features, targets
        num_workers: Number of parallel workers when more than one book is being
        processed.

    """
    if filenames is None and text is None and gutenberg_ids is None:
        raise ValueError(f"Must define either filenames, text or gutenberg ids.")
    if (filenames is not None) and (text is not None):
        raise ValueError(f"Either filenames, text, or gutenberg_ids must be defined.")

    list_features = []
    list_targets = []

    if filenames is not None:
        feature_list, target_list = dataset_from_file(
            filenames[0],
            features=features,
            targets=targets,
            max_size=max_size,
            dataset_generator=dataset_generator,
        )

        list_features.append(feature_list)
        list_targets.append(target_list)

    if text is not None:
        feature_list, target_list = dataset_generator(
            text_in=text, features=features, targets=targets
        )

        list_features.append(feature_list)
        list_targets.append(target_list)

    if gutenberg_ids is not None:
        if num_workers > 0:  # Run in parallel
            print("Downloading in parallel", num_workers)
            pdataset = partial(
                dataset_from_gutenberg,
                features=features,
                targets=targets,
                max_size=max_size,
                dataset_generator=dataset_generator,
            )
            with Pool(num_workers) as p:
                results = p.map(
                    pdataset,
                    gutenberg_ids,
                )

            for feature_res, target_res in results:
                list_features.append(feature_res)
                list_targets.append(target_res)

        else:  # Run in serial
            print("Downloading in serial")
            for index in gutenberg_ids:
                feature_list, target_list = dataset_from_gutenberg(
                    index,
                    features=features,
                    targets=targets,
                    max_size=max_size,
                    dataset_generator=dataset_generator,
                )

                list_features.append(feature_list)
                list_targets.append(target_list)
    print("list_features", list_features, "list_targets", list_targets)
    return list_features, list_targets


class RandomizeCharacters:
    def __init__(
        self,
        features: int,
        symbols: int,
        random_frac: float,
        add_channel_dimension: bool = False,
    ):
        self._features = features
        self._symbols = symbols
        self._random_frac = random_frac
        self._num_rand = int(random_frac * features)
        self._add_channel_dimension = add_channel_dimension

    def __call__(self, sample: Tensor):
        """
        Args :
            sample : tensor with values between 0 and features
        """

        rand_id = torch.randint(low=0, high=self._features, size=(self._num_rand,))
        rand_values = torch.randint(low=0, high=self._symbols, size=(self._num_rand,))

        if self._add_channel_dimension is True:
            sample[:, rand_id] = rand_values
        else:
            sample[rand_id] = rand_values

        return sample


class SingleTextDataset(Dataset):
    def __init__(
        self,
        filenames: List[str] = None,
        gutenberg_ids: List[int] = None,
        text: str = None,
        features: int = 10,
        targets: int = 1,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Any, Any]
        ] = generate_dataset,
        num_workers: int = 0,
        add_channel_dimension: bool = False,
        transforms: Callable[[Tensor], Tensor] = None,
        transformer: bool = False,
        embedding_size: int = None,
        as_index: bool=False
    ):
        """
        Args :
            filenames : List of filenames to load data from
            features : Number of input features (characters)
            targets : Number of output features (characters)
            max_size : Set the maximum number of characters to read from file.  Defaults
            to -1 which is to read everything.
            dataset_generator: A function that converts text into a tuple of features, targets
            num_workers: Number of parallel workers when more than one book is being
            processed.
            add_channel_dimension: For convnets we need to add a channel dimension to the data
            transformer: Whether it should be formatted for a (high order) transformer or not
            embedding_size: Size of the embedding if a transformer is being used.
            as_index: Inputs should be indexes instead of floats
        """

        list_features, list_targets = dataset_sequential(
            filenames=filenames,
            gutenberg_ids=gutenberg_ids,
            text=text,
            features=features,
            targets=targets,
            max_size=max_size,
            dataset_generator=dataset_generator,
            num_workers=num_workers,
        )

        list_features = list(itertools.chain(*list_features))
        list_targets = list(itertools.chain(*list_targets))

        self.inputs = torch.stack(list_features)
        self.output = torch.stack(list_targets)
        if add_channel_dimension is True:
            self.inputs = self.inputs.unsqueeze(1)

        self.targets = targets
        self.transforms = transforms
        self.valid_ids = list(range(0, len(list_features)))
        self._transformer = transformer
        self._embedding_size = embedding_size
        self._as_index = as_index

    def __len__(self):
        return len(self.valid_ids)

    def normalize(self, data):
        return (data - 64 + 0.5) / 64.0

    def flat(self, idx) -> Tensor:
        """
        Flat text sequence
        """
        index = self.valid_ids[idx]
        if torch.is_tensor(index):
            index = index.tolist()

        inputs = self.inputs[index].clone()
        if self.transforms is not None:
            inputs = self.transforms(inputs)

        return self.normalize(inputs), self.output[index], idx

    def group(self, idx) -> Tensor:
        """
        Group the characters into equal
        sized embeddings.  Since I'm using high order layers these
        aren't actual embeddings, they are just groups of n
        characters where the number of characters is "embedding_size"
        """
        index = self.valid_ids[idx]
        if torch.is_tensor(index):
            index = index.tolist()

        inputs = self.inputs[index].clone()
        if self.transforms is not None:
            inputs = self.transforms(inputs)

        if self._as_index is False:
            return (
                self.normalize(inputs).reshape(inputs.shape[0], -1, self._embedding_size),
                self.output[index].reshape(self.output.shape[0], -1, self._embedding_size),
                idx,
            )
        else :
            return (
                inputs.reshape(inputs.shape[0], -1, self._embedding_size),
                self.output[index].reshape(self.output.shape[0], -1, self._embedding_size),
                idx,
            )

    def __getitem__(self, idx) -> Tensor:
        if self._transformer is True:
            return self.group(idx)
        else:
            return self.flat(idx)

    def __call__(self, idx) -> Tensor:
        return self.__getitem__(idx)


class TextTransformerDataset(Dataset):
    """
    This dataset is different than the typical transformer dataset. Instead
    of using a mask, I select random locations an make sure each sample
    is the same size for a batch. Samples are randomized within a batch, but
    sample width is only randomized per batch.
    """

    def __init__(
        self,
        filenames: List[str] = None,
        gutenberg_ids: List[int] = None,
        text: str = None,
        characters_per_feature: int = 10,
        max_features=100,
        targets: int = 1,
        max_size: int = -1,
        dataset_generator: Callable[
            [str, int, int], Tuple[Any, Any]
        ] = generate_flat_dataset,
        num_workers: int = 0,
        add_channel_dimension: bool = False,
        transforms: Callable[[Tensor], Tensor] = None,
        embedding_size: int = None,
        repeats: int = 1,
        as_index: bool=True,
    ):
        """
        Args :
            filenames : List of filenames to load data from
            characters_per_feature : Number of characters that define a feature (characters - this is basically the token size)
            max_features : maximum number of features to use (4k for a 4k context windows)
            targets : Number of output features (characters)
            max_size : Set the maximum number of characters to read from file.  Defaults
            to -1 which is to read everything.
            dataset_generator: A function that converts text into a tuple of features, targets
            num_workers: Number of parallel workers when more than one book is being
            processed.
            add_channel_dimension: For convnets we need to add a channel dimension to the data
            embedding_size: Size of the embedding if a transformer is being used.
            as_index: inputs should be indexes not floats
        """

        list_features, list_targets = dataset_sequential(
            filenames=filenames,
            gutenberg_ids=gutenberg_ids,
            text=text,
            features=None,
            targets=None,
            max_size=max_size,
            dataset_generator=dataset_generator,
            num_workers=num_workers,
        )

        list_features = list(itertools.chain(*list_features))
        # list_targets = list(itertools.chain(*list_targets))

        self.inputs = torch.stack(list_features)

        # self.output = torch.stack(list_targets)
        if add_channel_dimension is True:
            self.inputs = self.inputs.unsqueeze(1)

        self.targets = targets
        self.transforms = transforms
        self.valid_ids = list(range(0, len(list_features)))

        # print('valid_ids', self.valid_ids)
        print("inputs.shape", self.inputs.shape)
        self._embedding_size = embedding_size
        self._characters_per_feature = characters_per_feature
        self._max_features = max_features
        self._max_characters = self._characters_per_feature * max_features

        self.data_size = len(self.inputs) - self._max_characters
        self._repeats = repeats
        self._as_index = as_index

    def __len__(self):
        return int((len(self.inputs) - self._max_characters) * self._repeats)

    def index_converter():
        pass

    def normalize(self, data):
        return (data - 64 + 0.5) / 64.0

    def group(self, idx) -> Tensor:
        """
        Group the characters into equal
        sized embeddings.  Since I'm using high order layers these
        aren't actual embeddings, they are just groups of n
        characters where the number of characters is "embedding_size"
        """

        # Now that's lazy
        idx = idx % self.data_size

        index = self.valid_ids[idx]
        if torch.is_tensor(index):
            index = index.tolist()

        max_size = min(torch.numel(self.inputs) - index, self._max_characters)

        # TODO: May not need this clone
        inputs = self.inputs[index : (index + max_size)].clone()
        if self.transforms is not None:
            inputs = self.transforms(inputs)

        return (
            inputs.reshape(self._max_features, self._characters_per_feature),
            idx,
        )

    def __getitem__(self, idx) -> Tensor:
        return self.group(idx)

    def __call__(self, idx) -> Tensor:
        return self.__getitem__(idx)
