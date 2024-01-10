from dino_utils import (
    count_num_features_for_dino,
    get_model_for_dino,
    get_model_resolution_for_dino,
    get_model_slug_for_dino,
    get_preprocessing_for_dino,
    label_image_for_dino,
)
from keras_utils import (
    count_num_features_for_keras,
    get_dummy_preprocessing_for_keras,
    get_model_for_keras,
    get_model_resolution_for_keras,
    get_model_slug_for_keras,
    label_image_for_keras,
)
from openai_utils import (
    count_num_features_for_clip,
    get_model_for_clip,
    get_model_resolution_for_clip,
    get_model_slug_for_clip,
    get_preprocessing_for_clip,
    label_image_for_clip,
)


def get_my_model_of_choice(choice_index=0):
    available_models = [
        get_model_slug_for_keras(),
        get_model_slug_for_clip(),
        get_model_slug_for_dino(),
    ]

    # The following line is where you can switch between Keras' MobileNet and OpenAI's CLIP:
    chosen_model = available_models[choice_index]

    return chosen_model


def get_num_features(model=None, args=None):
    if get_model_slug_for_clip() == get_my_model_of_choice():
        num_features = count_num_features_for_clip(model)
    elif get_model_slug_for_dino() == get_my_model_of_choice():
        num_features = count_num_features_for_dino(model, args)
    else:
        num_features = count_num_features_for_keras(model)

    return num_features


def get_preprocessing_tool():
    if get_model_slug_for_clip() == get_my_model_of_choice():
        preprocessing_tool = get_preprocessing_for_clip()
    elif get_model_slug_for_dino() == get_my_model_of_choice():
        preprocessing_tool = get_preprocessing_for_dino()
    else:
        preprocessing_tool = get_dummy_preprocessing_for_keras()

    return preprocessing_tool


def get_target_model_size(resolution=None):
    if resolution is None:
        if get_model_slug_for_clip() == get_my_model_of_choice():
            resolution = get_model_resolution_for_clip()
        elif get_model_slug_for_dino() == get_my_model_of_choice():
            resolution = get_model_resolution_for_dino()
        else:
            resolution = get_model_resolution_for_keras()

    target_model_size = (resolution, resolution)

    return target_model_size


def get_input_shape(target_model_size, num_channels=3):
    # Image data format: channels last
    input_shape = (*list(target_model_size), num_channels)

    return input_shape


def load_model(target_model_size=None, include_top=False, pooling="avg", args=None):
    if target_model_size is None:
        target_model_size = get_target_model_size()

    input_shape = get_input_shape(target_model_size)
    if get_model_slug_for_clip() == get_my_model_of_choice():
        model = get_model_for_clip(
            input_shape=input_shape,
            include_top=include_top,
            pooling=pooling,
        )
    elif get_model_slug_for_dino() == get_my_model_of_choice():
        model = get_model_for_dino(args)
    else:
        model = get_model_for_keras(
            input_shape=input_shape,
            include_top=include_top,
            pooling=pooling,
        )

    return model


def convert_image_to_features(image, model, preprocess=None):
    if get_model_slug_for_clip() == get_my_model_of_choice():
        yhat = label_image_for_clip(image, model, preprocess=preprocess)
    elif get_model_slug_for_dino() == get_my_model_of_choice():
        yhat = label_image_for_dino(image, model, preprocess=preprocess)
    else:
        yhat = label_image_for_keras(image, model, preprocess=preprocess)

    features = yhat.flatten()

    return features


if __name__ == "__main__":
    chosen_model = get_my_model_of_choice()
    print(f"Slug of the chosen model: {chosen_model}")
