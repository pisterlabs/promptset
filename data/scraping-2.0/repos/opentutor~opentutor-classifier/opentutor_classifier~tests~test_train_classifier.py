#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from os import path
from typing import List

import pytest
import responses
import os

from opentutor_classifier import (
    ExpectationTrainingResult,
    ARCH_LR2_CLASSIFIER,
    DEFAULT_LESSON_NAME,
    ARCH_COMPOSITE_CLASSIFIER,
    ARCH_OPENAI_CLASSIFIER,
)
from opentutor_classifier.dao import (
    ModelRef,
    find_predicton_config_and_pickle,
)
from opentutor_classifier.config import confidence_threshold_default
from opentutor_classifier.lr2.constants import MODEL_FILE_NAME
from opentutor_classifier.openai.constants import GROUNDTRUTH_FILENAME
from opentutor_classifier.openai.train import OpenAIGroundTruth
from .utils import (
    assert_testset_accuracy,
    assert_train_expectation_results,
    create_and_test_classifier,
    fixture_path,
    read_example_testset,
    test_env_isolated,
    train_classifier,
    train_default_classifier,
    _TestExpectation,
    run_classifier_testset,
)

CONFIDENCE_THRESHOLD_DEFAULT = confidence_threshold_default()


@pytest.fixture(scope="module")
def data_root() -> str:
    return fixture_path("data")


@pytest.fixture(scope="module")
def shared_root(word2vec) -> str:
    return path.dirname(word2vec)


@pytest.mark.parametrize(
    "lesson,arch,expected_length", [("candles", ARCH_OPENAI_CLASSIFIER, 85)]
)
def test_train_openai_ground_truth(
    tmpdir,
    data_root: str,
    shared_root: str,
    lesson: str,
    arch: str,
    expected_length: int,
):
    os.environ["OPENAI_API_KEY"] = "fake"
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson=lesson, arch=arch
    ) as test_config:
        train_classifier(lesson, test_config, False)
        dao = test_config.find_data_dao()
        config_and_model = find_predicton_config_and_pickle(
            ModelRef(arch, lesson, GROUNDTRUTH_FILENAME), dao
        )
        result: OpenAIGroundTruth = OpenAIGroundTruth.from_dict(config_and_model.model)

        assert len(result.training_answers) == expected_length


@pytest.mark.parametrize("lesson", [("question1"), ("question2")])
def test_outputs_models_at_specified_root(
    tmpdir, data_root: str, shared_root: str, lesson: str
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson=lesson
    ) as test_config:
        result = train_classifier(lesson, test_config)
        assert path.exists(path.join(result.models, MODEL_FILE_NAME))
        assert path.exists(path.join(result.models, "config.yaml"))


@pytest.mark.parametrize(
    "arch,expected_model_file_name",
    [
        (ARCH_LR2_CLASSIFIER, MODEL_FILE_NAME),
    ],
)
def test_outputs_models_at_specified_model_root_for_default_model(
    arch: str, expected_model_file_name: str, tmpdir, data_root: str, shared_root: str
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, lesson=DEFAULT_LESSON_NAME
    ) as test_config:
        result = train_default_classifier(test_config)
        assert path.exists(path.join(result.models, expected_model_file_name))


def _test_train_and_predict(
    lesson: str,
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    expected_training_result: List[ExpectationTrainingResult],
    expected_accuracy: float,
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        testset = read_example_testset(
            lesson, confidence_threshold=confidence_threshold
        )
        assert_testset_accuracy(
            arch,
            train_result.models,
            shared_root,
            testset,
            expected_accuracy=expected_accuracy,
        )


@pytest.mark.parametrize(
    "example,arch,confidence_threshold,expected_training_result,expected_accuracy",
    [
        (
            "missing-expectation-training-data",
            ARCH_COMPOSITE_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(expectation_id="1", accuracy=0.6875),
                ExpectationTrainingResult(expectation_id="2", accuracy=0.6875),
            ],
            0.4,
        ),
        (
            "ies-rectangle",
            ARCH_LR2_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(expectation_id="0", accuracy=0.90),
                ExpectationTrainingResult(expectation_id="1", accuracy=0.95),
                ExpectationTrainingResult(expectation_id="2", accuracy=0.95),
            ],
            1,
        ),
        (
            "candles",
            ARCH_LR2_CLASSIFIER,
            CONFIDENCE_THRESHOLD_DEFAULT,
            [
                ExpectationTrainingResult(expectation_id="0", accuracy=0.84),
                ExpectationTrainingResult(expectation_id="1", accuracy=0.81),
                ExpectationTrainingResult(expectation_id="2", accuracy=0.81),
                ExpectationTrainingResult(expectation_id="3", accuracy=0.95),
            ],
            0.85,
        ),
    ],
)
@pytest.mark.slow
def test_train_and_predict_slow(
    example: str,
    arch: str,
    # confidence_threshold for now determines whether an answer
    # is really classified as GOOD/BAD (confidence >= threshold)
    # or whether it is interpretted as NEUTRAL (confidence < threshold)
    confidence_threshold: float,
    expected_training_result: List[ExpectationTrainingResult],
    expected_accuracy: float,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    _test_train_and_predict(
        example,
        arch,
        confidence_threshold,
        expected_training_result,
        expected_accuracy,
        tmpdir,
        data_root,
        shared_root,
    )


@pytest.mark.parametrize(
    "lesson,arch",
    [
        (
            "shapes",
            ARCH_LR2_CLASSIFIER,
        ),
    ],
)
def test_predict_on_model_trained_with_cluster_features_but_cluster_features_later_disabled(
    lesson: str,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch=arch, lesson=lesson
    ) as test_config:
        monkeypatch.setenv("TRAIN_QUALITY_DEFAULT", str(2))
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)

        monkeypatch.setenv("TRAIN_QUALITY_DEFAULT", str(0))
        testset = read_example_testset(lesson)
        run_classifier_testset(arch, train_result.models, shared_root, testset)


@pytest.mark.parametrize(
    "lesson,use_default,arch",
    [
        (
            "shapes",
            False,
            ARCH_LR2_CLASSIFIER,
        ),
        (
            "shapes",
            True,
            ARCH_LR2_CLASSIFIER,
        ),
    ],
)
def test_predict_off_model_trained_with_cluster_features_but_cluster_features_later_enabled(
    lesson: str,
    use_default: bool,
    arch: str,
    tmpdir,
    data_root: str,
    shared_root: str,
    monkeypatch,
):
    with test_env_isolated(
        tmpdir,
        data_root,
        shared_root,
        arch=arch,
        lesson=lesson,
        is_default_model=use_default,
    ) as test_config:
        monkeypatch.setenv("TRAIN_QUALITY_DEFAULT", str(0))
        train_result = (
            train_default_classifier(test_config)
            if use_default
            else train_classifier(lesson, test_config)
        )
        assert path.exists(train_result.models)
        import logging

        logging.warning(f"models={train_result.models}")
        monkeypatch.setenv("TRAIN_QUALITY_DEFAULT", str(2))
        testset = read_example_testset(lesson)
        run_classifier_testset(
            arch,
            path.join(path.dirname(train_result.models), DEFAULT_LESSON_NAME)
            if use_default
            else train_result.models,
            shared_root,
            testset,
        )


def _test_train_and_predict_specific_answers_slow(
    lesson: str,
    arch: str,
    evaluate_input_list: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    with test_env_isolated(
        tmpdir, data_root, shared_root, arch, lesson=lesson
    ) as test_config:
        train_result = train_classifier(lesson, test_config)
        assert path.exists(train_result.models)
        assert_train_expectation_results(
            train_result.expectations, expected_training_result
        )
        for evaluate_input, ans in zip(evaluate_input_list, expected_evaluate_result):
            create_and_test_classifier(
                lesson,
                path.split(path.abspath(train_result.models))[0],
                shared_root,
                evaluate_input,
                [ans],
                arch=arch,
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "lesson,arch,evaluate_input_list,expected_training_result,expected_evaluate_result",
    [
        (
            "ies-rectangle",
            ARCH_LR2_CLASSIFIER,
            [
                # "5",
                # "It is 3 and 7 and 4 and 0",
                # "30 and 74",
                "37 x 40",
                # "thirty seven by forty",
                # "forty by thirty seven",
                # "37 by forty",
                # "thirty-seven by forty",
                # "37.0 by 40.000",
                # "thirty seven by fourty",
            ],
            [ExpectationTrainingResult(expectation_id="2", accuracy=0.89)],
            [
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Bad", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                _TestExpectation(evaluation="Good", score=0.80, expectation="2"),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
                # _TestExpectation(evaluation="Good", score=0.80, expectation=2),
            ],
        ),
    ],
)
def test_train_and_predict_specific_answers_slow(
    lesson: str,
    arch: str,
    evaluate_input_list: List[str],
    expected_training_result: List[ExpectationTrainingResult],
    expected_evaluate_result: List[_TestExpectation],
    tmpdir,
    data_root: str,
    shared_root: str,
):
    _test_train_and_predict_specific_answers_slow(
        lesson,
        arch,
        evaluate_input_list,
        expected_training_result,
        expected_evaluate_result,
        tmpdir,
        data_root,
        shared_root,
    )


@responses.activate
@pytest.mark.parametrize(
    "lesson,arch,evaluate_input_list,expected_evaluate_result",
    [
        (
            # It's important to test what would happen
            # if--in the past--we had trained a model for a lesson
            # but then subsequently lost the actual trained model.
            # This is an important case, because having trained the model
            # might have generated features which would live on in the config/db,
            # and those generated features would cause shape-errors at prediction time
            # when used with the default model
            "ies-mixture-with-trained-features-but-model-is-lost",
            ARCH_LR2_CLASSIFIER,
            ["a"],
            [
                _TestExpectation(evaluation="Bad", score=0.50, expectation="2"),
            ],
        )
    ],
)
def test_default_classifier_train_and_predict(
    lesson: str,
    arch: str,
    evaluate_input_list: List[str],
    expected_evaluate_result: List[_TestExpectation],
    data_root: str,
    shared_root: str,
    tmpdir,
):
    with test_env_isolated(
        tmpdir,
        data_root,
        shared_root,
        arch=arch,
        is_default_model=True,
        lesson=lesson,
    ) as config:
        train_result = train_default_classifier(config=config)
        assert path.exists(train_result.models)
        for evaluate_input, ans in zip(evaluate_input_list, expected_evaluate_result):
            create_and_test_classifier(
                lesson,
                path.split(path.abspath(train_result.models))[0],
                shared_root,
                evaluate_input,
                [ans],
                arch=arch,
            )
