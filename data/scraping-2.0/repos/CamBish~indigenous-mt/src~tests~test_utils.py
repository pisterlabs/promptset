# FILEPATH: /home/cambish/code-repos/indigenous-mt/tests/test_utils.py
import os

import openai
import pandas as pd
import pytest
from pyarrow import parquet as pq

from src import utils


def test_check_environment_variables(monkeypatch):
    # Test that no exception is raised when all environment variables are set
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("MODEL", "model")
    monkeypatch.setenv("SOURCE_LANGUAGE", "source")
    monkeypatch.setenv("TARGET_LANGUAGE", "target")
    monkeypatch.setenv("TEXT_DOMAIN", "domain")
    try:
        utils.check_environment_variables()
    except KeyError:
        pytest.fail("check_environment_variables() raised KeyError unexpectedly!")


def test_check_environment_variables_missing(monkeypatch):
    # Test that a KeyError is raised when environment variables are missing
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("SOURCE_LANGUAGE", raising=False)
    monkeypatch.delenv("TARGET_LANGUAGE", raising=False)
    monkeypatch.delenv("TEXT_DOMAIN", raising=False)
    with pytest.raises(KeyError):
        utils.check_environment_variables()


def test_get_filenames(mocker):
    # Mock the os.walk function to return a specific set of files
    mocker.patch(
        "os.walk",
        return_value=[("/path/to/dir", [], ["file1.txt", "file2.txt", "file3.txt"])],
    )
    filenames = utils.get_filenames("/path/to/dir")
    assert filenames == ["file1.txt", "file2.txt", "file3.txt"]


def test_has_parallel_cree_english_data():
    # Test that the function correctly identifies when parallel data exists
    filenames = ["file1_cr.txt", "file1_en.txt", "file2_cr.txt"]
    assert utils.has_parallel_cree_english_data("file1_cr.txt", filenames)
    assert not utils.has_parallel_cree_english_data("file2_cr.txt", filenames)


def test_get_cree_and_english_data_paths():
    # Test that the function correctly generates the paths to the Cree and English data files
    root = "/path/to/dir"
    filename = "file1_cr.txt"
    source_path, target_path = utils.get_cree_and_english_data_paths(root, filename)
    assert source_path == "/path/to/dir/file1_cr.txt"
    assert target_path == "/path/to/dir/file1_en.txt"


def test_read_lines_from_file(mocker):
    # Test that the function correctly reads lines from a file
    mocker.patch("builtins.open", mocker.mock_open(read_data="line1\nline2\nline3\n"))
    lines = utils.read_lines_from_file("/path/to/file.txt")
    assert lines == ["line1", "line2", "line3"]


def test_load_cree_parallel_data(mocker):
    # Mock the os.walk function to return a specific set of files
    mocker.patch(
        "os.walk",
        return_value=[
            ("/path/to/dir", [], ["file1_cr.txt", "file1_en.txt", "file2_cr.txt"])
        ],
    )
    df = utils.load_cree_parallel_data("/path/to/dir")
    expected_df = pd.DataFrame(
        {
            "source_text": ["file1_cr.txt"],
            "target_text": ["file1_en.txt"],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_serialize_gold_standards(mocker):
    # Mock the os.walk function to return a specific set of files
    mocker.patch(
        "os.walk",
        return_value=[("/path/to/dir", [], ["file1_gs.txt", "file2_gs.txt"])],
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data="line1\nline2\nline3\n"))
    mocker.patch("os.makedirs")
    utils.serialize_gold_standards("/path/to/dir", "/path/to/output.txt")


def test_load_gold_standards(mocker):
    # Mock the os.walk function to return a specific set of files
    mocker.patch(
        "os.walk",
        return_value=[("/path/to/dir", [], ["file1_gs.txt", "file2_gs.txt"])],
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data="line1\nline2\nline3\n"))
    gs_df = utils.load_gold_standards("/path/to/dir")
    expected_df = pd.DataFrame(
        {
            "gold_standard_text": ["line1", "line2", "line3"],
        }
    )
    pd.testing.assert_frame_equal(gs_df, expected_df)


def test_extract_and_align_gold_standard(mocker):
    # Mock the os.walk function to return a specific set of files
    mocker.patch(
        "os.walk",
        return_value=[("/path/to/dir", [], ["file1_gs.txt", "file2_gs.txt"])],
    )
    mocker.patch("builtins.open", mocker.mock_open(read_data="line1\nline2\nline3\n"))
    gs_text = utils.extract_and_align_gold_standard("file1")
    assert gs_text == "line1\nline2\nline3"


def test_link_gold_standard(mocker):
    mocker.patch("os.makedirs")
    utils.link_gold_standard(
        ["link1", "link2"], "/path/to/inuktitut", "/path/to/english"
    )
    assert os.makedirs.called_with("/path/to/inuktitut")
    assert os.makedirs.called_with("/path/to/english")
    assert os.makedirs.call_count == 2


def test_load_parallel_corpus(mocker):
    mocker.patch("pq.read_table")
    utils.load_parallel_corpus("/path/to/corpus.parquet")
    assert pq.read_table.called_with("/path/to/corpus.parquet")
    assert pq.read_table.call_count == 1


def test_eval_results():
    df = pd.DataFrame(
        {
            "source_text": ["line1", "line2", "line3"],
            "target_text": ["line1", "line2", "line3"],
            "predicted_text": ["line1", "line2", "line3"],
            "translated_text": ["line1", "line2", "line3"],
        }
    )
    utils.eval_results(df)


def test_chat_completion_request_api(mocker):
    mocker.patch("openai.Completion.create")
    utils.chat_completion_request_api(["message1", "message2"])
    assert openai.Completion.create.called_with(
        messages=[
            {"role": "system", "content": "message1"},
            {"role": "system", "content": "message2"},
        ]
    )
    assert openai.Completion.create.call_count == 1


def test_n_shot_examples():
    gold_std = pd.DataFrame(
        {
            "source_text": ["line1", "line2", "line3"],
            "target_text": ["line1", "line2", "line3"],
        }
    )
    expected_output = "Text: line1 | Translation: line1 ###\nText: line2 | Translation: line2 ###\nText: line3 | Translation: line3 ###\n"
    output = utils.n_shot_examples(gold_std, 3)
    assert output == expected_output


def test_n_shot_prompting():
    sys_msg = "System message"
    gold_std = pd.DataFrame(
        {
            "source_text": ["line1", "line2", "line3"],
            "target_text": ["line1", "line2", "line3"],
        }
    )
    pll_corpus = pd.DataFrame(
        {
            "source_text": ["line1", "line2", "line3"],
            "target_text": ["line1", "line2", "line3"],
        }
    )
    utils.n_shot_prompting(sys_msg, gold_std, pll_corpus, 3, 3)
