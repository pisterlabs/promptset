import sys
from langchain_utils.utils import convert_str_slice_notation_to_slice, save_stdin_to_tempfile


def test_convert_str_slice_notation_to_slice():
    assert convert_str_slice_notation_to_slice('1:3') == slice(1, 3)
    assert convert_str_slice_notation_to_slice('1:') == slice(1, None)
    assert convert_str_slice_notation_to_slice(':3') == slice(None, 3)
    assert convert_str_slice_notation_to_slice(':') == slice(None, None)
    assert convert_str_slice_notation_to_slice('3') == slice(3)
    assert convert_str_slice_notation_to_slice('1:8:2') == slice(1, 8, 2)


def test_save_stdin_to_tempfile() -> None:
    """Test saving stdin to a temporary file."""
    import io
    import os

    # Set up stdin with test input
    test_input = "Hello World!"
    stdin = io.StringIO(test_input)
    stdin_backup = sys.stdin
    sys.stdin = stdin

    # Call function to save stdin to tempfile
    temp_file_path = save_stdin_to_tempfile()

    # Check that tempfile was created
    assert os.path.exists(temp_file_path)

    # Check contents of tempfile
    with open(temp_file_path, 'r') as f:
        assert f.read() == test_input

    # Restore stdin
    sys.stdin = stdin_backup
