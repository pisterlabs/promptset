
import pytest
import os
from mock import patch
from summary_for_test import summarize
from faker import Faker
import openai
import youtube_transcript_api


current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "summary.md")


@patch('summary_for_test.YouTubeTranscriptApi.get_transcript', return_value=[{'text': 'This is a test transcript.'}])
@patch('openai.Completion.create', return_value={'choices': [{'text': 'This is a summary.'}]})
@patch('summary_for_test.YouTube')
def test_summarize(mock_youtube, mock_openai_create, mock_yt_transcripts):
    if os.path.exists(filename):
        os.remove(filename)
        # print(f"{filename} deleted!")
    # time.sleep(5)

    mock_youtube_instance = mock_youtube.return_value
    mock_youtube_instance.title = 'Test Title'
    mock_youtube_instance.author = 'Test Author'
    url = 'https://www.youtube.com/watch?v=6GQRnPlephU'

    summarize(url)
    mock_yt_transcripts.assert_called_once()
    mock_openai_create.assert_called()
    mock_youtube.assert_called_once_with(url)
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcript of' in summary
        assert 'Executive Summary' in summary
        assert 'Main Takeaways' in summary
    os.remove(filename)


@pytest.fixture
def mock_openai():
    with patch('openai.Completion.create', return_value={'choices': [{'text': 'This is a summary.'}]}) as mock_openai_create:
        yield mock_openai_create


def test_summarize_openai_with_regular_url(mock_openai):
    url = 'https://www.youtube.com/watch?v=e9KJ3kd80fQ'

    summarize(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcript of' in summary
        assert 'Executive Summary' in summary
        assert 'Main Takeaways' in summary
    os.remove(filename)


def test_summarize_openai_with_ampersand_url(mock_openai):
    url = 'https://www.youtube.com/watch?v=e9KJ3kd80fQ&t=18s'
    summarize(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcript of' in summary
        assert 'Executive Summary' in summary
        assert 'Main Takeaways' in summary
    os.remove(filename)


def test_summarize_openai_with_youtube_shortened_url(mock_openai):
    url = 'https://youtu.be/e9KJ3kd80fQ'
    summarize(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcript of' in summary
        assert 'Executive Summary' in summary
        assert 'Main Takeaways' in summary
    os.remove(filename)


def test_summarize_exception_wrong_url():
    fake = Faker()
    url = f'https//youtu.be/{fake.word()}'
    result = summarize(url)
    assert result == "Error: check url"


def test_summarize_exception_no_caption():
    url = 'https://youtu.be/e9KJ3kd80fQ'
    language = 'fr'
    result = summarize(url, language)
    assert result == "Error: no transcript on selected language. Check language"


def test_summarize_exception_no_env(monkeypatch):
    url = 'https://youtu.be/e9KJ3kd80fQ'
    fake = Faker()
    new_env = fake.word()
    monkeypatch.setenv("OPENAI_API", new_env)
    result = summarize(url)
    assert result == "Error: check OpenAI API key"


def test_summarize_exception_rate_limit():
    url = 'https://youtu.be/e9KJ3kd80fQ'
    with pytest.raises(openai.error.RateLimitError, match="Error: You have exceeded OpenAI requests. Try again later"):
        summarize(url)
        raise openai.error.RateLimitError(
            "Error: You have exceeded OpenAI requests. Try again later")


def test_summarize_exception_service_unavailable():
    url = 'https://youtu.be/e9KJ3kd80fQ'
    with pytest.raises(openai.error.ServiceUnavailableError, match="Error: OpenAI servers are saturated, try again later"):
        summarize(url)
        raise openai.error.ServiceUnavailableError(
            "Error: OpenAI servers are saturated, try again later")


def test_summarize_exception_transcripts_disable():
    url = 'https://www.youtube.com/watch?v=u9NESbmj4bU'
    with pytest.raises(youtube_transcript_api._errors.TranscriptsDisabled, match="Error: transcripts are disabled in this video. You cannot summarize it"):
        summarize(url)
        raise youtube_transcript_api._errors.TranscriptsDisabled(
            "Error: transcripts are disabled in this video. You cannot summarize it")
