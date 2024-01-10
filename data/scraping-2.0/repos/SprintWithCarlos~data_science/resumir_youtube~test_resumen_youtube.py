
import pytest
import os
from mock import patch
from resumen_para_pruebas import resumir
from faker import Faker
import openai
import youtube_transcript_api


current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "resumen.md")


@patch('resumen_para_pruebas.YouTubeTranscriptApi.get_transcript', return_value=[{'text': 'This is a test transcript.'}])
@patch('openai.Completion.create', return_value={'choices': [{'text': 'This is a summary.'}]})
@patch('resumen_para_pruebas.YouTube')
def test_resumir(mock_youtube, mock_openai_create, mock_yt_transcripts):
    if os.path.exists(filename):
        os.remove(filename)
        # print(f"{filename} deleted!")
    # time.sleep(5)

    mock_youtube_instance = mock_youtube.return_value
    mock_youtube_instance.title = 'Test Title'
    mock_youtube_instance.author = 'Test Author'
    url = 'https://www.youtube.com/watch?v=6GQRnPlephU'

    resumir(url)
    mock_yt_transcripts.assert_called_once()
    mock_openai_create.assert_called()
    mock_youtube.assert_called_once_with(url)
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcripción de' in summary
        assert 'Resumen' in summary
        assert 'Puntos Principales' in summary
    os.remove(filename)


@pytest.fixture
def mock_openai():
    with patch('openai.Completion.create', return_value={'choices': [{'text': 'This is a summary.'}]}) as mock_openai_create:
        yield mock_openai_create


def test_resumir_openai_with_regular_url(mock_openai):
    url = 'https://www.youtube.com/watch?v=6GQRnPlephU'

    resumir(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcripción de' in summary
        assert 'Resumen' in summary
        assert 'Puntos Principales' in summary
    os.remove(filename)


def test_resumir_openai_with_ampersand_url(mock_openai):
    url = 'https://www.youtube.com/watch?v=6GQRnPlephU&t=18s'
    resumir(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcripción de' in summary
        assert 'Resumen' in summary
        assert 'Puntos Principales' in summary
    os.remove(filename)


def test_resumir_openai_with_youtube_shortened_url(mock_openai):
    url = 'https://youtu.be/6GQRnPlephU'
    resumir(url)
    mock_openai.assert_called()
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        summary = f.read()
        assert 'Transcripción de' in summary
        assert 'Resumen' in summary
        assert 'Puntos Principales' in summary
    os.remove(filename)


def test_resumir_exception_wrong_url():
    fake = Faker()
    url = fake.url()
    result = resumir(url)
    assert result == "Ha ocurrido un error: verifica la url introducida"


def test_resumir_exception_no_caption():
    url = 'https://youtu.be/6GQRnPlephU'
    language = 'fr'
    result = resumir(url, language)
    assert result == "Ha ocurrido un error: no hay transcripción en el idioma seleccionado. Verifica idioma"


def test_resumir_exception_no_env(monkeypatch):
    url = 'https://youtu.be/6GQRnPlephU'
    fake = Faker()
    new_env = fake.word()
    monkeypatch.setenv("OPENAI_API", new_env)
    result = resumir(url)
    assert result == "Ha ocurrido un error: verifica la clave de OpenAI"



def test_resumir_exception_rate_limit():
    url = 'https://youtu.be/e9KJ3kd80fQ'
    with pytest.raises(openai.error.RateLimitError, match="Ha ocurrido un error: has superado el límite de peticiones a OpenAI. Intenta más tarde"):
        resumir(url)
        raise openai.error.RateLimitError(
            "Ha ocurrido un error: has superado el límite de peticiones a OpenAI. Intenta más tarde")






def test_resumir_exception_service_unavailable():
    url = 'https://youtu.be/e9KJ3kd80fQ'
    with pytest.raises(openai.error.ServiceUnavailableError, match="Ha ocurrido un error: los servidores de OpenAI están saturados, intenta más tarde"):
        resumir(url)
        raise openai.error.ServiceUnavailableError(
            "Ha ocurrido un error: los servidores de OpenAI están saturados, intenta más tarde")



def test_resumir_exception_transcripts_disable():
    url = 'https://www.youtube.com/watch?v=u9NESbmj4bU'
    with pytest.raises(youtube_transcript_api._errors.TranscriptsDisabled, match="Ha ocurrido un error: los subtítulos están deshabilitados en este vídeo. No es posible resumirlo"):
        resumir(url)
        raise youtube_transcript_api._errors.TranscriptsDisabled(
            "Ha ocurrido un error: los subtítulos están deshabilitados en este vídeo. No es posible resumirlo")
