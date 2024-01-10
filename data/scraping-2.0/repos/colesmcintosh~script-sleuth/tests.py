from langchain.prompts import PromptTemplate
import main

def test_fetch_code_file_exit(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: 'exit')
    assert main.fetch_code_file() == (None, None, None, None)

def test_fetch_code_file_select_first(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: '1')
    monkeypatch.setattr('main.glob.glob', lambda _ , **__: ['test_dir/hello_world.cpp'])
    result = main.fetch_code_file('test_dir')
    assert result[1].split('/')[-1] == 'hello_world.cpp'
    assert result[3] == 'C++'

def test_format_prompt(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: 'What does this function do?')
    result_prompt, result_question = main.format_prompt('test.py')
    assert result_question == 'What does this function do?'
    assert isinstance(result_prompt, PromptTemplate)

def test_get_llm_openai(monkeypatch):
    mock_model = "mock_model"
    monkeypatch.setattr('main.ChatOpenAI', lambda *args, **kwargs: mock_model)
    assert main.get_llm("openai") == mock_model

def test_get_llm_huggingface_hub(monkeypatch):
    mock_model = "mock_model"
    monkeypatch.setattr('main.HuggingFaceHub', lambda *args, **kwargs: mock_model)
    assert main.get_llm("huggingface") == mock_model

def test_get_llm_huggingface_pipeline(monkeypatch):
    mock_model = "mock_model"
    monkeypatch.setattr('main.HuggingFacePipeline.from_model_id', lambda *args, **kwargs: mock_model)
    assert main.get_llm("huggingface", local_model=True) == mock_model

def test_main_exit(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda *args: 'exit')
    assert main.main() is None

def test_format_prompt_back(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _: 'back')
    result_prompt, result_question = main.format_prompt('test.py')
    assert result_question == 'back'

def test_main_quit(monkeypatch):
    inputs = iter(['test_dir', '1', 'quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert main.main() is None

def test_main_back(monkeypatch):
    inputs = iter(['test_dir', '1', 'back', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert main.main() is None

