import os, sys, json, time
from unittest import mock
sys.path.append('../')

from main import eval_sheet, grade_sheet
from modules.parse import parse_wrapper
from modules.oai_api import get_completion
from modules.local_llm_api import get_model_fn
from openai.types.chat import ChatCompletion

def load_chat_completion(fn: str) -> ChatCompletion:
    '''need this since oai_api.get_completion takes a ChatCompletion object'''
    with open(fn, 'r') as f:
        data = json.load(f)
        try:
            return ChatCompletion(**data)
        except Exception as e:
            print(e)
    
RESPONSE_STUB_FN = './data/stubs/completion.json'
MODEL_RESPONSE_STUB = load_chat_completion(RESPONSE_STUB_FN)

VALID_MODEL_PATH = '../../../data/llama-2-7b.Q4_K_M.gguf'


def test_stub_loaded():
    '''test to make sure subsequent tests are valid'''
    msg = get_completion(MODEL_RESPONSE_STUB)
    assert len(msg) > 0


def test_get_model_fn():
    # valid model_name should return a path
    model_fn = get_model_fn('llama_7b')
    assert model_fn.endswith('llama-2-7b.Q4_K_M.gguf')
    # valid path to a file should return the path
    # model_fn = get_model_fn(VALID_MODEL_PATH)
    # assert model_fn.endswith('llama-2-7b.Q4_K_M.gguf')
    # invalid model_name should raise ValueError
    try:
        model_fn = get_model_fn('llama_7b_fake')
        assert False
    except ValueError:
        assert True
    except:
        assert False


def test_eval_basic_1():
    '''
        demonstrate mocking:
         - output file(s): .md + .json
         - submit_prompt
    '''
    
    with mock.patch('modules.output.open',  mock.mock_open()) as mock_output_file:
        with mock.patch('main.submit_prompt') as mock_submit_prompt:
            
            mock_submit_prompt.return_value = MODEL_RESPONSE_STUB
            
            output = eval_sheet(
                './data/input-one.md',
                '../data/md-schema.yaml',
                'gpt-3.5-turbo',
                'should-not-be-used.txt',
                verbose_level=0,
            )

            # two questions thus it should be called twice
            assert mock_submit_prompt.call_count == 2
    
            # verify the output file is written to, but not much else
            written_data = [e[0][0] for e in mock_output_file().write.call_args_list]
            
            assert len(written_data) > 0


def test_eval_basic_2():
    '''
        test with local model: llama_7b
    '''
    
    with mock.patch('modules.output.open',  mock.mock_open()) as mock_output_file:
        with mock.patch('main.prompt_model') as mock_prompt_model:
            
            mock_prompt_model.return_value = ("stubbed answer", None)
            
            output = eval_sheet(
                './data/input-one.md',
                '../data/md-schema.yaml',
                'llama_7b',
                'should-not-be-used.txt',
                verbose_level=0,
            )

            # two questions thus it should be called twice
            assert mock_prompt_model.call_count == 2
    
            # verify the output file is written to, but not much else
            written_data = [e[0][0] for e in mock_output_file().write.call_args_list]
            
            assert len(written_data) > 0
            

def test_eval_grading():
    '''
        test the grading functionality:
        we'll capture if grading output file has the correct
        array of booleans by saying true answer to Question-1 
        in the mock response of the llama completion method
    '''
    
    TEST_FN = './data/dir-two/input-one.md'
    TEST_SCHEMA = '../data/md-schema.yaml'

    # A) capture the output_obj of a run
    # setting grading output to None to prevent output_obj from containing 
    # a graded section
    with mock.patch('modules.output.open',  mock.mock_open()) as mock_output_file:
        with mock.patch('main.prompt_model') as mock_prompt_model:
            
            mock_prompt_model.return_value = ('C) The worm', None)        
            
            output_obj = eval_sheet(
                input_md_fn=TEST_FN,
                input_schema_fn=TEST_SCHEMA,
                model_name='llama_7b',
                output_md_fn='output-stub-xx.md',
                output_json_fn=None,     # force only output to be output-xx.md
                output_grade_fn=None,    # prevent output-md from having
            )

            # The test sheet has two questions, verify those are being being hit
            assert mock_prompt_model.call_count == 2

    # B) now do grading call here, instead of referencing the output json
    
    # first grab the doc_obj which would exist inside eval_sheet call 
    # and be passed to it normally
    input_doc_obj = parse_wrapper(
        TEST_FN,
        TEST_SCHEMA,
    )

    # Now run the output objects from (A) and (B) through the grading function
    # analyze the boolean outputs
    list_grades = grade_sheet(
        json_doc=input_doc_obj,
        output_obj=output_obj,
    )

    print(f"list_grades: {list_grades}")

    assert len(list_grades) == 2

    # This one should be true because we mocked its response to be correct
    assert list_grades[0] == True

    # This one should be false because we mocked its response to be incorrect
    assert list_grades[1] == False
