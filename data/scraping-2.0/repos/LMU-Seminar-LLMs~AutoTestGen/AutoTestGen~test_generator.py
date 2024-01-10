import openai
import logging
from . import config
from .container_manager import ContainerManager
from .templates import list_errors, combine_samples
from .templates import (
    COMPILING_ERROR_REPROMPT,
    COMBINING_SAMPLES_PROMPT,
    TEST_ERROR_REPROMPT
)

def generate_tests(
    initial_prompt: list[dict[str, str]],
    cont_manager: ContainerManager,
    obj_name: str,
    temp: float=0.1,
    n_samples: int=1,
    max_iter: int=5,
    logger: logging.Logger=logging.getLogger(__name__)
) -> dict:
    """
    Runs the pipeline for generating tests.

    Args:
        initial_prompt: Initial prompt to use for generating tests.
        obj_name: Name of the object to test.
        temp (float): Temperature to use for sampling.
        n_samples (int): Number of samples to generate.
        max_iter (int): Maximum number of iterations to run.
        logger (logging.Logger): App logger.
    
    Returns:
        dict: Dictionary containing:
            - messages (list[dict[str, str]]): List of messages
            - test (str): Source code of the test.
            - report (dict): Test results and coverage report.
    
    Raises:
        ValueError: If the API_KEY or MODEL is not set.
    """

    logger.info("Sending initial prompt to OpenAI API ...")
    response = _generate_response(initial_prompt, n_samples, temp)
    result: list[dict] = []

    if len(response) > 1:
        logger.info(f"Received {len(response)} responses from OpenAI API")
        # If n_sample > 1: Preporcessing Step is done:
        #  - Evaluate each response separately and use the information to
        #  - put together initial promt.
        sample_results = []
        for i, resp in enumerate(response): 
            # PostProcess response
            logger.info(f"Postprocessing response {i + 1} ...")
            post = config.ADAPTER.postprocess_resp(resp, obj_name=obj_name)
            # Run response
            logger.info(f"Running response {i + 1} in container ...")
            test_report = cont_manager.run_tests_in_container(post)
            logger.info("Observing Results ...")
            if test_report["compile_error"]:
                result = (
                    "Provided code failed to compile with:\n"
                    f"{test_report['compile_error']}"
                )
            elif test_report["errors"]:
                result = (
                    "Executing tests failed with the following errors:\n"
                    f"{list_errors(test_report['errors'])}"
                )
            else:
                result = "Tests were successfully executed."
            sample_results.append((resp, result))
            
        # Update initial prompt
        logger.info("Combining samples and reprompting API ...")
        initial_prompt[1]["content"] = COMBINING_SAMPLES_PROMPT.format(
            initial_prompt=initial_prompt[1]["content"],
            n_samples=n_samples,
            combined_samples=combine_samples(sample_results),
            language=config.ADAPTER.language
        )
        # Generate new single response
        response = _generate_response(initial_prompt, 1, temp)
        
    # Run iterations
    for i in range(max_iter):
        logger.info(f"Starting Iteration {i + 1} ...")
        # PostProcess
        resp_post = config.ADAPTER.postprocess_resp(
            response[0],
            obj_name=obj_name
        )
        # Run tests
        logger.info("Running response in container ...")
        test_report = cont_manager.run_tests_in_container(resp_post)
        # Infer
        if test_report["compile_error"]:
            logger.info("Code failed to compile")
            # If compiling code failed: reprompt
            new_prompt = COMPILING_ERROR_REPROMPT.format(
                error_msg=test_report["compile_error"],
                language=config.ADAPTER.language
            )
            initial_prompt.extend(
                [
                    {'role': 'assistant', 'content': resp_post},
                    {'role': 'user', 'content': new_prompt}
                ]
            )
            logger.info("Reprompting API ...")
            response = _generate_response(initial_prompt, 1, temp)
            continue
        
        elif test_report["errors"]:
            logger.info("Some of the tests failed to run")
        # Errors occured while running tests: reprompt
            new_prompt = TEST_ERROR_REPROMPT.format(
                id_error_str=list_errors(test_report["errors"]),
                language=config.ADAPTER.language
            )
            # If errors occured
            initial_prompt.extend(
                [
                    {'role': 'assistant', 'content': resp_post},
                    {'role': 'user', 'content': new_prompt}
                ]
            )
            logger.info("Reprompting API ...")
            response = _generate_response(initial_prompt, 1, temp)
            continue
        else:
            logger.info("Tests were successfully executed")
            break

    # If max_iter reached and no valid response: return last resp.
    result = {
        "messages": initial_prompt,
        "test": resp_post,
        "report": test_report
    }
    return result
        
def _generate_response(
    messages: list[dict[str, str]],
    n_samples: int,
    temp: float
) -> list[str]:
    """
    Prompts OpenAI API. Helper function for generate_tests_pipeline.

    Args:
        messages: List of dicts containing role-content keys.
        n_samples (int): Number of samples to generate.
        temp (float): Temperature to use for sampling.
    """
    if config.API_KEY is None:
        raise ValueError("API_KEY is not set. Call set_api_keys first.")
    if config.MODEL is None:
        raise ValueError("MODEL is not set. Call set_model first.")
    
    openai.api_key = config.API_KEY
    openai.organization = config.ORG_KEY

    resp = openai.ChatCompletion.create(
        model=config.MODEL,
        messages=messages,
        temperature=temp,
        n=n_samples
    )
    resp_lst = [r["message"]["content"] for r in resp["choices"]]
    return resp_lst