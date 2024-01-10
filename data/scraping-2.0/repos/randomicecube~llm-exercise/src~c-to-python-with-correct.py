################################################################################
# C to Python translation - inputs are the IntroClass given-solution per benchmark

import os
import subprocess
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from random import randint

print_debug = lambda arg: print("[DEBUG] " + str(arg))
print_info = lambda arg: print("[INFO] " + str(arg))
print_error = lambda arg: print("[ERROR] " + str(arg))

################################################################################

# Pass False as an argument if you don't use agenix (slash if you aren't me)
def get_api_token(agenix = True):
  if agenix:
    pwd = os.getcwd()
    os.chdir(os.environ.get("HOME") + "/gaspafiles/secrets")
    key = subprocess.check_output(
      ["agenix", "-d", "hugging-face.age"]
    ).decode("utf-8").replace("\n", "")
    os.chdir(pwd)
    return key
  # Otherwise just have a .env file and have your API key there, read it and so on
  return os.environ.get("HUGGING_FACE_API_KEY")

MODEL = "HuggingFaceH4/starchat-beta"
API_TOKEN = get_api_token()

################################################################################

def create_model(seed):
  return HuggingFaceHub(
    repo_id=MODEL,
    huggingfacehub_api_token=API_TOKEN,
    task = "text-generation",
    model_kwargs = {
      "max_new_tokens": 512,
      "repetition_penalty": 1.05,
      "temperature": 0.15,
      "top_p": 0.975,
      "return_full_text": True,
      "seed": seed,
    }
)

prompt = PromptTemplate(
  input_variables=[ "code" ],
  # The __main__ part seems to be important for it not to just do the function itself
  template="""
    Directly translate the following C code to Python
    (don't forget to add a __main__, and don't forget the input and output strings displayed must be EXACTLY the same (including spaces and newlines)):
    \n{code}
  """
)

prompt_with_previous_test_failure = PromptTemplate(
  input_variables=[ "code", "expected_output", "actual_output" ],
  template="""
    Directly translate the following C code to Python
    (don't forget to fix the test errors, that the input and output strings displayed must be EXACTLY the same (including spaces and newlines), and to have a __main__):
    \n{code}\nExpected output: {expected_output}\nActual output: {actual_output}.
  """
)

def perform_query(code, model, previous_test_failure = None):
  if previous_test_failure:
    expected_output, actual_output = previous_test_failure
    chain = LLMChain(prompt=prompt_with_previous_test_failure, llm=model)
    reply = chain.run({
      "code": code,
      "expected_output": expected_output,
      "actual_output": actual_output
    })
  else:
    print_debug("No previous error")
    chain = LLMChain(prompt=prompt, llm=model)
    reply = chain.run(code)
  
  print_info(reply)
  reply = reply.partition("```python")[2] # get everything after the code starts being written
  reply = reply.partition("```")[0] # we can discard everything after the code ends
  # there's also some cases where the final ``` doesn't seem to be put (?)
  reply = reply.partition("<|end|>")[0]
  # this assumes that no-one used ``` along the code itself, which is a bit of a hack
  return reply

################################################################################

SRC_DIR = os.getcwd()
BENCHMARK_LOCATION = SRC_DIR + "/../data/IntroClass/"
PYTHON_CODE_LOCATION = SRC_DIR + "/../data/c-to-python-correct/"
BENCHMARKS = map(
 lambda b: BENCHMARK_LOCATION + b,
 [ "checksum/", "digits/", "grade/", "median/", "smallest/", "syllables/" ]
)
TO_AVOID = [ "tests" ]
TEST_TYPES = [ "blackbox", "whitebox" ]

class QueryResult:
  def __init__(self, result, outputs = None):
    # Note that result is a string, which may be "COMPILER_FAILURE", "TEST_FAILURE" or "TEST_SUCCESS"
    self.result = result
    self.outputs = outputs # a tuple with two strings, the expected output and the actual output


def create_tests(python_dir, benchmark):
  # we want to copy the tests/ folder from the benchmark to python_dir
  pwd = os.getcwd()
  os.chdir(benchmark)
  subprocess.run(["cp", "-r", "tests/", python_dir])
  os.chdir(pwd)

def test_code(python_dir):
  # We'll try 3 runs, and if all of them fail, we'll just give up
  for attempt in range(1, 4):
    test_result = run_tests(python_dir)
    if test_result.result == "TEST_FAILURE":
      print_debug(f"Test failure: {test_result.outputs}")
      if attempt == 3:
        return test_result
      continue
    break

  return QueryResult("TEST_SUCCESS")

def run_tests(python_dir):
  # The tests are just simple .in files, with the expected output being in the corresponding .out file
  pwd = os.getcwd()
  os.chdir(python_dir)
  for test_type in TEST_TYPES:
    for test in os.listdir(f"tests/{test_type}"):
      if not test.endswith(".in"):
        continue
      test_name = test[:-3]
      with open(f"tests/{test_type}/{test}", "r") as test_file:
        input_data = test_file.read()
        result = subprocess.run(["python", "src/main.py"], input=input_data, capture_output=True, text=True, timeout=5)
        with open(f"tests/{test_type}/{test_name}.out", "r") as expected_output:
          expected_output = expected_output.read()
          # Ideally we'd keep checking more and more tests, but for simplicity's
          # sake we'll just stop at the first failure
          if expected_output != result.stdout:
            print_debug(f"Test failure for {test_type}/{test_name}.in")
            print_debug(f"Expected output: {expected_output}")
            print_debug(f"Actual output: {result.stdout}")
            os.chdir(pwd)
            return QueryResult("TEST_FAILURE", (expected_output, result.stdout))
  os.chdir(pwd)
  return QueryResult("TEST_SUCCESS")

def process_submission(
  submission_path, benchmark_name,
  no_test_errors = True, previous_test_failure = None
):
  global compilation_failures, test_failures, test_successes
  try:
    print_debug(f"Processing {submission_path + benchmark_name + '.c'}")
    with open(submission_path + benchmark_name + ".c", "r") as s:
      code = s.read()
      llm = create_model(seed=randint(0, 1000000))
      reply = perform_query(code, llm, previous_test_failure)
      if not os.path.isdir(python_dir + "/src"):
        os.mkdir(python_dir + "/src")
      with open(python_dir + "/src/main.py", "w") as python_file:
        python_file.write(reply)
      query_result = test_code(python_dir)
      
      match query_result.result:
        case "TEST_FAILURE":
          if no_test_errors:
            process_submission(submission_path, benchmark_name, False, query_result.outputs)
          else:
            print_info(f"Test failure for {submission_path + benchmark_name + '.c'}")
            test_failures += 1
        case "TEST_SUCCESS":
          print_info(f"Test success for {submission_path + benchmark + '.c'}")
          test_successes += 1

  except FileNotFoundError:
    print(f"The submission {submission_path + benchmark_name + '.c'} was not found.")
  except Exception as e:
    print(f"An error occurred: {str(e)}")

test_failures = 0
test_successes = 0

if __name__ == "__main__":
  for benchmark in BENCHMARKS:
    print_debug(f"Processing benchmark {benchmark}")
    benchmark_name = benchmark.split("/")[-2]
    test_folder = benchmark + "tests/"
    python_dir = PYTHON_CODE_LOCATION + benchmark_name

    if not os.path.isdir(python_dir):
      os.mkdir(python_dir)

    create_tests(python_dir, benchmark)
    process_submission(test_folder, benchmark_name)

  print_info(f"Test failures: {test_failures}, Test successes: {test_successes}")

