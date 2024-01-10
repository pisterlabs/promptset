################################################################################
# C to Rust translation - inputs are the IntroClass given-solution per benchmark

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
  template="""
    Translate the following C code to Rust.
    The code must be a direct translation, do not change the logic nor add anything else:
    \n{code}
  """
)
prompt_with_previous_compilation_error = PromptTemplate(
  input_variables=[ "code", "previous_compilation_error" ],
  template="""
    Directly translate the following C code to Rust
    (don't forget to fix the compilation errors displayed below; common examples
    are type mismatches and forgetting to use `use std::io` and the likes):
    \n{code}
    \nError: {previous_compilation_error}.
  """
)

prompt_with_previous_test_failure = PromptTemplate(
  input_variables=[ "code", "expected_output", "actual_output" ],
  template="""
    Directly translate the following C code to Rust
    (don't forget to fix the test's output-errors displayed below):
    \n{code}
    \nExpected output: {expected_output}
    \nActual output: {actual_output}.
  """
)

def perform_query(code, model, previous_compilation_error = None, previous_test_failure = None):
  if previous_compilation_error:
    chain = LLMChain(
      prompt=prompt_with_previous_compilation_error,
      llm=model
    )
    reply = chain.run({"code": code, "previous_compilation_error": previous_compilation_error})
  elif previous_test_failure:
    expected_output, actual_output = previous_test_failure
    chain = LLMChain(
      prompt=prompt_with_previous_test_failure,
      llm=model
    )
    reply = chain.run({
      "code": code,
      "expected_output": expected_output,
      "actual_output": actual_output
    })
  else:
    print_debug("No previous error")
    chain = LLMChain(prompt=prompt, llm=model)
    reply = chain.run(code)
  reply = reply.partition("```rust")[2] # get everything after the code starts being written
  reply = reply.partition("```")[0] # we can discard everything after the code ends
  # there's also some cases where the final ``` doesn't seem to be put (?)
  reply = reply.partition("<|end|>")[0]
  # this assumes that no-one used ``` along the code itself, which is a bit of a hack
  return reply

################################################################################

SRC_DIR = os.getcwd()
BENCHMARK_LOCATION = SRC_DIR + "/../data/IntroClass/"
RUST_CODE_LOCATION = SRC_DIR + "/../data/c-to-rust-correct/"
BENCHMARKS = map(
 lambda b: BENCHMARK_LOCATION + b,
 [ "checksum/", "digits/", "grade/", "median/", "smallest/", "syllables/" ]
)
TO_AVOID = [ "tests" ]
TEST_TYPES = [ "blackbox", "whitebox" ]

# I want objects which have both _what happened_ and also, if there was an error, its error code
# This is useful to re-ask the model to fix the code, considering the error code
# Moreover, for test failures, we can also give back the expected output and the actual output
class QueryResult:
  def __init__(self, result, error = None, outputs = None):
    # Note that result is a string, which may be "COMPILER_FAILURE", "TEST_FAILURE" or "TEST_SUCCESS"
    self.result = result
    self.error = error
    self.outputs = outputs # a tuple with two strings, the expected output and the actual output


def cargo_init(rust_dir):
  pwd = os.getcwd()
  os.chdir(rust_dir)
  subprocess.run(["cargo", "init"])
  os.chdir(pwd)

# With more time, using rust's test feature (w/ `cargo test`) could be fun
# It seemed a bit too complicated to me for this exercise, though
def create_tests(rust_dir, benchmark):
  # we want to copy the tests/ folder from the benchmark to rust_dir
  pwd = os.getcwd()
  os.chdir(benchmark)
  subprocess.run(["cp", "-r", "tests/", rust_dir])
  os.chdir(pwd)

def test_code(rust_dir):
  global compilation_failures, test_failures, test_successes

  # We'll try 3 runs to try and compile, and 3 others to run the tests
  for attempt in range(1, 4):
    pwd = os.getcwd()
    os.chdir(rust_dir)
    compilation_result = subprocess.run(["rustc", "src/main.rs"], capture_output=True, text=True)
    os.chdir(pwd)

    if compilation_result.returncode != 0:
      print_debug(f"Compilation failure: {compilation_result.stderr}")
      compilation_failures += 1
      if attempt == 3:
        return QueryResult("COMPILER_FAILURE", compilation_result.stderr)
    else:
      break

  for attempt in range(1, 4):
    test_result = run_tests(rust_dir)
    if test_result.result == "COMPILER_FAILURE":
      compilation_failures += 1
      # We don't want to keep trying to run the tests if we can't even compile, plus could lead to an infinite loop
      return test_result
    elif test_result.result == "TEST_FAILURE":
      print_debug(f"Test failure: {test_result.outputs}")
      test_failures += 1
      if attempt == 3:
        return test_result
    else:
      break
  
  test_successes += 1
  return QueryResult("TEST_SUCCESS")

def run_tests(rust_dir):
  # Once again, we won't be using `cargo test`, but rather just running the program itself with specific inputs (and checking the outputs)
  # The tests are just simple .in files, with the expected output being in the corresponding .out file
  pwd = os.getcwd()
  os.chdir(rust_dir)
  for test_type in TEST_TYPES:
    for test in os.listdir(f"tests/{test_type}"):
      if not test.endswith(".in"):
        continue
      test_name = test[:-3]
      with open(f"tests/{test_type}/{test}", "r") as test_file:
        input_data = test_file.read()
        result = subprocess.run(["./main"], input=input_data.encode("utf-8"), capture_output=True, timeout=5)
        with open(f"tests/{test_type}/{test_name}.out", "r") as expected_output:
          expected_output = expected_output.read()
          # Ideally we'd keep checking more and more tests, but for simplicity's
          # sake we'll just stop at the first failure
          if expected_output != result.stdout.decode("utf-8"):
            os.chdir(pwd)
            print_debug(f"Test failure for {test_type}/{test_name}.in")
            print_debug(f"Expected output: {expected_output}")
            print_debug(f"Actual output: {result.stdout.decode('utf-8')}")
            return QueryResult("TEST_FAILURE", outputs = (expected_output, result.stdout.decode("utf-8")))
  os.chdir(pwd)
  return QueryResult("TEST_SUCCESS")

def process_submission(
  submission_path, benchmark_name,
  no_compilation_errors = True, previous_error = None,
  no_test_errors = True, previous_test_failure = None
):
  try:
    print_debug(f"Processing {submission_path + benchmark_name + '.c'}")
    with open(submission_path + benchmark_name + ".c", "r") as s:
      code = s.read()
      llm = create_model(seed = randint(0, 1000000))
      reply = perform_query(code, llm, previous_error, previous_test_failure)
      if not os.path.isdir(rust_dir + "/src"):
        os.mkdir(rust_dir + "/src")
      with open(rust_dir + "/src/main.rs", "w") as rust_file:
        rust_file.write(reply)
      query_result = test_code(rust_dir)
      
      match query_result.result:
        case "COMPILER_FAILURE":
          if no_compilation_errors:
            process_submission(submission_path, benchmark_name, False, query_result.error, no_test_errors)
        case "TEST_FAILURE":
          if no_test_errors:
            # I'm forcing False for no_compilation_errors, just so there's no possibility of back-and-forth between the two
            process_submission(submission_path, benchmark_name, False, None, False, query_result.outputs)
        case "TEST_SUCCESS":
          print_info(f"Test success for {submission_path + benchmark + '.c'}")

  except FileNotFoundError:
    print(f"The submission {submission_path + benchmark_name + '.c'} was not found.")
  except Exception as e:
    print(f"An error occurred: {str(e)}")

################################################################################

compilation_failures = 0
test_failures = 0
test_successes = 0

if __name__ == "__main__":
  for benchmark in BENCHMARKS:
    print_debug(f"Processing benchmark {benchmark}")
    # create directory in c-to-rust if it hasn't been already done
    benchmark_name = benchmark.split("/")[-2]
    test_folder = benchmark + "tests/"
    rust_dir = RUST_CODE_LOCATION + benchmark_name

    if not os.path.isdir(rust_dir):
      os.mkdir(rust_dir)
      cargo_init(rust_dir)

    # we also need to create the tests
    create_tests(rust_dir, benchmark)
    process_submission(test_folder, benchmark_name)

  print_info(f"Compilation failures: {compilation_failures}, Test failures: {test_failures}, Test successes: {test_successes}")
