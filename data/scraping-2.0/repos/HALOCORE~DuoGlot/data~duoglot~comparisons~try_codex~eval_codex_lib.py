
import openai
import json
import os
import subprocess
import signal


################################## Utilities #######################################

def command_execute(command, timeout=10):
  print("--- Executing command:", command)
  try:
    proc = subprocess.Popen(command, cwd=os.path.abspath(os.path.dirname(__file__)), shell=True, preexec_fn=os.setsid)
    proc.wait(timeout)
  except Exception as e:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    raise e
  

def read_content(filename):
  with open(filename, 'r') as f:
    return f.read()

def save_content(filename, content):
  print("Write file:", filename)
  with open(filename, 'w') as f:
    f.write(content)
  os.system('chmod 777 ' + filename)

def append_content(filename, content):
  print("Write file:", filename)
  with open(filename, 'a') as f:
    f.write(content)


import time
import timeit
def complete_prompt(prompt, TEMP, LOGPROBS, MAX_TOKENS, MODEL_NAME):
  # pip install --upgrade openai
  openai.api_key = json.loads(read_content("./drvtry")) 
  current_max_tokens = MAX_TOKENS
  retry_count = 0
  while True:
    try:
      retry_count += 1
      time1 = timeit.default_timer()
      print(f"Querying OpenAI with: {MODEL_NAME} temp={TEMP} max_tokens={current_max_tokens} logprobs={LOGPROBS} len(prompt)={len(prompt)} ...")
      res = openai.Completion.create(
        engine=MODEL_NAME,
        prompt=prompt,
        max_tokens=current_max_tokens,
        temperature=TEMP,
        logprobs=LOGPROBS,
        # echo=True,
        n=1
      )
      time2 = timeit.default_timer()
      return str(res), time2 - time1
    except Exception as e:
      print(e)
      msg = str(e)
      if msg.startswith("This model's maximum context length is 4097 tokens"):
        request_size = int(msg.split("however you requested ")[1].split(" tokens")[0])
        prompt_size = request_size - current_max_tokens
        current_max_tokens = 4097 - prompt_size
        if (current_max_tokens < 0): return json.dumps({"error_msg": msg}), None
      elif msg.startswith("This model's maximum context length is 8001 tokens"):
        request_size = int(msg.split("however you requested ")[1].split(" tokens")[0])
        prompt_size = request_size - current_max_tokens
        current_max_tokens = 8001 - prompt_size
        if (current_max_tokens < 0): return json.dumps({"error_msg": msg}), None
      elif retry_count >= 8:
        print("!!! FAILED AFTER 8 RETRIES !!!")
        return json.dumps({"error_msg": msg}), None
      time.sleep(30)
      continue




################################## steps #######################################

def run_step_gold(bench_prefix, is_allinone, is_prepend_bef_test, gold_cleanup_func, source_path, testscript_list, gold_path, prompted_path, PROMPT, **otherargs):
  skipcount = 0
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    jsfile = file.replace(".py", ".js")
    filedir = file.replace(".py", "")
    if not jsfile in testscript_list:
      print(f"WARNING: {file} no testscript. Skipped for now.")
      skipcount += 1
      continue
    srcpath = os.path.join(source_path, file)
    content = read_content(srcpath)

    if not is_allinone:
      before_test = content.split("def test()")[0]
      gold_func = content.split('"-----------------"')[1]
      gold_code = gold_cleanup_func((before_test.strip() + '\n' + gold_func.strip()).strip() if is_prepend_bef_test else gold_func.strip())
    else:
      gold_code = gold_cleanup_func(content)
    
    gold_savepath = os.path.join(gold_path, file)
    save_content(gold_savepath, gold_code)
    prompted_savepath = os.path.join(prompted_path, file + ".txt")
    prompted_code = PROMPT(gold_code)
    save_content(prompted_savepath, prompted_code)
  print("SKIPPED:", skipcount)



def run_step_trans(bench_prefix, source_path, testscript_list, prompted_path, trans_path, TEMP, LOGPROBS, MAX_TOKENS_LIST, MODEL_NAME, **otherargs):
  skipcount = 0
  import random
  random.seed(42)
  shuffled_files = sorted(list(os.listdir(source_path)))
  random.shuffle(shuffled_files)
  for file in shuffled_files:
    if not file.startswith(bench_prefix): continue
    jsfile = file.replace(".py", ".js")
    prompted_file = file.replace(".py", ".py.txt")
    respfile = file.replace(".py", ".py.codex.json")
    if not jsfile in testscript_list:
      print(f"WARNING: {file} no testscript. Skipped for now.")
      skipcount += 1
      continue
    promptfilepath = os.path.join(prompted_path, prompted_file)
    trans_targetpath = os.path.join(trans_path, respfile)
    if os.path.exists(trans_targetpath):
      print(f"{trans_targetpath} exists. Skipping...")
      continue
    content = read_content(promptfilepath)
    resp, timespan = complete_prompt(content, TEMP, LOGPROBS, MAX_TOKENS_LIST[0], MODEL_NAME)
    print(f"({file}) TRANSTIME:" + str(timespan))
    save_content(trans_targetpath, resp)
    time.sleep(10)

  print("SKIPPED:", skipcount)


def run_step_trans_maximum(bench_prefix, trans_path, source_path, testscript_list, prompted_path, TEMP, LOGPROBS, MAX_TOKENS_LIST, MODEL_NAME, **otherargs):
  trans_clean_files = [x for x in sorted(list(os.listdir(trans_path))) if x.endswith(".clean.js")]
  skipcount = 0
  import random
  random.seed(42)
  shuffled_files = sorted(list(os.listdir(source_path)))
  random.shuffle(shuffled_files)
  for file in shuffled_files:
    if not file.startswith(bench_prefix): continue
    is_clean_jsfile_exists = file.replace(".py", ".clean.js") in trans_clean_files
    jsfile = file.replace(".py", ".js")
    prompted_file = file.replace(".py", ".py.txt")
    respfile = file.replace(".py", ".py.codex.json")
    if not jsfile in testscript_list:
      print(f"WARNING: {file} no testscript. Skipped for now.")
      skipcount += 1
      continue
    promptfilepath = os.path.join(prompted_path, prompted_file)
    trans_targetpath = os.path.join(trans_path, respfile)
    if is_clean_jsfile_exists:
      # print(f"# clean.js exists for {file}. Skipping...")
      continue
    else:
      # if os.path.exists(trans_targetpath):
      #   print(f"{trans_targetpath} exists. Skipping...")
      #   continue
      # print("DRY_RUN_TODO:", file)
      # continue
      content = read_content(promptfilepath)
      resp, timespan = complete_prompt(content, TEMP, LOGPROBS, MAX_TOKENS_LIST[1], MODEL_NAME)
      print(f"({file}) TRANSTIME:" + str(timespan))
      save_content(trans_targetpath, resp)
      time.sleep(10)

  print("SKIPPED:", skipcount)


def run_step_extract(bench_prefix, source_path, trans_path, **otherargs):
  not_exist_count = 0
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    respfile = file.replace(".py", ".py.codex.json")
    resp_trans_file = file.replace(".py", ".txt")
    trans_targetpath = os.path.join(trans_path, respfile)
    extract_txtpath = os.path.join(trans_path, resp_trans_file)
    if not os.path.exists(trans_targetpath): 
      not_exist_count += 1
      continue
    content = read_content(trans_targetpath)
    content_data = json.loads(content)
    # check for error message
    if "error_msg" in content_data:
      print(f"Skipping {file}: {content_data['error_msg']}")
      not_exist_count += 1
      continue

    assert len(content_data["choices"]) == 1
    first_choice = content_data["choices"][0]
    finish_reason = first_choice["finish_reason"]
    text = first_choice["text"]
    save_content(extract_txtpath, text)

def run_step_transclean(bench_prefix, source_path, trans_path, **otherargs):
  not_exist_count = 0
  not_finished_or_unexpected_count = 0
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    respfile = file.replace(".py", ".txt")
    resp_trans_file = file.replace(".py", ".clean.js")
    resp_transfix_file = file.replace(".py", ".clean.fix.js")
    raw_txtpath = os.path.join(trans_path, respfile)
    clean_txtpath = os.path.join(trans_path, resp_trans_file)
    clean_typofixpath = os.path.join(trans_path, resp_transfix_file)
    if not os.path.exists(raw_txtpath): 
      not_exist_count += 1
      continue
    content = read_content(raw_txtpath)
    if len(content.split("\n}")) < 2:
      print("FAILED TO CLEAN:", raw_txtpath)
      not_finished_or_unexpected_count += 1
      continue
    before, after = content.split("\n}")[:2]
    before = before + "\n}"
    save_content(clean_txtpath, before)
    ssfixed = before.replace("function f(", "function f_gold(")
    save_content(clean_typofixpath, ssfixed)
  print("not_exist_count:", not_exist_count, " not_finished_or_unexpected_count:", not_finished_or_unexpected_count)

def run_step_transcleanman(bench_prefix, source_path, trans_path, **otherargs):
  not_exist_count = 0
  not_finished_or_unexpected_count = 0
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    respfile = file.replace(".py", ".txt")
    resp_trans_file = file.replace(".py", ".clean.js")
    resp_transfix_file = file.replace(".py", ".clean.fix.js")
    raw_txtpath = os.path.join(trans_path, respfile)
    clean_txtpath = os.path.join(trans_path, resp_trans_file)
    clean_typofixpath = os.path.join(trans_path, resp_transfix_file)
    if not os.path.exists(raw_txtpath): 
      not_exist_count += 1
      continue
    
    content = read_content(raw_txtpath)
    if not os.path.exists(clean_txtpath):
      not_done_content = "// I AM NOT DONE\n" + content
      save_content(clean_txtpath, not_done_content)
    
    while True:
      cleaned_content = read_content(clean_txtpath)
      if cleaned_content.find("// I AM NOT DONE") >= 0:
        input(f"<PAUSED> Please manually clean-up {clean_txtpath} and then remove '//I AM NOT DONE'. \n         After you have done so, press ENTER...")
      else:
        save_content(clean_typofixpath, cleaned_content)
        break

    # before, after = content.split("\n}")[:2]
    # before = before + "\n}"
    
    # ssfixed = before.replace("function f(", "function f_gold(")
  print("Manual clean-up finished.")
  print("not_exist_count:", not_exist_count)


def run_step_fill(bench_prefix, source_path, testscript_list, trans_path, testscript_path, combined_savepath, **otherargs):
  skipcount = 0
  errorcount = 0
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    jsfile = file.replace(".py", ".js")
    jsfile1 = file.replace(".py", ".clean.js")
    jsfile2 = file.replace(".py", ".clean.fix.js")
    filedir = file.replace(".py", "")
    if not jsfile in testscript_list:
      print(f"WARNING: {file} no testscript. Skipped for now.")
      skipcount += 1
      continue
    trans_file = file.replace(".py", ".clean.js")
    trans_typofix_file = file.replace(".py", ".clean.fix.js")
    clean_txtpath = os.path.join(trans_path, trans_file)
    clean_fixedpath = os.path.join(trans_path, trans_typofix_file)
    if not os.path.exists(clean_txtpath):
      print(f"NOTICE: {file} no translation. count as error.")
      errorcount += 1
      continue
    goldtrans_content1 = read_content(clean_txtpath)
    goldtrans_content2 = read_content(clean_fixedpath)

    testpath = os.path.join(testscript_path, jsfile)
    test_content = read_content(testpath)

    def savecomb(content, filename):
      combpath = f"{os.path.join(combined_savepath, filename)}"
      combined_testcode = test_content.replace("//TRANSlATED_PLACEHOLDER_NO_OUTPUT_EXPECTED", content)
      combined_testcode = combined_testcode.replace("const SKIP_LOGGING = false", "const SKIP_LOGGING = true")
      save_content(combpath, combined_testcode)
    
    savecomb(goldtrans_content1, jsfile1)
    savecomb(goldtrans_content2, jsfile2)

  print("skipcount: " + str(skipcount) + "  errorcount: " + str(errorcount))


def run_step_run(bench_prefix, additionallogpath, source_path, testscript_list, combined_savepath, runoutputpath, **otherargs):
  skipcount = 0
  notranscount1 = 0
  notranscount2 = 0
  errcount1 = 0
  errcount2 = 0
  save_content(additionallogpath, "")
  for file in sorted(list(os.listdir(source_path))):
    if not file.startswith(bench_prefix): continue
    jsfile = file.replace(".py", ".js")
    jsfile1 = file.replace(".py", ".clean.js")
    jsfile2 = file.replace(".py", ".clean.fix.js")
    filedir = file.replace(".py", "")
    if not jsfile in testscript_list:
      print(f"WARNING: {file} no testscript. Skipped for now.")
      skipcount += 1
      continue
    srcpath = os.path.join(source_path, file)
    
    def runcomb(jsfilename):
      nonlocal notranscount1
      nonlocal notranscount2
      nonlocal errcount1
      nonlocal errcount2
      comb_path = os.path.join(combined_savepath, jsfilename)
      if not os.path.exists(comb_path):
        if jsfilename == jsfile1: 
          notranscount1 += 1 
        else: 
          assert jsfilename == jsfile2
          notranscount2 += 1 
        return
      output_stdout_path = os.path.join(runoutputpath, jsfilename + ".out")
      output_stderr_path = os.path.join(runoutputpath, jsfilename + ".err")
      try:
        command_execute(f"node {comb_path} >{output_stdout_path} 2>{output_stderr_path}")
      except Exception as e:
        append_content(additionallogpath, f"Failed to execute {comb_path}:\n{str(e)}\n")
        if jsfilename == jsfile1: 
          errcount1 += 1 
        else: 
          assert jsfilename == jsfile2
          errcount2 += 1 
    
    runcomb(jsfile1)
    runcomb(jsfile2)
  
  print("SKIPPED:", skipcount)
  print("NOTRANS1:", notranscount1)
  print("NOTRANS2:", notranscount2)
  print("ERROR1:", errcount1)
  print("ERROR2:", errcount2)



def run_step_stat(bench_prefix, additionallogpath, source_path, testscript_list, runoutputpath, is_ext = False, show_err = False, **otherargs):
  additional_log_dict = {y[0]:y[1] for y in [x.split(":\n") for x in read_content(additionallogpath).split("Failed to execute ") if x.strip() != ""]}
  
  for suffix in [".clean.js", ".clean.fix.js"]:
    skipcount = 0
    norun_count = 0
    executor_err_count = 0
    test_err_count = 0
    pass_count = 0
    pass_list = []
    test_err_list = []
    additional_log_content = read_content(additionallogpath)
    # additional_log_dict = {y[0]:y[1] for y in [x.split(":\n") for x in additional_log_content.split("Failed to execute ") if x.strip() != ""]}
    for file in sorted(list(os.listdir(source_path))):
      if not file.startswith(bench_prefix): continue
      jsfile = file.replace(".py", ".js")
      filljsfile = file.replace(".py", suffix)
      filedir = file.replace(".py", "")
      if not jsfile in testscript_list:
        print(f"WARNING: {file} no testscript. Skipped for now.")
        skipcount += 1
        continue
      if additional_log_content.find(jsfile) >= 0:
        executor_err_count += 1
        if show_err: print("SHOWERR:", file)
        test_err_list.append(file.replace(".py", ""))
        continue
      srcpath = os.path.join(source_path, file)
      output_stdout_path = os.path.join(runoutputpath, filljsfile + ".out")
      output_stderr_path = os.path.join(runoutputpath, filljsfile + ".err")
      if not os.path.exists(output_stdout_path):
        assert not os.path.exists(output_stderr_path)
        norun_count += 1
        continue
      err = read_content(output_stderr_path).strip()
      if err != "":
        test_err_count += 1
        if show_err: print("SHOWERR:", file)
        test_err_list.append(file.replace(".py", ""))
      else: 
        pass_count += 1
        pass_list.append(file.replace(".py", ""))
        
    ratio = pass_count / (pass_count + test_err_count)
    info = {"pass_list": pass_list, "test_err_list": test_err_list}
    extsuffix = "ext." if is_ext else ""
    save_content(f"./eval_codex_solved_set{suffix}.{extsuffix}json", json.dumps(info, indent=2))
    print("ratio:", ratio,  " skipcount:", skipcount, " norun_count:", norun_count, " executor_err_count:", executor_err_count, " pass_count:", pass_count, " test_err_count:", test_err_count)
    
STEP_FUNC_DICT = {
  "gold": run_step_gold,
  "trans": run_step_trans,
  "trans_maximum": run_step_trans_maximum,
  "extract": run_step_extract,
  "transclean": run_step_transclean,
  "transcleanman": run_step_transcleanman,
  "fill": run_step_fill,
  "run": run_step_run,
  "stat": run_step_stat
}

def run_step(step_name, all_args):
  if step_name in STEP_FUNC_DICT:
    return STEP_FUNC_DICT[step_name](**all_args)
  else: raise Exception("Unknown step")

def run_steps(
    step_names, 
    bench_prefix, 
    is_allinone,
    PROMPT, TEMP, LOGPROBS, MAX_TOKENS_LIST, MODEL_NAME,
    source_path, testscript_path, testscript_list, gold_path, prompted_path, trans_path, combined_savepath, additionallogpath, runoutputpath,
    is_prepend_bef_test, is_ext, show_err, 
    gold_cleanup_func
  ):
  all_args = {
    "bench_prefix": bench_prefix,
    # "PROMPT": PROMPT,
    "is_allinone": is_allinone,
    "TEMP": TEMP, 
    "LOGPROBS": LOGPROBS,
    "MAX_TOKENS_LIST": MAX_TOKENS_LIST, 
    "MODEL_NAME": MODEL_NAME,
    "source_path": source_path,
    "testscript_path": testscript_path,
    # "testscript_list": testscript_list,
    "gold_path": gold_path,
    "prompted_path": prompted_path, 
    "trans_path": trans_path, 
    "combined_savepath": combined_savepath, 
    "additionallogpath": additionallogpath,
    "runoutputpath": runoutputpath,
    "is_prepend_bef_test": is_prepend_bef_test,
    "is_ext": is_ext, 
    "show_err": show_err,
  }
  print(f"\n\n------------------------------- Run steps all_args: -------------------------------")
  print(json.dumps(all_args, indent=2))
  all_args["PROMPT"] = PROMPT
  all_args["testscript_list"] = testscript_list
  all_args["gold_cleanup_func"] = gold_cleanup_func

  for step_name in step_names:
    if step_name not in STEP_FUNC_DICT:
      raise Exception("Unknown step_name: " + step_name)
  for step_name in step_names:
    print(f"\n\n------------------------------- Run step: {step_name} -------------------------------")
    run_step(step_name, all_args)