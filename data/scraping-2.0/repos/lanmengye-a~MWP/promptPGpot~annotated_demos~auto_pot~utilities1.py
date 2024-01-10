# just like progress "Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning"
import collections
import re
import random
import time
from functools import lru_cache
import openai
import sympy
from tool import safe_execute,floatify_ans
random.seed(123)
import chatgpt0



def replace_variate(expression):
    # 将expression中的变量number#替换为a-z
    var_arr = []
    for i in range(10):
        expression = expression.replace(f"number{i}", chr(97+i))
        var_arr.append(chr(97 + i))
    return expression,var_arr
    # expression = expression.replace("number0", "a")

from tool import parse_api_result
def get_gpt3_output(prompt, patience,midfix=None,label = None):
    program,prediction = None,None

    while True:
        start = time.time()
        try:
            if time.time()-start > 10:
                raise Exception("time out")
            result = call_gpt3(prompt)
            codes = parse_api_result(result)
            # codes = call_gpt3("text-davinci-003",question)
            result_counter = collections.Counter()
            result_dict = collections.defaultdict(list)
            for idx, r in enumerate(codes):
                ans = safe_execute(r)
                ans = floatify_ans(ans)
                codes[idx] = midfix+r
                if ans is not None and not isinstance(ans, str):
                    result_counter.update([(ans)])
                    result_dict[ans].append(idx)
                    # code_valid.append(r)
            if len(result_counter) > 0:
                prediction = result_counter.most_common(1)[0][0]
            if label is None or prediction is not None  and (prediction)==(label):
                program = codes[result_dict[prediction][0]].replace("    ", "")
                break
            if prediction is not None  and (prediction)!=(label):
                raise Exception("wrong prediction when training")
        except openai.error.RateLimitError as e:
            print("there is a error:",e)
            time.sleep(min((5-patience)**2, 60))
        except Exception as e:
            print("there is a error:",e)
            #判断错误类型
            print(patience)
            patience -= 1
            if patience == 0:
                break
                # print("there is some error ", pidId, ":", raw_formula,formula)
            time.sleep(1)


    return program,prediction

@lru_cache(maxsize=10000)
def call_gpt3(prompt):
    reply = chatgpt0.call_gpt3(prompt)
    return reply

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(formula,numbs):
    try:
        args = str(numbs).split()
        size = len(args)
        var_arr = []
        for i in range(size):
            var_arr.append(chr(97 + i))
        var_str = " ".join(var_arr)
        var = sympy.symbols(var_str)
        if (len(var) == 1):
            var = [var]
        dic = zip(var, args)
        # 将formula 转为sympy表达式
        formula = sympy.sympify(formula)
        result = formula.subs(dic)
        return result
        raise
    except Exception as e:
        return False

def normalize_formula(formula):
    try:

        formula = formula.split("Answer =")[1]
        formula = formula.strip(".")
        formula = formula.replace(" ", "")
        formula, vars = replace_variate(formula)
        formula = re.findall(r"(\([\da-z\+\-*\/()]+\))", formula)[0]
    except Exception as e:
        return False
    return formula


def normalize_answer(text):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]
    if not isinstance(text, str):
        text = str(text)  # 将非字符串对象转换为字符串
    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)
    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)
        try:
            if result is not None:
                number = int(text)
            elif "/" in text:
                nums = text.split("/")
                number = round(float(nums[0]) / float(nums[1]), 3)
            else:
                number = round(float(text), 3)
            number = str(number)
            number = re.sub(r"\.[0]+$", "", number)
            return number
        except:
            return text


