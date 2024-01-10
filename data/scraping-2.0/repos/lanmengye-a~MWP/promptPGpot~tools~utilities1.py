# just like progress "Leveraging Training Data in Few-Shot Prompting for Numerical Reasoning"
import collections
import re
import random
import time
from functools import lru_cache
import openai
import sympy
from tool import safe_execute,floatify_ans

from tools import chatgpt0



def replace_variate(expression):
    # 将expression中的变量number#替换为a-z
    var_arr = []
    for i in range(10):
        expression = expression.replace(f"number{i}", chr(97+i))
        var_arr.append(chr(97 + i))
    return expression,var_arr
    # expression = expression.replace("number0", "a")

from tool import parse_api_result
def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision
from math import isclose
from typing import Union
def finqa_equal(prediction: Union[int, float],
                reference: Union[float, int],
                include_percentage: bool = False,
                is_close: float = False) -> bool:
    if prediction is None:
        return False
    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.001):
                    return True
            precision = min(get_precision(prediction), get_precision(item))
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False
from math import isclose
from typing import Union
def one_patience(prompt,midfix):
    get_result = False
    while not get_result :
        try:
            result = call_gpt3(prompt)
            # if prediction is not None:
            get_result = True
        except Exception as e:
            print("there is an error",e)
            time.sleep(3)
    codes = parse_api_result(result)
    result_counter = collections.Counter()
    result_dict = collections.defaultdict(list)
    for idx, r in enumerate(codes):
        ans = safe_execute(r)
        ans = floatify_ans(ans)
        codes[idx] = midfix + r.strip("\n\n")
        if ans is not None and not isinstance(ans, str):
            result_counter.update([abs(ans)])
            result_dict[abs(ans)].append(idx)
    if len(result_counter) > 0:
        prediction = result_counter.most_common(1)[0][0]
        program = codes[result_dict[prediction][0]].replace("    ", "")
    else:
        prediction, program = None, None
    return program, prediction

def type_diag(ans:Union[str,int,float]):
    if isinstance(ans,str):
        ans = eval(ans)
    else:
        ans = float(ans)
    return ans
def get_gpt3_output(prompt, patience=1,midfix=None,label = None):
    program, prediction = None, None
    while patience!=0:
        # print(f"patience:{patience}")
        program, prediction = one_patience(prompt, midfix)
        if label is not None:
            label = type_diag(label)
            if not finqa_equal(prediction,label):
                print(" there is an unequal result")
                patience -= 1
            else:
                patience = 0
        else:
            patience = 0
    return program, prediction


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


