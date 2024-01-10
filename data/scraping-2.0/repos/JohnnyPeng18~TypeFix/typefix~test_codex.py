import openai
import json
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
) 
from __init__ import logger
from bug_locator import FunctionLocator
import ast
import re
import os
from tqdm import tqdm
import traceback
from ast_operation import ASTDiffer, CommentRemover



example_1 = """
# Provide a fix for the buggy function
# Buggy Function
def _find_offsets(radars, projparams, grid_origin_alt):
    offsets = []
    for radar in radars:
        x_disp, y_disp = geographic_to_cartesian(
            radar.longitude['data'], radar.latitude['data'], projparams)
        try:
            z_disp = float(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, float(y_disp), float(x_disp)))
        except:
            z_disp = np.mean(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, np.mean(y_disp), np.mean(x_disp)))
    return offsets
# Fixed Function
def _find_offsets(radars, projparams, grid_origin_alt):
    offsets = []
    for radar in radars:
        x_disp, y_disp = geographic_to_cartesian(
            radar.longitude['data'], radar.latitude['data'], projparams)
        try:
            z_disp = float(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, float(y_disp), float(x_disp)))
        except TypeError:
            z_disp = np.mean(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, np.mean(y_disp), np.mean(x_disp)))
    return offsets
"""

example_2 = """
# Provide a fix for the buggy function
# Buggy Function
def get_indices(self):
    cons = []
    for dfn in self.bonds + self.dihedrals + self.angles:
        cons.extend(dfn[1])
    for dfn in self.bondcombos:
        for bond in dfn[1]:
            cons.extend(bond[0:2])
    return set(cons)
# Fixed Function
def get_indices(self):
    cons = []
    for dfn in self.bonds + self.dihedrals + self.angles:
        cons.extend(dfn[1])
    for dfn in self.bondcombos:
        for bond in dfn[1]:
            cons.extend(bond[0:2])
    return list(set(cons))
"""

example_3 = """
# Provide a fix for the buggy function
# Buggy Function
def refresh(self):
    if self.checksum is None:
        self.checksum = self.id + self.ctrl
        for i in range(len(self.params)):
            self.checksum += self.params[i]
        self.checksum = self.checksum % 256
        self.checksum = 2 ** 8 - self.checksum
        self.checksum = self.checksum % 256
        self.len = 0x02 + len(self.params)
# Fixed Function
def refresh(self):
    if self.checksum is None:
        self.checksum = self.id + self.ctrl
        for i in range(len(self.params)):
            if isinstance(self.params[i], int):
                self.checksum += self.params[i]
            else:
                self.checksum += int(self.params[i].encode('hex'), 16)
        self.checksum = self.checksum % 256
        self.checksum = 2 ** 8 - self.checksum
        self.checksum = self.checksum % 256
        self.len = 0x02 + len(self.params)
"""

example_4 = """
# Provide a fix for the buggy function
# Buggy Function
def default(self, obj):
    if isinstance(obj, datetime.datetime):
        tz = obj.strftime("%z")
        tz = tz if tz else '+0000'
        return obj.strftime("%Y-%m-%dT%H:%M:%S.%f") + tz
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)  # Standard string notation
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, relativedelta):
        return format_relativedelta(obj)
    return json.JSONEncoder.default(self, obj)
# Fixed Function
def default(self, obj):
    if isinstance(obj, datetime.datetime):
        tz = obj.strftime("%z")
        tz = tz if tz else '+0000'
        return obj.strftime("%Y-%m-%dT%H:%M:%S.%f") + tz
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)  # Standard string notation
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, relativedelta):
        return format_relativedelta(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    return json.JSONEncoder.default(self, obj)
"""


code = {'luigi-4': {}, 'pandas-145': {}, 'salt/salt-50958': {}}

code['luigi-4']['prefix'] = '''
# Provide a fix for the buggy function
# Buggy Function
def copy(self, cursor, f):
    logger.info("Inserting file: %s", f)
    colnames = ''
    if len(self.columns) > 0:
        colnames = ",".join([x[0] for x in self.columns])
        colnames = '({})'.format(colnames)
    cursor.execute("""
        COPY {table} {colnames} from '{source}'
        CREDENTIALS '{creds}'
        {options}
        ;""".format(
        table=self.table,
        colnames=colnames,
        source=f,
        creds=self._credentials(),
        options=self.copy_options)
    )
# Fixed Function
def copy(self, cursor, f):
    logger.info("Inserting file: %s", f)
    colnames = ''
'''
code['luigi-4']['suffix'] = '''
        colnames = ",".join([x[0] for x in self.columns])
        colnames = '({})'.format(colnames)
    cursor.execute("""
        COPY {table} {colnames} from '{source}'
        CREDENTIALS '{creds}'
        {options}
        ;""".format(
        table=self.table,
        colnames=colnames,
        source=f,
        creds=self._credentials(),
        options=self.copy_options)
    )'''


code['luigi-4']['correct'] = '''
def copy(self, cursor, f):
    logger.info("Inserting file: %s", f)
    colnames = ''
    if self.columns and len(self.columns) > 0:
        colnames = ",".join([x[0] for x in self.columns])
        colnames = '({})'.format(colnames)
    cursor.execute("""
        COPY {table} {colnames} from '{source}'
        CREDENTIALS '{creds}'
        {options}
        ;""".format(
        table=self.table,
        colnames=colnames,
        source=f,
        creds=self._credentials(),
        options=self.copy_options)
    )
'''

code['pandas-145']['prefix'] = '''
# Provide a fix for the buggy function
# Buggy Function
def dispatch_to_series(left, right, func, str_rep=None, axis=None):
    import pandas.core.computation.expressions as expressions
    right = lib.item_from_zerodim(right)
    if lib.is_scalar(right) or np.ndim(right) == 0:
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    elif isinstance(right, ABCDataFrame):
        assert right._indexed_same(left)
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b.iloc[:, i]) for i in range(len(a.columns))}
    elif isinstance(right, ABCSeries) and axis == "columns":
        assert right.index.equals(left.columns)
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b.iloc[i]) for i in range(len(a.columns))}
    elif isinstance(right, ABCSeries):
        assert right.index.equals(left.index)  # Handle other cases later
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    else:
        raise NotImplementedError(right)
    new_data = expressions.evaluate(column_op, str_rep, left, right)
    return new_data
# Fixed Function
def dispatch_to_series(left, right, func, str_rep=None, axis=None):
    import pandas.core.computation.expressions as expressions
    right = lib.item_from_zerodim(right)
    if lib.is_scalar(right) or np.ndim(right) == 0:
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    elif isinstance(right, ABCDataFrame):
        assert right._indexed_same(left)
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b.iloc[:, i]) for i in range(len(a.columns))}
    elif isinstance(right, ABCSeries) and axis == "columns":
        assert right.index.equals(left.columns)
'''

code['pandas-145']['suffix'] = '''
    elif isinstance(right, ABCSeries):
        assert right.index.equals(left.index)  # Handle other cases later
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    else:
        raise NotImplementedError(right)
    new_data = expressions.evaluate(column_op, str_rep, left, right)
    return new_data'''

code['pandas-145']['correct'] = '''
def dispatch_to_series(left, right, func, str_rep=None, axis=None):
    import pandas.core.computation.expressions as expressions
    right = lib.item_from_zerodim(right)
    if lib.is_scalar(right) or np.ndim(right) == 0:
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    elif isinstance(right, ABCDataFrame):
        assert right._indexed_same(left)
        def column_op(a, b):
            return {i: func(a.iloc[:, i], b.iloc[:, i]) for i in range(len(a.columns))}
    elif isinstance(right, ABCSeries) and axis == "columns":
        assert right.index.equals(left.columns)
        if right.dtype == "timedelta64[ns]":
            right = np.asarray(right)
            def column_op(a, b):
                return {i: func(a.iloc[:, i], b[i]) for i in range(len(a.columns))}
        else:

            def column_op(a, b):
                return {i: func(a.iloc[:, i], b.iloc[i]) for i in range(len(a.columns))}
    elif isinstance(right, ABCSeries):
        assert right.index.equals(left.index)  # Handle other cases later

        def column_op(a, b):
            return {i: func(a.iloc[:, i], b) for i in range(len(a.columns))}
    else:
        raise NotImplementedError(right)
    new_data = expressions.evaluate(column_op, str_rep, left, right)
    return new_data
'''

code['salt/salt-50958']['prefix'] = '''
# Provide a fix for the buggy function
# Buggy Function
def func():
    HAS_LIBS = False
    try:
        import twilio
        if twilio.__version__ > 5:
            TWILIO_5 = False
            from twilio.rest import Client as TwilioRestClient
            from twilio.rest import TwilioException as TwilioRestException
        else:
            TWILIO_5 = True
            from twilio.rest import TwilioRestClient
            from twilio import TwilioRestException
        HAS_LIBS = True
    except ImportError:
        pass
# Fixed Function
def func():
    HAS_LIBS = False
    try:
        import twilio
'''

code['salt/salt-50958']['suffix'] = '''
            TWILIO_5 = False
            from twilio.rest import Client as TwilioRestClient
            from twilio.rest import TwilioException as TwilioRestException
        else:
            TWILIO_5 = True
            from twilio.rest import TwilioRestClient
            from twilio import TwilioRestException
        HAS_LIBS = True
    except ImportError:
        pass'''

code['salt/salt-50958']['correct'] = '''
HAS_LIBS = False
try:
    import twilio
    twilio_version = tuple([int(x) for x in twilio.__version_info__])
    if twilio_version > (5, ):
        TWILIO_5 = False
        from twilio.rest import Client as TwilioRestClient
        from twilio.rest import TwilioException as TwilioRestException
    else:
        TWILIO_5 = True
        from twilio.rest import TwilioRestClient
        from twilio import TwilioRestException
    HAS_LIBS = True
except ImportError:
    pass
'''

def remove_comment(text):
    pattern = re.compile(
    r"""\s*\#(?:[^\n])*| "{3}(?:\\.|[^\\])*"{3}| '{3}(?:\\.|[^\\])*'{3} | ^\s*\n""",
    re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    text = re.sub(pattern, '', text)
    return re.sub(pattern, '', text)

def remove_empty_line(text):
    pattern = re.compile(
    r"""^\s*\n""",
    re.VERBOSE | re.MULTILINE | re.DOTALL
    )
    return re.sub(pattern, '', text)


def handle_identation(lines, num = None):
    if num == None:
        num = 0
        newlines = []
        for s in lines[0]:
            if s == ' ':
                num += 1
            else:
                break
        if num == 0:
            return lines, num
        else:
            for l in lines:
                newlines.append(l[num:])

            return newlines, num
    else:
        newlines = []
        for l in lines:
            newlines.append(l[num:])
        return newlines



@retry(wait=wait_exponential(min=10, max=60), stop=stop_after_attempt(7))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def build_prompt(buggy_source, buggy_lines, added):
    root = ast.parse(buggy_source)
    lines = buggy_source.splitlines()
    locator = FunctionLocator()
    node = locator.run(root, buggy_lines)
    if node.end_lineno - node.lineno > 100:
        logger.warning('Too Long Funciton.')
    add = False
    for a in added:
        if a == True:
            add = True
            break
        elif a == 2:
            add = 2
            break
    buggy, num = handle_identation(lines[node.lineno - 1: node.end_lineno])
    if not add:
        min_lineno = min(buggy_lines) - 1
        max_lineno = max(buggy_lines) - 1
        prefix = lines[node.lineno - 1: min_lineno]
        suffix = lines[max_lineno + 1: node.end_lineno]
    elif add == True:
        prefix = lines[node.lineno - 1: buggy_lines[0] - 1]
        suffix = lines[buggy_lines[0] - 1: node.end_lineno]
    elif add == 2:
        prefix = lines[node.lineno - 1: buggy_lines[0]]
        suffix = lines[buggy_lines[0]: node.end_lineno]
    
    prefix = handle_identation(prefix, num = num)
    suffix = handle_identation(suffix, num = num)

    prefix = remove_comment("\n".join(prefix))
    if len(suffix) == 0:
        suffix = None
    else:
        suffix = remove_comment("\n".join(suffix))
    buggy = remove_comment("\n".join(buggy))

    return prefix, suffix, buggy

def build_prompt_for_test(buggy_source, buggy_lines, added):
    root = ast.parse(buggy_source)
    lines = buggy_source.splitlines()
    add = False
    for a in added:
        if a == True:
            add = True
            break
        elif a == 2:
            add = 2
            break
    if not add:
        min_lineno = min(buggy_lines) - 1
        max_lineno = max(buggy_lines) - 1
        prefix = lines[: min_lineno]
        suffix = lines[max_lineno + 1:]
    elif add == True:
        prefix = lines[: buggy_lines[0] - 1]
        suffix = lines[buggy_lines[0] - 1:]
    elif add == 2:
        prefix = lines[: buggy_lines[0]]
        suffix = lines[buggy_lines[0]:]

    return prefix, suffix


def get_pred(buggy_file, buggy_lines, added, repo):
    buggy_source = open(buggy_file, "r").read().replace('\t', '    ')
    if repo in code:
        prompt = code[repo]['prefix']
        suffix = code[repo]['suffix']
        prompt = remove_empty_line(prompt)
        suffix = remove_empty_line(suffix)
    else:
        prefix, suffix, buggy = build_prompt(buggy_source, buggy_lines, added)
        prompt = '# Provide a fix for the buggy function\n# Buggy Function\n' + buggy + '\n# Fixed Function\n' + prefix
        if prompt[-1] != '\n':
            prompt += '\n'
        prompt = remove_empty_line(prompt)
    
    prompt = example_1 + example_2 + example_3 + prompt
    prompt = remove_empty_line(prompt)

    data = {}
    no_change_count = 0
    top_p = 0.95
    temperature = 0.8
    try:
        if len(buggy_lines) == 1 and added[0] == False:
            for i in tqdm(range(0, 200)):
                if suffix != None:
                    completion = completion_with_backoff(engine="code-davinci-002", prompt=prompt, max_tokens = 128, n = 25, top_p = top_p, temperature = temperature, suffix = suffix, stop = ['# Provide a fix for the buggy function', '# Buggy Function', '# Fixed Function'])
                else:
                    completion = completion_with_backoff(engine="code-davinci-002", prompt=prompt, max_tokens = 128, n = 25, top_p = top_p, temperature = temperature, stop = ['# Provide a fix for the buggy function', '# Buggy Function', '# Fixed Function'])
                num = 0
                for c in completion.choices:
                    if c["text"] not in data:
                        data[c["text"]] = c
                        num += 1
                if num == 0:
                    no_change_count += 1
                if num > 0:
                    no_change_count = 0
                if no_change_count > 40:
                    break
                logger.debug('Usage: {}'.format(completion.usage))
                logger.debug(f'Current predictions: {len(data)}')
                time.sleep(4)
        else:
            for i in tqdm(range(0, 200)):
                if suffix != None:
                    completion = completion_with_backoff(engine="code-davinci-002", prompt=prompt, max_tokens = 256, n = 15, top_p = top_p, temperature = temperature, suffix = suffix, stop = ['# Provide a fix for the buggy function', '# Buggy Function', '# Fixed Function'])
                else:
                    completion = completion_with_backoff(engine="code-davinci-002", prompt=prompt, max_tokens = 256, n = 15, top_p = top_p, temperature = temperature, stop = ['# Provide a fix for the buggy function', '# Buggy Function', '# Fixed Function'])
                num = 0
                for c in completion.choices:
                    if c["text"] not in data:
                        data[c["text"]] = c
                        num += 1
                if num == 0:
                    no_change_count += 1
                if num > 0:
                    no_change_count = 0
                if no_change_count > 40:
                    break
                logger.debug('Usage: {}'.format(completion.usage))
                logger.debug(f'Current predictions: {len(data)}')
                time.sleep(4)
    except Exception as e:
        logger.error('Error occurred: {}'.format(e))
        return data, True

    return data, False


def get_preds(benchmark_info, benchmark_path, patch_path, benchmark = "bugsinpy"):
    data = []
    failed_cases = []
    metadata = json.loads(open(benchmark_info, "r").read())
    if benchmark == "bugsinpy":
        for r in tqdm(metadata):
            for i in metadata[r]:
                for f in metadata[r][i]["code_files"]:
                        #if f'{r}-{i}' != 'luigi-22':
                        #    continue
                        if not f.endswith(".py"):
                            continue
                        try:
                            prefix = f.replace(".py", "-")
                            buggy_files = []
                            for bf in metadata[r][i]["buglines"]:
                                if bf.startswith(prefix):
                                    buggy_files.append(bf)
                            if len(buggy_files) == 0:
                                buggy_files.append(f)
                            for bf in buggy_files:
                                logger.debug('Handling File#{} in Case#{}'.format(bf, f'{r}-{i}'))
                                final_patch_file_path = os.path.join(patch_path, r, f'{r}-{i}', bf.replace('/', '_').replace('.py', '.json'))
                                if os.path.exists(final_patch_file_path):
                                    continue
                                buggy_file = os.path.join(benchmark_path, r, f'{r}-{i}', bf)
                                try:
                                    patch_data, error = get_pred(buggy_file, metadata[r][i]["buglines"][bf], metadata[r][i]["added"][bf], f'{r}-{i}')
                                    if not os.path.exists(os.path.join(patch_path, r, f'{r}-{i}')):
                                        os.system('mkdir -p {}'.format(os.path.join(patch_path, r, f'{r}-{i}')))
                                    with open(final_patch_file_path, 'w', encoding = 'utf-8') as pf:
                                        pf.write(json.dumps(patch_data, sort_keys=True, indent=4, separators=(',', ': ')))
                                    if error:
                                        failed_cases.append([f'{r}/{r}-{i}', f, bf])
                                except Exception as e:
                                    traceback.print_exc()
                                    logger.error(f'Error occurred: {e}')
                                    failed_cases.append([f'{r}/{r}-{i}', f, bf, f"{e}"])
                        except Exception as e:
                            traceback.print_exc()
                            logger.error(f'Error occurred: {e}')
                            failed_cases.append([f'{r}/{r}-{i}', f, f"{e}"])
    elif benchmark == 'typebugs':
        failed_cases = []
        for r in tqdm(metadata):
            for f in metadata[r]["code_files"]:
                if not f.endswith(".py"):
                    continue
                try:
                    prefix = f.replace(".py", "-")
                    buggy_files = []
                    for bf in metadata[r]["buglines"]:
                        if bf.startswith(prefix):
                            buggy_files.append(bf)
                    if len(buggy_files) == 0:
                        buggy_files.append(f)
                    for bf in buggy_files:
                        logger.debug('Handling File#{} in Case#{}'.format(bf, r))
                        final_patch_file_path = os.path.join(patch_path, r, bf.replace('/', '_').replace('.py', '.json'))
                        if os.path.exists(final_patch_file_path):
                            continue
                        buggy_file = os.path.join(benchmark_path, r, bf)
                        try:
                            patch_data, error = get_pred(buggy_file, metadata[r]["buglines"][bf], metadata[r]["added"][bf], r)
                            if not os.path.exists(os.path.join(patch_path, r)):
                                os.system('mkdir -p {}'.format(os.path.join(patch_path, r)))
                            with open(final_patch_file_path, 'w', encoding = 'utf-8') as pf:
                                pf.write(json.dumps(patch_data, sort_keys=True, indent=4, separators=(',', ': ')))
                            if error:
                                failed_cases.append([r, f, bf])
                        except Exception as e:
                            traceback.print_exc()
                            logger.error(f'Error occurred: {e}')
                            failed_cases.append([r, f, bf, f"{e}"])
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f'Error occurred: {e}')
                    failed_cases.append([r, f, f"{e}"])
    
    with open(os.path.join(patch_path, "failed_cases.json"), "w", encoding = "utf-8") as ff:
        ff.write(json.dumps(failed_cases, sort_keys=True, indent=4, separators=(',', ': ')))


def evaluate_correctness(benchmark_info, benchmark_path, patch_path, benchmark = "bugsinpy"):
    metadata = json.loads(open(benchmark_info, "r").read())
    if benchmark == "bugsinpy":
        num = 0
        correct_num = 0
        succeed_cases = []
        failed_cases = []
        for r in tqdm(metadata):
            for i in metadata[r]:
                path = os.path.join(benchmark_path, r, f'{r}-{i}')
                for f in metadata[r][i]["code_files"]:
                        #if f'{r}-{i}' != 'luigi-22':
                        #    continue
                        if not f.endswith(".py"):
                            continue
                        prefix = f.replace(".py", "-")
                        buggy_files = []
                        for bf in metadata[r][i]["buglines"]:
                            if bf.startswith(prefix):
                                buggy_files.append(bf)
                        if len(buggy_files) == 0:
                            buggy_files.append(f)
                        for bf in buggy_files:
                            locator = FunctionLocator()
                            if r not in code:
                                locator = FunctionLocator()
                                correct = ast.parse(open(os.path.join(path, f'correct/{f}')).read())
                                node = locator.run(correct, metadata[r][i]["buglines"][bf])
                                remover = CommentRemover()
                                correct_node = remover.run(node)
                            else:
                                node = ast.parse(code[r]['correct'])
                                if len(node.body) == 1:
                                    node = node.body[0]
                                remover = CommentRemover()
                                correct_node = remover.run(node)
                            patch_file_path = os.path.join(patch_path, r, f'{r}-{i}', bf.replace('/', '_').replace('.py', '.json'))
                            if not os.path.exists(patch_file_path):
                                continue
                            num += 1
                            patch = json.loads(open(patch_file_path, "r").read())
                            success = False
                            buggy_file = os.path.join(benchmark_path, r, f'{r}-{i}', bf)
                            buggy_source = open(buggy_file, 'r').read()
                            if f'{r}-{i}' in code:
                                prompt = code[f'{r}-{i}']['prefix']
                                prompt = prompt.split('# Fixed Function')[-1]
                                suffix = code[f'{r}-{i}']['suffix']
                                prompt = remove_empty_line(prompt)
                                suffix = remove_empty_line(suffix)
                            else:
                                prefix, suffix, buggy = build_prompt(buggy_source, metadata[r][i]["buglines"][bf], metadata[r][i]["added"][bf])
                                prompt = prefix
                                if len(prompt) > 1 and prompt[-1] != '\n':
                                    prompt += '\n'
                                prompt = remove_empty_line(prompt)
                            for p in patch:
                                patch_code = prompt + patch[p]["text"]
                                if suffix != None:
                                    patch_code += suffix
                                try:
                                    patch_root = ast.parse(patch_code)
                                except Exception as e:
                                    logger.debug('Cannot parse patch code, reason:{}'.format(e))
                                if ASTDiffer.compare(patch_root.body[0] if len(patch_root.body) == 1 else patch_root, correct_node):
                                    correct_num += 1
                                    success = True
                                    succeed_cases.append([r, i, bf, p, patch[p]['index']])
                                    break
                            if not success:
                                failed_cases.append([f'{r}-{i}', bf])
        with open(os.path.join(patch_path, "correctness_succeed_cases.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(succeed_cases, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(os.path.join(patch_path, "correctness_failed_cases.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(failed_cases, sort_keys=True, indent=4, separators=(',', ': ')))
        logger.info("Totally {} instances, correctly generate patches for {} instances, correct fix rate: {}.".format(num, correct_num, correct_num/num))
    elif benchmark == 'typebugs':
        num = 0
        correct_num = 0
        succeed_cases = []
        failed_cases = []
        for r in tqdm(metadata):
            path = os.path.join(benchmark_path, r)
            #if r != 'core/core-29829':
            #    continue
            for f in metadata[r]["code_files"]:
                if not f.endswith(".py"):
                    continue
                prefix = f.replace(".py", "-")
                buggy_files = []
                for bf in metadata[r]["buglines"]:
                    if bf.startswith(prefix):
                        buggy_files.append(bf)
                if len(buggy_files) == 0:
                    buggy_files.append(f)
                for bf in buggy_files:
                    #if not bf.endswith('126.py'):
                    #    continue
                    if r not in code:
                        locator = FunctionLocator()
                        correct = ast.parse(open(os.path.join(path, f'correct/{f}')).read())
                        node = locator.run(correct, metadata[r]["buglines"][bf])
                        remover = CommentRemover()
                        correct_node = remover.run(node)
                    else:
                        node = ast.parse(code[r]['correct'])
                        if len(node.body) == 1:
                            node = node.body[0]
                        remover = CommentRemover()
                        correct_node = remover.run(node)
                    patch_file_path = os.path.join(patch_path, r, bf.replace('/', '_').replace('.py', '.json'))
                    if not os.path.exists(patch_file_path):
                        continue
                    num += 1
                    patch = json.loads(open(patch_file_path, "r").read())
                    success = False
                    buggy_file = os.path.join(benchmark_path, r, bf)
                    buggy_source = open(buggy_file, 'r').read()
                    if r in code:
                        prompt = code[r]['prefix']
                        prompt = prompt.split('# Fixed Function')[-1]
                        suffix = code[r]['suffix']
                        prompt = remove_empty_line(prompt)
                        suffix = remove_empty_line(suffix)
                    else:
                        prefix, suffix, buggy = build_prompt(buggy_source, metadata[r]["buglines"][bf], metadata[r]["added"][bf])
                        prompt = prefix
                        if len(prompt) > 1 and prompt[-1] != '\n':
                            prompt += '\n'
                        prompt = remove_empty_line(prompt)
                    for i, p in enumerate(patch):
                        patch_code = prompt + patch[p]["text"]
                        if suffix != None:
                            patch_code += suffix
                        #if patch[p]["index"] != 0:
                        #    continue
                        try:
                            patch_root = ast.parse(patch_code)
                        except Exception as e:
                            logger.debug('Cannot parse patch code, reason:{}'.format(e))
                        if ASTDiffer.compare(patch_root.body[0] if len(patch_root.body) == 1 else patch_root, correct_node):
                            correct_num += 1
                            success = True
                            succeed_cases.append([r, bf, p, i])
                            break
                    if not success:
                        failed_cases.append([r, bf])
        with open(os.path.join(patch_path, "correctness_succeed_cases.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(succeed_cases, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(os.path.join(patch_path, "correctness_failed_cases.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(failed_cases, sort_keys=True, indent=4, separators=(',', ': ')))
        logger.info("Totally {} instances, correctly generate patches for {} instances, correct fix rate: {}.".format(num, correct_num, correct_num/num))



def prepare_test(benchmark_info, benchmark_path, patch_path, benchmark = "bugsinpy"):
    metadata = json.loads(open(benchmark_info, "r").read())
    if benchmark == "bugsinpy":
        for r in tqdm(metadata):
            for i in metadata[r]:
                path = os.path.join(benchmark_path, r, f'{r}-{i}')
                for f in metadata[r][i]["code_files"]:
                    if not f.endswith(".py"):
                        continue
                    prefix = f.replace(".py", "-")
                    buggy_files = []
                    for bf in metadata[r][i]["buglines"]:
                        if bf.startswith(prefix):
                            buggy_files.append(bf)
                    if len(buggy_files) == 0:
                        buggy_files.append(f)
                    for bf in buggy_files:
                        buggy_file = os.path.join(benchmark_path, r, f'{r}-{i}', bf)
                        buggy_source = open(buggy_file, 'r').read()
                        prefix, suffix = build_prompt_for_test(buggy_source, metadata[r][i]["buglines"][bf], metadata[r][i]["added"][bf])
                        patch_file_path = os.path.join(patch_path, r, f'{r}-{i}', bf.replace('/', '_').replace('.py', '.json'))
                        if not os.path.exists(patch_file_path):
                            continue
                        num += 1
                        patch = json.loads(open(patch_file_path, "r").read())
                        new_patches = []
                        for i, p in enumerate(patch):
                            code = "\n".join(prefix) + "\n" + patch[p]["text"] + "\n" + "\n".join(suffix)
                            try:
                                ast.parse(code)
                                new_patches.append(patch[p]["text"])
                            except:
                                pass
                        new_patch = {'prefix': "\n".join(prefix), "suffix": "\n".join(suffix), "patches": new_patches}
                        with open(patch_file_path, "w", encoding = "utf-8") as pf:
                            pf.write(json.dumps(new_patch, sort_keys=True, indent=4, separators=(',', ': ')))
    elif benchmark == "typebugs":
        for r in tqdm(metadata):
            path = os.path.join(benchmark_path, r)
            for f in metadata[r]["code_files"]:
                if not f.endswith(".py"):
                    continue
                prefix = f.replace(".py", "-")
                buggy_files = []
                for bf in metadata[r]["buglines"]:
                    if bf.startswith(prefix):
                        buggy_files.append(bf)
                if len(buggy_files) == 0:
                    buggy_files.append(f)
                for bf in buggy_files:
                    if not f.endswith(".py"):
                        continue
                    prefix = f.replace(".py", "-")
                    buggy_files = []
                    for bf in metadata[r]["buglines"]:
                        if bf.startswith(prefix):
                            buggy_files.append(bf)
                    if len(buggy_files) == 0:
                        buggy_files.append(f)
                    for bf in buggy_files:
                        buggy_file = os.path.join(benchmark_path, r, bf)
                        buggy_source = open(buggy_file, 'r').read()
                        prefix, suffix = build_prompt_for_test(buggy_source, metadata[r]["buglines"][bf], metadata[r]["added"][bf])
                        patch_file_path = os.path.join(patch_path, r, bf.replace('/', '_').replace('.py', '.json'))
                        if not os.path.exists(patch_file_path):
                            continue
                        num += 1
                        patch = json.loads(open(patch_file_path, "r").read())
                        new_patches = []
                        for i, p in enumerate(patch):
                            code = "\n".join(prefix) + "\n" + patch[p]["text"] + "\n" + "\n".join(suffix)
                            try:
                                ast.parse(code)
                                new_patches.append(patch[p]["text"])
                            except:
                                pass
                        new_patch = {'prefix': "\n".join(prefix), "suffix": "\n".join(suffix), "patches": new_patches}
                        with open(patch_file_path, "w", encoding = "utf-8") as pf:
                            pf.write(json.dumps(new_patch, sort_keys=True, indent=4, separators=(',', ': ')))









if __name__ == '__main__':
    #get_preds('TypeErrorFix/benchmarks/all_bug_info_bugsinpy.json', 'TypeErrorFix/benchmarks/bugsinpy', 'codex_patches/bugsinpy')
    #get_preds('TypeErrorFix/benchmarks/all_bug_info_typebugs.json', 'TypeErrorFix/benchmarks/typebugs', 'codex_patches/typebugs', benchmark = 'typebugs')
    evaluate_correctness('TypeErrorFix/benchmarks/all_bug_info_bugsinpy.json', 'TypeErrorFix/benchmarks/bugsinpy', 'codex_patches/bugsinpy')
    #evaluate_correctness('TypeErrorFix/benchmarks/all_bug_info_typebugs.json', 'TypeErrorFix/benchmarks/typebugs', 'codex_patches/typebugs', benchmark = 'typebugs')
