import argparse
import importlib
import inspect
from dataclasses import dataclass
import re
import openai

# TODO: argv input (for each? index? get length?)
# - foreach <stack name> as if each one is pushed to one arg function

# TODO: blocks; homoiconic data representation (maybe)
# - stored in lcl? then there is a default data stack (arg), and call stack is separate
# - you can pop stacks until they are empty
# - not just push but insertions
# - pass block references as well as args. if you push a block it includes its tags?
#   - perhaps do this by calling goto <fct> which doesn't touch the call stack, return_ctrl would be modified

# TODO: type system
# - everything has a prompt represtation and a program representation;
#   no need for all the if/elses in the runtime
# - should all local variables be converted to blocks? ie pop pushes to a block


# TODO: callable as python
# TODO: repeat preprocess directive
# TODO: escape primitive uses in prompt

debug = False

black = lambda text: f"\033[30m{text}\033[0m"
red = lambda text: f"\033[31m{text}\033[0m"
green = lambda text: f"\033[32m{text}\033[0m"
yellow = lambda text: f"\033[33m{text}\033[0m"
blue = lambda text: f"\033[34m{text}\033[0m"
magenta = lambda text: f"\033[35m{text}\033[0m"
cyan = lambda text: f"\033[36m{text}\033[0m"
white = lambda text: f"\033[37m{text}\033[0m"

def load_functions(module_name):
    module = importlib.import_module(module_name)
    functions = inspect.getmembers(module, inspect.isfunction)
    fct_dict = {name.replace('_', "-"): fct for name, fct in functions}
    return fct_dict


# TYPES ##########################################;

class Block:
    def __init__(self, initial=None):
        self.lines = []
        self.index = -1

        if initial:
            self.push(initial)

    def push(self, x):
        # x is a typed prompt object

        if isinstance(x, list):
            self.lines += x
            return

        if not self.index == -1:
            self.lines.insert(self.index, x)
            self.index += 1
            return

        self.lines.append(x)

    def pop(self, n=1, pop_all=False):
        # TODO: index MUST roll back to what came before popped segment
        # there will be bugs when you start moving the pointer

        if pop_all:
            if self.index == -1:
                popped = self.lines
                self.lines = []
                return popped
            popped = self.lines[:self.index]
            del self.lines[:self.index]
            popped = popped[0] if len(popped) == 1 else popped
            self.index = 0
            return popped

        if self.index == -1:
            popped = self.lines[-n:]
            del self.lines[-n:]
            popped = popped[0] if n == 1 else popped
            return popped

        popped = self.lines[self.index - n:self.index]
        popped = popped[0] if len(popped) == 1 else popped
        del self.lines[self.index - n:self.index]

        return popped

    def flip(self):
        self.index = len(self.lines) - self.index - 1
        self.lines.reverse()

    def select(self, index=-1):
       self.index = index

    def __str__(self):
        return "".join([str(line) for line in self.lines])

    def __getitem__(self, index):
        return self.lines[index]

    def __len__(self):
        return len(self.lines)


class Line:
    def __init__(self, value:str):
        self.value = value
        if value.endswith("\n"):
            self.value = value[:-1]

    def __str__(self):
        return self.value + "\n"

class Bool:
    def __init__(self, value:str):
        self.value = True if value.strip() == "True" else False

    def __str__(self):
        return str(self.value) + "\n"

    def __bool__(self):
        return self.value

class Int:
    def __init__(self, value:str):
        self.value = int(value)

    def __str__(self):
        return str(self.value) + "\n"

class Float:
    def __init__(self, value:str):
        self.value = float(value)

    def __str__(self):
        return str(self.value) + "\n"


# class Enum:
#     def __init__(self, value:str, options:list[str]):
#         self.value = value
#         self.options = options

#     def __str__(self):
#         return str(self.value) + "\n"

# TYPES ########################################;

@dataclass
class Frame:
    fct: str
    pc: int
    block: str
    # blocks: dict[str, Block]
    lcl: dict
    block_stack: list[str]

    def __str__(self):
        r = f"- {self.fct} "
        return cyan(r + (50-len(r)) * "-")

@dataclass
class ForFrame:
    for_block_name: str
    open_pc: int
    close_pc: int
    block_index: int
    iter_var: str
    iter_block: str

    def __str__(self):
        r = f"- {self.for_block_name.replace('_', ' ')} (iteration {self.block_index}) "
        return green(r + (50-len(r)) * "-")


# OPERATIONS #####################################;
# utils ####################
def parse_arg(x: str):
    # transform string into typed prompt object
    x_stripped = x.strip()

    if x_stripped.isdigit():
        value = Int(x_stripped)
    elif x_stripped.replace(".","").isdigit():
        value = Float(x_stripped)
    elif x_stripped in ["True", "False"]:
        value = Bool(x_stripped)
    # TODO: enums
    else:
        value = Line(x)

    return value

def dissect_prompt(s):
    result = []
    last_end = 0
    for match in re.finditer(r'(?<!\\)(\[.*?\])|(?<!\\)({.*?})', s):
        # Add the "prompt" type for text between matches
        if last_end != match.start():
            result.append((s[last_end:match.start()].replace("\\[", "[").replace("\\]", "]").replace("\\{", "{").replace("\\}", "}"), "prompt"))
        # Determine the type based on the first character of the match
        if match.group().startswith("["):
            result.append((match.group()[1:-1], "hole"))
        else:
            result.append((match.group()[1:-1], "variable"))
        last_end = match.end()
    # Add the "prompt" type for text after the last match
    if last_end != len(s):
        result.append((s[last_end:].replace("\\[", "[").replace("\\]", "]").replace("\\{", "{").replace("\\}", "}"), "prompt"))
    return result

def get_completion(input_string, stop=[]):

    if not stop:
        stop = ["\n"]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": input_string}
        ],
        stop=stop
    )

    return response['choices'][0]['message']['content']

# primitives ################

def push(x):
    global lcl
    global block

    x_strip = x.strip()

    # if not x_strip in lcl:
    x = parse_arg(x)
    lcl[block].push(x)
    return

    # lcl[block].push(lcl[x_strip])

def fill_prompt(prompt):
    global lcl

    segments = dissect_prompt(prompt)
    context = [str(lcl[block])]
    filled = []
    for segment, segment_type in segments:
        stop_tokens = []
        if segment_type == "hole":
            if "|" in segment:
                options = segment.split("|")
                segment = options.pop(0)
                for option in options:
                    if option.startswith("*"):
                        stop_tokens.append(option[1:])
                        continue
                    # TODO: other constraints]
                    input(repr(option))
                    raise Exception(f"Invalid hole constraint: {option}")

            completion = get_completion("".join(context + filled), stop=stop_tokens)
            filled.append(completion)
            lcl[segment] = parse_arg(completion)

        elif segment_type == "variable":
            assert segment in lcl, f"(line {pc+1}) No local variable '{segment}'"
            if isinstance(lcl[segment], Block):
                filled.append(str(lcl[segment]))
            else:
                filled.append(str(lcl[segment].value))
        else:
            filled.append(segment)

    return "".join(filled)

def pop(n_and_var):
    global lcl
    global block

    n_and_var = n_and_var.rstrip().split(" ")

    n_pop_args = len(n_and_var)
    pop_all = False

    if n_pop_args == 0:
        lcl[block].pop()
        return

    if n_pop_args == 1:
        arg = n_and_var[0]
        if arg.isdigit():
            n = int(arg)
            lcl[block].pop(n)
            return

        if arg == "*":
            lcl[block].pop(pop_all=True)
            return

        # otherwise the argument is a variable name, pop 1 from it
        n_and_var = [1] + n_and_var
        n_pop_args = 2

    if n_pop_args == 3:
        # for syntax like "pop 3 to block"
        assert n_and_var[1] == "to", f"(line {pc+1}) pop: for 3 arguments middle must be 'to' but was '{n_and_var[1]}'"
        n_and_var.pop(1)
        n_pop_args = 2

    if n_pop_args == 2:
        n, var = n_and_var

        if n == "to":
            # syntax like "pop to var"
            n = 1

        if n == "*":
            pop_all = True
            n = 1 # ignored; n is overridden when pop_all

        if not isinstance(n, int):
            assert n.isdigit(), f"(line {pc+1}) pop: {n} is not a number or 'to'"
            n = int(n)

        if var in lcl:
            # append to existing block
            if isinstance(lcl[var], Block):
                x = lcl[block].pop(n, pop_all=pop_all)
                lcl[var].push(x)
                return

            assert n == 1, f"(line {pc+1}) pop: cannot pop more than 1 to non-block variable {var}"

            # otherwise overwrite existing variable
            lcl[var] = lcl[block].pop(n, pop_all=pop_all)
            return

        # create new variable
        if n > 1:
            x = lcl[block].pop(n, pop_all=pop_all)
            lcl[var] = Block(x)
            return

        val = lcl[block].pop(pop_all=pop_all)

        if isinstance(val, list):
            lcl[var] = Block(val)
            return

        lcl[var] = val

        return

    raise Exception(f"(line {pc+1}) pop: too many arguments: {n_pop_args}")

def goto(symbol):
    global pc
    global symbols
    symbol = symbol.strip()
    pc = symbols[symbol] -1 # -1 because pc is incremented after each instruction

def if_goto(symbol):
    global pc
    global symbols

    symbol = symbol.strip()

    do_jump = lcl[block].pop()

    assert isinstance(do_jump, Bool), f"(line {pc+1}) if_goto: expected Bool but got {type(do_jump)}"

    if do_jump:
        pc = symbols[symbol] - 1


def open_block(name):
    global lcl
    global block
    global block_stack

    block_stack.append(block)
    block = name
    if not name in lcl:
        lcl[block] = Block()

    assert isinstance(lcl[block], Block), f"(line {pc+1}) block: cannot open {type(lcl[block])} as block"

def close_block(name):
    global lcl
    global block
    global block_stack

    assert block == name, f"(line {pc+1}) close block: expected </{name}> but got </{block}>"

    block = block_stack.pop()


def call(fct_and_nargs: str, nargs: int=None):

    pop_all = False

    if nargs:
        fct, nargs = fct_and_nargs.strip(), nargs
    else:
        fct_and_nargs = fct_and_nargs.split(" ")
        if len(fct_and_nargs) == 1:
            fct, nargs = fct_and_nargs[0].strip(), 0 # if nargs isn't specified assume no args
        elif len(fct_and_nargs) == 2:
            fct, nargs = fct_and_nargs
            fct = fct.strip()
            nargs = nargs.strip()
            if not nargs == "*":
                assert nargs.isdigit(), f"(line {pc+1}) call: expected number of arguments, got {nargs} in '{fct} {nargs}'"
                nargs = int(nargs)
            else:
                assert nargs == "*",  f"(line {pc+1}) call: only '*' is supported for variable number of arguments"
                nargs = 1
                pop_all = True
        else:
            raise Exception(f"(line {pc+1}) call: too many arguments to in call '{fct_and_nargs}'")


    # TODO: ability to define functions with fixed nargs
    global call_stack
    global block
    global lcl
    global block_stack

    if isinstance(nargs, str):
        nargs = nargs.strip()
        if not nargs == "*":
            assert nargs.isdigit(), f"(line {pc+1}) call: expected number of arguments, got {nargs}"
            nargs = int(nargs.strip())
        else:
            assert nargs == "*",  f"(line {pc+1}) call: only '*' is supported for variable number of arguments"
            nargs = 1
            pop_all = True

    args = lcl[block].pop(nargs, pop_all=pop_all)

    if fct in primitives:
        result = primitives[fct](args)
        lcl[block].push(result)
        return

    frame = Frame(fct, pc, block, lcl, block_stack)
    call_stack.append(frame)
    block = "arg"
    lcl = {"arg": Block(args)}
    block_stack = []
    goto(fct)

def break_loop():
    assert isinstance(call_stack[-1], ForFrame), f"(line {pc+1}) break: can only break out of loops"
    frame = call_stack[-1]
    close_for_loop()
    goto(frame.close_pc)

def return_ctrl():
    global call_stack
    global pc
    global block
    global lcl
    global block_stack

    frame = call_stack.pop()
    pc = frame.pc
    result = lcl[block] # fetch final state of called function
    lcl = frame.lcl
    block = frame.block
    if len(result) == 1:
        result = result[0]
    lcl[block].push(result)
    block_stack = frame.block_stack


# TODO: open/clear/close block in for loop
def open_for_loop(var_and_block):
    global call_stack
    global lcl
    global pc

    # make the language server happy
    for_block_name = None
    iter_var = None
    iter_block = None

    if isinstance(call_stack[-1], ForFrame) and call_stack[-1].close_pc is not None:
        call_stack[-1].block_index += 1
        lcl[call_stack[-1].for_block_name].pop(pop_all=True)
    else:
        for_block_name = "for_" + var_and_block.replace(" ", "_").rstrip()
        open_block(for_block_name)
        # lcl[block].pop(pop_all=True)
        var_and_block = var_and_block.split(" ")
        assert len(var_and_block) == 3, f"(line {pc+1}) for: expected the form 'for x in y', got {' '.join(var_and_block)}"

        iter_var, _, iter_block = var_and_block
        iter_block = iter_block.strip()

        assert isinstance(lcl[iter_block], Block), f"(line {pc+1}) iterate: expected block, got {type(lcl[iter_block])}"
        assert len(lcl[iter_block]) > 0, f"(line {pc+1}) for: block {iter_block} is empty"
        call_stack.append(ForFrame(for_block_name, pc - 1, None, 0, iter_var, iter_block))

    frame = call_stack[-1]
    for_block_name = frame.for_block_name
    block_index = frame.block_index
    iter_var = frame.iter_var
    iter_block = frame.iter_block

    if block_index == len(lcl[iter_block]):
        pc = frame.close_pc
        call_stack.pop()
        del lcl[iter_var]
        close_block(for_block_name)
        return

    lcl[iter_var] = lcl[iter_block][block_index]

def close_for_loop():
    global pc
    global call_stack
    assert isinstance(call_stack[-1], ForFrame), f"(line {pc+1}) endfor: missing for statement"

    if not call_stack[-1].close_pc:
        call_stack[-1].close_pc = pc

    # close_block(call_stack[-1].for_block_name)
    pc = call_stack[-1].open_pc


# SILAS ########################################

call_stack = []
heap = []
primitives = {} # python functions
symbols = {} # label -> line number mappings
static = {}

pc = 0
block = "arg"
lcl = {"arg": Block()}
block_stack = []

def preprocess(lines):
    global symbols
    lines = [line.lstrip() for line in lines]

    # process labels and function definitions
    i = 0
    remaining_lines = len(lines)
    # while remaining_lines > 0:
    while i < len(lines):
        remaining_lines -= 1
        if lines[i].startswith("## "):
            label = lines[i][3:].strip()
            symbols[label] = i
            # lines.pop(i)
            i += 1
            continue
        if lines[i].startswith("# "):
            name = lines[i][2:].strip()
            name = name.strip()
            # print(name)
            symbols[name] = i
            # lines.pop(i)
            i += 1
            continue

        # print(str(i) + " " + str(len(lines)))
        i += 1

    # input(f"RRETURNED ON {i}")
    return lines

def execute_line(line):
    global pc

    if line == "":
        return

    if line.startswith("> "):
        # TODO: instead use {} to insert locals anywhere in line
        line = line[2:]
        line = fill_prompt(line)
        push(line)
        return

    if line.startswith("pop"):
        line = line[3:].lstrip()
        pop(line)
        return

    if line.startswith("goto "):
        line = line[5:]
        goto(line)
        return

    if line.startswith("if-goto "):
        line = line[8:]
        if_goto(line)
        return

    if line.startswith("call "):
        line = line[5:]
        call(line)
        return

    if line.startswith("return"):
        return_ctrl()
        return

    if any(line.startswith(f"{x} ") for x in primitives):
        call(line)
        return

    if line.startswith("for "):
        line = line[4:]
        open_for_loop(line)
        return

    if line.startswith("endfor"):
        close_for_loop()
        return

    if line.startswith("</"):
        # extract until >
        line = line[2:]
        line = line.split(">")
        assert len(line) == 2, f"(line {pc+1}) Closing block statement missing '>': {line} "
        close_block(line[0])
        return

    if line.startswith("<"):
        # extract until >
        line = line[1:]
        line = line.split(">")
        assert len(line) == 2, f"(line {pc+1}) Opening block statement missing '>': {line} "
        open_block(line[0])
        return

    if line.startswith("debug"):
        print_stack()
        return

    if line.startswith("break"):
        break_loop()
        return

    if line.startswith("exit"):
        pc = -1
        return

    if line.startswith("~"):
        return

    if line.startswith("#"):
        return

    if line.startswith(" "):
        return

    raise Exception(f"(line {pc+1}) Unknown line: {line}")


def run(lines, debug=False, verbose=False):
    global pc
    global call_stack
    global primitives

    lines = preprocess(lines)

    call_stack.append(Frame("Main", -2, "arg", {"arg": Block()}, []))

    while pc >= 0:
        line = lines[pc]
        execute_line(line)
        pc += 1
        if verbose:
            print_stack()
        if debug:
            input("\n(CR to continue)")

    return lcl[block]

# CLI ########################################

def print_stack():
    if len(call_stack) == 0:
        return

    print("\033c")
    print(blue(f"= TRACE =========================================="))

    prev_frame = call_stack[0]
    for frame in call_stack[1:]:
        print(prev_frame)
        # print(frame.lcl)
        if not isinstance(frame, ForFrame):
            for line in frame.lcl[frame.block].lines:
                print(line)

        prev_frame = frame
    print(prev_frame)
    print()

    print(lcl[block])
    if isinstance(call_stack[-1], ForFrame):
        print(green("--------------------------------------------------"))

    l = len(block) + 4
    rem = 50 - l
    s = "-"*(int(rem/2)) + f" [{block}] "
    s = s + "-"*(50-len(s))
    print(cyan(s))
    print()
    # print(str(lcl))


    print(blue(f"= LOCALS ========================================="))
    for var, val in lcl.items():
        if var == block:
            continue

        val_type = str(type(val)).split("'")[1].split(".")[1]
        s = f"- {var} ({val_type}) "
        s = s + "-"*(50-len(s))
        print(cyan(s))
        if val:
            print()
            print(val, end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("filename", type=str, help="Input filename")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode: pause at each line")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode: print the prompt stack")

    clargs = parser.parse_args()

    file = clargs.filename
    with open(file, "r") as f:
        lines = f.readlines()

    debug = clargs.debug
    verbose = clargs.verbose

    if debug:
        verbose = True

    run(lines, debug=debug, verbose=verbose)

if __name__ == "__main__":
    main()


