"""A conversational debugger and drop-in replacement for pdb. Python's default
interactive debugging session is already a crude conversation with your
program or interpreter, in a sense - this just lets your program communicate to
you more effectively.

Quickstart
----------
Here's a broken version of bubble sort that places a `duck()` call on the
second to last line where you might normally call `breakpoint()`.

```
from roboduck import duck

def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                nums[j + 1] = nums[j]
                nums[j] = nums[j + 1]
                duck()   # <--------------------------- instead of breakpoint()
    return nums

nums = [3, 1, 9, 2, 1]
bubble_sort(nums)
```
"""
import cmd
from functools import partial
import inspect
import ipynbname
from langchain.callbacks.base import CallbackManager
from pdb import Pdb
import sys
import uuid
import warnings

from roboduck.langchain.chat import Chat
from roboduck.langchain.callbacks import LiveTypingCallbackHandler
from roboduck.utils import type_annotated_dict_str, colored, truncated_repr, \
    colordiff_new_str, \
    parse_completion
from roboduck.decorators import store_class_defaults, add_docstring
from roboduck.ipy_utils import load_ipynb, load_current_ipython_session, \
    is_ipy_name


@store_class_defaults(attr_filter=lambda x: x.startswith('last_'))
class CodeCompletionCache:
    """Stores values related to the last completion from DuckDB in a way that
    a. our `duck` jupyter magic can access, and
    b. allows us to easily reset all defaults

    The magic only needs to access it in Paste mode (-p flag) to insert
    the fixed code snippet into a new code cell.
    """

    # LLM full completion.
    last_completion = []
    # LLM natural language explanation.
    last_explanation = []
    # User code snippet.
    last_code = []
    # LLM generated code.
    last_new_code = []
    # LLM generated code with new bits highlighted (print to render correctly).
    last_code_diff = []
    # Technically tied to the DuckDB instance, but in roboduck a new instance
    # is created for each session. We only store one value here rather than a
    # list because it will be the same for every turn in the session.
    last_session_id = ''
    # List of dicts. Allows storing extra values when passing a
    # custom parse_func to DuckDB.
    last_extra = []

    @classmethod
    def update_cache(cls, **kwargs):
        for k, v in kwargs.items():
            cache = getattr(cls, k)
            if isinstance(cache, list):
                cache.append(v)
            elif isinstance(cache, str):
                setattr(cls, k, v)
            else:
                raise ValueError(
                    'Roboduck encountered an unexpected value in '
                    f'CodeCompletionCache: attr {k} has type {type(cache)},'
                    f'expected str or list.'
                )

    @classmethod
    def get(cls, name, newest=True):
        """Get the first/last truthy value of a given attribute if one exists,
        e.g. the most recent code completion.

        Parameters
        ----------
        name : str
            Attribute name. Should be a class attribute like 'last_code'.
        newest : bool
            Determines whether to get the first (newest=False) or last
            (newest=True) truthy value.

        Returns
        -------
        any
            The first/last truthy value of the requested class attribute. If
            no truthy values are found, we return None.

        Examples
        --------
        ```
        # Get the most recent LLM response.
        CodeCompletionCache.get('last_completion')
        ```
        """
        if not hasattr(cls, name):
            raise AttributeError(f'{cls.__name__} has no attribute {name}.')

        items = getattr(cls, name)
        if isinstance(items, (list, tuple)):
            if newest:
                items = items[::-1]
            for val in items:
                if val:
                    return val
        return items or None


class DuckDB(Pdb):
    """Conversational debugger powered by LLM (e.g. gpt-3.5-turbo or gpt-4).
    Once you're in a debugging session, regular pdb commands will work as usual
    but any user command containing a question mark will be interpreted as a
    question for the lLM. Prefixing your question with "[dev]" will print out
    the full prompt before making the query (mostly useful when working on the
    library).
    """

    def __init__(self, prompt_name='debug', max_len_per_var=79, silent=False,
                 pdb_kwargs=None, parse_func=parse_completion, color='green',
                 **chat_kwargs):
        """
        Parameters
        ----------
        prompt_name : str
            Name of prompt template to use when querying chatGPT. Roboduck
            currently provides several builtin options
            (see roboduck.prompts.chat):
                debug - for interactive debugging sessions on the relevant
                    snippet of code.
                debug_full - for interactive debugging sessions on the whole
                    notebook (no difference from "debug" for scripts). Risks
                    creating a context that is too long.
                debug_stack_trace - for automatic error explanations or
                    logging.
            Alternatively, can also define your own template in a yaml file
            mimicking the format of the builtin templates and pass in the
            path to that file as a string.
        max_len_per_var : int
            Limits number of characters per variable when communicating
            current state (local or global depending on `full_context`) to
            gpt. If unbounded, that section of the prompt alone could grow
            very big . I somewhat arbitrarily set 79 as the default, i.e.
            1 line of python per variable. I figure that's usually enough to
            communicate the gist of what's happening.
        silent : bool
            If True, print gpt completions to stdout. One example of when False
            is appropriate is our logging module - we want to get the
            explanation and update the exception message which then gets
            logged, but we don't care about typing results in real time.
        pdb_kwargs : dict or None
            Additional kwargs for base Pdb class.
        parse_func : function
            This will be called on the generated text each time gpt provides a
            completion. It returns a dictionary whose values will be stored
            in CodeCompletionCache in this module. See the default function's
            docstring for guidance on writing a custom function.
        color : str
            Color to print gpt completions in. Sometimes we want to change this
            to red, such as in the errors module, to make it clearer that an
            error occurred.
        chat_kwargs : any
            Additional kwargs to configure our Chat class (passed to
            its `from_template` factory). Common example would be setting
            `chat_class=roboduck.langchain.chat.DummyChatModel`
            which mocks api calls (good for development, saves money).
        """
        super().__init__(**pdb_kwargs or {})
        # These are prompts in the pdb sense, not the LLM sense. I.e. they
        # are shown at the start of the line, right before the place where the
        # user or the LLM will begin typing.
        self.prompt = '>>> '
        self.duck_prompt = '[Duck] '
        self.query_kwargs = {}
        chat_kwargs['name'] = prompt_name
        if silent:
            chat_kwargs['streaming'] = False
        else:
            chat_kwargs['streaming'] = True
            chat_kwargs['callback_manager'] = CallbackManager(
                [LiveTypingCallbackHandler(color=color)]
            )
        # Dev color is what we print the prompt in when user asks a question
        # in dev mode.
        self.color = color
        self.dev_color = 'blue' if self.color == 'red' else 'red'
        # Must create self.chat before setting _chat_prompt_keys,
        # and full_context after both of those.
        self.chat = Chat.from_template(**chat_kwargs)
        self.default_user_key, self.backup_user_key = self._chat_prompt_keys()
        self.full_context = 'full_code' in self.field_names()
        self.prompt_name = prompt_name
        self.repr_func = partial(truncated_repr, max_len=max_len_per_var)
        self.silent = silent
        self.parse_func = parse_func
        # This gets updated every time the user asks a question.
        self.prev_kwargs_hash = None
        # This can generally be treated as a session ID since roboduck always
        # creates a new debugger object for each session. It lets us know when
        # to clear the CodeCompletionCache.
        self.uuid = str(uuid.uuid1())

    def _chat_prompt_keys(self):
        """Retrieve default and backup user reply prompt keys (names) from
        self.chat object. If the prompt template has only one reply type,
        the backup key will equal the default key.
        """
        keys = list(self.chat.user_templates)
        default = keys[0]
        backup = default
        if len(keys) > 1:
            backup = keys[1]
            if len(keys) > 2:
                warnings.warn(
                    'You\'re using a chat prompt template with >2 types or '
                    'user replies. This is not recommended because it\'s '
                    'not clear how to determine which reply type to use. We '
                    'arbitrarily choose the first non-default key as the '
                    f'backup reply type ("{backup}").'
                )
        return default, backup

    def error(self, line):
        """Add a hint when displaying errors that roboduck only responds to
        *questions* in natural language.

        Parameters
        ----------
        line : str
        """
        super().error(line)
        if any(term in line for term in ('SyntaxError', 'NameError')):
            print(
                '*** If you meant to respond to Duck in natural language, '
                'remember that it only provides English responses to '
                'questions. Statements are evaluated as Pdb commands.',
                file=self.stdout
            )

    def field_names(self, key=''):
        """Get names of variables that are expected to be passed into default
        user prompt template.

        Parameters
        ----------
        key : str
            Determines which user prompt type to use. By default, roboduck
            provides "contextful" (which will include the source code, variable
            values, and the stack trace when appropriate) and "contextless"
            (which includes only the user question). We default to
            "contextful" here.

        Returns
        -------
        set[str]
        """
        return self.chat.input_variables(key)

    def _get_next_line(self, code_snippet):
        """Retrieve next line of code that will be executed. Must call this
        before we remove the duck() call. We use this in `_get_prompt_kwargs`
        during interactive debugging sessions.

        Parameters
        ----------
        code_snippet : str
        """
        lines = code_snippet.splitlines()
        max_idx = len(lines) - 1

        # Adjust f_lineno because it's 1 - indexed by default.
        # Set default next_line in case we don't find any valid line.
        line_no = self.curframe.f_lineno - 1
        next_line = ''
        while line_no <= max_idx:
            if lines[line_no].strip().startswith('duck('):
                line_no += 1
            else:
                next_line = lines[line_no]
                break
        return next_line

    def _get_prompt_kwargs(self):
        """Construct a dictionary describing the current state of our code
        (variable names and values, source code, file type). This will be
        passed to our langchain chat.reply() method to fill in the debug prompt
        template.

        Returns
        -------
        dict
            contains keys 'code', 'local_vars', 'global_vars', 'file_type'.
            If we specified full_context=True on init, we also include the key
            'full_code'.
        """
        res = {}

        # Get current code snippet.
        # Fails when running code from cmd line like:
        # 'python -c "print(x)"'.
        # Haven't been able to find a way around this yet.
        try:
            # Find next line before removing duck call to avoid messing up our
            # index.
            code_snippet = inspect.getsource(self.curframe)
            res['next_line'] = self._get_next_line(code_snippet)
            res['code'] = self._remove_debugger_call(code_snippet)
        except OSError as err:
            self.error(err)

        # Get full source code if necessary.
        if self.full_context:
            # File is a string, either a file name or something like
            # <ipython-input-50-e97ed612f523>.
            file = inspect.getsourcefile(self.curframe.f_code)
            if file.startswith('<ipython'):
                # If we're in ipython, ipynbname.path() throws a
                # FileNotFoundError.
                try:
                    full_code = load_ipynb(ipynbname.path())
                    res['file_type'] = 'jupyter notebook'
                except FileNotFoundError:
                    full_code = load_current_ipython_session()
                    res['file_type'] = 'ipython session'
            else:
                with open(file, 'r') as f:
                    full_code = f.read()
                res['file_type'] = 'python script'
            res['full_code'] = self._remove_debugger_call(full_code)
            used_tokens = set(res['full_code'].split())
        else:
            # This is intentionally different from the used_tokens line in the
            # if clause - we only want to consider local code here.
            used_tokens = set(res['code'].split())

        # Namespace is often polluted with lots of unused globals (htools is
        # very much guilty of this 😬) and we don't want to clutter up the
        # prompt with these.
        res['local_vars'] = type_annotated_dict_str(
            {k: v for k, v in self.curframe_locals.items()
             if k in used_tokens and not is_ipy_name(k)},
            self.repr_func
        )
        res['global_vars'] = type_annotated_dict_str(
            {k: v for k, v in self.curframe.f_globals.items()
             if k in used_tokens and not is_ipy_name(k)},
            self.repr_func
        )
        return res

    @staticmethod
    def _remove_debugger_call(code_str):
        """Remove `duck` function call (our equivalent of `breakpoint` from
        source code string. Including it introduces a slight risk that gpt
        will fixate on this mistery function as a potential bug cause.

        Parameters
        ----------
        code_str : str
            Source code snippet. We want to remove the `duck()` call (which
            sometimes includes kwargs) to prevent this from distracting the
            LLM.

        Returns
        -------
        str
        """
        return '\n'.join(line for line in code_str.splitlines()
                         if not line.strip().startswith('duck('))

    def onecmd(self, line):
        """Base class describes this as follows:

        Interpret the argument as though it had been typed in response to the
        prompt. Checks whether this line is typed at the normal prompt or in
        a breakpoint command list definition.

        We add an extra check in the if block to check if the user asked a
        question. If so, we ask gpt. If not, we treat it as a regular pdb
        command.

        Parameters
        ----------
        line : str or tuple
            If str, this is a regular line like in the standard debugger.
            If tuple, this contains (line str, stack trace str - see
            roboduck.errors.post_mortem for the actual insertion into the
            cmdqueue). This is for use with the debug_stack_trace mode.
        """
        if isinstance(line, tuple):
            line, stack_trace = line
        else:
            stack_trace = ''
        if not self.commands_defining:
            if '?' in line:
                return self.ask_language_model(
                    line,
                    stack_trace=stack_trace,
                    verbose=line.startswith('[dev]')
                )
            return cmd.Cmd.onecmd(self, line)
        else:
            return self.handle_command_def(line)

    def ask_language_model(self, question, stack_trace='', verbose=False):
        """When the user asks a question during a debugging session, query
        gpt for the answer and type it back to them live.

        Parameters
        ----------
        question : str
            User question, e.g. "Why are the first three values in nums equal
            to 5 when the input list only had a single 5?". (Example is from
            a faulty bubble sort implementation.)
        stack_trace : str
            When using the "debug_stack_trace" prompt, we need to pass a
            stack trace string into the prompt.
        verbose : bool
            If True, print the full gpt prompt in red before making the api
            call. User activates this mode by prefixing their question with
            '[dev]'. This overrides self.silent.
        """
        # Don't provide long context-laden prompt if nothing has changed since
        # the user's last question. This is often a followup/clarifying
        # question.
        prompt_kwargs = self._get_prompt_kwargs()
        kwargs_hash = hash(str(prompt_kwargs))
        if kwargs_hash == self.prev_kwargs_hash:
            prompt_kwargs.clear()
            prompt_key = self.backup_user_key
        else:
            prompt_key = self.default_user_key

        # Perform surgery on kwargs depending on what fields are expected.
        field_names = self.field_names(prompt_key)
        if 'question' in field_names:
            prompt_kwargs['question'] = question
        if stack_trace:
            prompt_kwargs['stack_trace'] = stack_trace

        # Validate that expected fields are present and provide interpretable
        # error message if not.
        kwargs_names = set(prompt_kwargs)
        only_in_kwargs = kwargs_names - field_names
        only_in_expected = field_names - kwargs_names
        error_msg = 'If you are using a custom prompt, you may need to ' \
                    'subclass roboduck.debug.DuckDB and override the ' \
                    '_get_prompt_kwargs method.'
        if only_in_kwargs:
            raise RuntimeError(
                f'Received unexpected kwarg(s): {only_in_kwargs}. {error_msg} '
            )
        if only_in_expected:
            raise RuntimeError(
                f'Missing required kwarg(s): {only_in_expected}. {error_msg}'
            )

        prompt = self.chat.user_message(key_=prompt_key,
                                        **prompt_kwargs).content
        if verbose:
            print(colored(prompt, 'red'))

        if not self.silent:
            print(colored(self.duck_prompt, self.color), end='')

        # The actual LLM call.
        res = self.chat.reply(**prompt_kwargs, key_=prompt_key)

        answer = res.content.strip()
        if not answer:
            answer = 'Sorry, I don\'t know. Can you try ' \
                     'rephrasing your question?'
            # This is intentionally nested in if statement because if answer is
            # truthy, we will have already printed it via our callback if not
            # in silent mode.
            if not self.silent:
                print(colored(answer, self.color))

        parsed_kwargs = self.parse_func(answer)
        if CodeCompletionCache.last_session_id != self.uuid:
            CodeCompletionCache.reset_class_vars()

        # Built-in prompts always ask for a fixed version of the relevant
        # snippet, not the whole code, so that's what we store here and use for
        # the diff operation. Contextless prompt has no `code` key, hence the
        # `get` usage.
        old_code = prompt_kwargs.get('code', '')
        new_code = parsed_kwargs['code']
        CodeCompletionCache.update_cache(
            last_completion=answer,
            last_explanation=parsed_kwargs['explanation'],
            last_code_diff=colordiff_new_str(old_code, new_code),
            last_code=old_code,
            last_new_code=new_code,
            last_extra=parsed_kwargs.get('extra', {}),
            last_session_id=self.uuid
        )
        self.prev_kwargs_hash = kwargs_hash

    def precmd(self, line):
        """We need to define this to make our errors module work. Our
        post_mortem function sometimes places a tuple in our debugger's
        cmdqueue and precmd is called as part of the default cmdloop method.
        Technically it calls postcmd too but we don't need to override that
        because it does nothing with its line argument.

        Parameters
        ----------
        line : str or tuple
            If a tuple, it means roboduck.errors.excepthook is being called
            and an error has occurred. The stack trace is passed in as the
            second of two items, where the first item is the same object that
            is normally passed in.
        """
        if isinstance(line, tuple):
            line, trace = line
            return super().precmd(line), trace
        return super().precmd(line)

    def print_stack_entry(self, frame_lineno, prompt_prefix='\n-> '):
        """This is called automatically when entering a debugger session
        and it prints a message to stdout like

        ```
        > <ipython-input-20-9c67d40d0f93>(2)<module>()
        -> print + 6
        ```

        In silent mode (like when using the roboduck logger with stdout=False),
        we want to disable that message. When silent=False, this behaves
        identically to the standard pdb equivalent.
        """
        if self.silent:
            return
        frame, lineno = frame_lineno
        if frame is self.curframe:
            prefix = '> '
        else:
            prefix = '  '
        self.message(prefix +
                     self.format_stack_entry(frame_lineno, prompt_prefix))


@add_docstring(DuckDB.__init__)
def duck(**kwargs):
    """Roboduck equivalent of native python breakpoint().
    The DuckDB docstring is below. Any kwargs passed in to this function
    will be passed to its constructor.
    """
    DuckDB(**kwargs).set_trace(sys._getframe().f_back)
