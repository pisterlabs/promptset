from __future__ import annotations
import itertools
from typing import Dict, Tuple, List
from openai.types.chat import ChatCompletionMessageParam

class Dialogue(Tuple[Dict[str,str]]):
    """A dialogue among multiple speakers, represented as an imutable tuple of
    dialogue turns. Each turn is a dict with 'speaker' and 'content' keys. The
    speaker values are just names like "teacher" and "student", or "Alice" and
    "Bob".
    
    See `agents.py` for classes that will extend the Dialogue using an LLM.
    """
    
    def __repr__(self) -> str:
        # Invoked by the repr() function, and also by print() and str() functions since we haven't defined __str__.
        return '\n'.join([f"({turn['speaker']}) {turn['content']}" for turn in self])
    
    def __rich__(self) -> str:
        # Like __str__, but invoked by rich.print().
        return '\n'.join([f"[white on blue]({turn['speaker']})[/white on blue] {turn['content']}" for turn in self])

    def __format__(self, specification: str) -> str:
        # Like __str__, but invoked by f"..." strings and the format() function.
        # We will ignore the specification argument.
        return self.__rich__()
    
    def script(self) -> str:
        """Return a single string that formats this dialogue like a play script,
        suitable for inclusion in an LLM prompt."""
        return '"""\n' + '\n\n'.join([f"{turn['speaker']}: {turn['content']}" for turn in self]) + '\n"""'

    def add(self, speaker: str, content: str) -> Dialogue:
        """Non-destructively append a given new turn to the dialogue."""
        return Dialogue(itertools.chain(self, ({'speaker': speaker, 'content': content},)))
    
    def rename(self, old: str, new: str) -> Dialogue:
        """Non-destructively rename a speaker in a dialogue."""
        d = Dialogue()
        for turn in self:
            d = d.add(new if turn['speaker']==old else turn['speaker'], turn['content'])
        return d
    
    # Support +,  *, and [] operators to concatenate and slice Dialogues.
    # This could be useful when constructing your own argubots.
    
    def __add__(self, other):
        if not isinstance(other, Dialogue):
            raise ValueError(f"Can only concatenate Dialogues with Dialogues, but got {type(other)}")
        return Dialogue(super().__add__(other))
    
    def __mul__(self, other):
        return Dialogue(super().__mul__(other))
    
    def __rmul__(self, other):
        return Dialogue(super().__rmul__(other))
    
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return Dialogue(result)
        else:
            return result


def format_as_messages(d: Dialogue, speaker: str, *, 
                      system: str | None = None, 
                      system_last: str | None = None,
                      speaker_names: bool | None = None, 
                      compress: bool | None = None, 
                      tool: str | None = None, 
                      tool_name: str | None = None
                      ) -> List[ChatCompletionMessageParam]:
    """Convert the given Dialogue into a sequence of messages that can be sent
    to OpenAI's chat completion API to ask the LLM to generate a new turn
    from the given speaker. 
    
    Each message, and the message returned by the API, is a dict with 'role'
    and 'content' keys, much like the turns in the Dialogue.
    
    We will pretend to the LLM that it generated all of the previous turns
    from `speaker` and now has to generate a new one.  OpenAI only recognizes
    a few speaker roles, not the speaker names.  So the messages that we
    create will use the 'assistant' role for all turns from `speaker`
    (because the LLM always generates in the 'assistant' role), and will use
    'user' for all other turns.
    
    But what if the dialogue has more than two speakers?  Then the 'user' and
    'assistant' roles are not enough to distinguish them.  In that case, we
    will indicate _within the message content_ who is speaking.  Also, for
    fear of confusing the LLM, we will avoid having consecutive 'user' turns
    by compressing them into a single message.  These behaviors kick in by
    default if we have more than two speakers, but they can also be
    separately controlled by keyword arguments.
    
    Args:
        * speaker: the name of the person who will speak the generated text
        * system: a system message(s) to include at the start of the prompt
        * system_last: a system message to include at the end of the prompt
        * tool: the output of a tool, which we are attaching to the API call
        * tool_name: the name of that tool 
        * speaker_names: whether to mark speaker_names in the message contents
        * compress: whether to compress consecutive user turns into a single turn
    """
    
    # Figure out default arguments.
    speakers = {turn['speaker'] for turn in d}  # set of all speakers
    speakers.add(speaker)                       # include the speaker for the new turn
    if speaker_names is None: speaker_names = (len(speakers) > 2)
    if compress is None: compress = (len(speakers) > 2)
        
    # Make list of turns suitable to pass to the chat completions interface.
    openai_messages = []
    if system is not None:
        openai_messages.append({'role': 'system', 'content': system})
    for turn in d:
        openai_messages.append({'role': 'assistant' if turn['speaker'] == speaker else 'user',
                'content': f"{turn['role']}: {turn['content']}" if speaker_names else turn['content']})
    if system_last is not None:
        openai_messages.append({'role': 'system', 'content': system_last})
    if tool is not None:
        openai_messages.append({'role': 'tool', 'content': tool})
        if tool_name is not None:
            openai_messages[-1]['name'] = tool_name

    if compress:
        i = 0
        while i < len(openai_messages):
            if openai_messages[i]['role'] == 'user':
                # i is the start of a sequence of consecutive user messages; 
                # set j to be just beyond the end of that sequence
                j = i+1
                while j < len(openai_messages) and openai_messages[j]['role'] == 'user':
                    j = j+1
                # compress that sequence into a single user message
                compressed = '\n\n'.join([turn['content'] for turn in openai_messages[i:j]])
                openai_messages[i:j] = [{'role': 'user',
                                         'content': f'"""\n{compressed}\n"""'}]
            i += 1

    return openai_messages
