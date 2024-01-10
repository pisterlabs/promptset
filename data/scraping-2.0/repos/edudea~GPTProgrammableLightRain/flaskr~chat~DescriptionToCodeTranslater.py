import os
import openai
from openai.error import InvalidRequestError
import re

openai.organization = "org-KkiGJrxFd5s302zC8XS3Icev"
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = ('Du hilfst einer Person ohne Programmierkenntnisse eine Lichterkette schrittweise zu programmieren.'
               ' Du erklärst den Quellcode und technische Details nicht und bleibst beim Du. Schreibe ein Programm'
               ' für Arduino IDE, dass ein Neopixel LED-Strip steuert. Die Lichterkette hat 25 LEDs und wird über'
               ' PIN 4 gesteuert; die NeoPixel-Bibliothek ist bereits installiert. Die Lichterkette ist aufgehängt,'
               ' die erste LED ist oben und die letzte LED unten. Verhindere ein langes Blockieren des loop.\n'
               ' Erkläre zuerst wie du die Anforderungen verstanden hast und gib abschließend den ganzen Quellcode aus.'
               ' Falls die Anforderungen unklar sind, sei kreativ und ergänze fehlende Informationen; erkläre in diesem'
               ' fall einfach deine getroffenen Annahmen.')

instruction_naming = (
    'Gib dem Lichteffekt einen kurzen, prägnanten Namen auf deutsch. Der Name soll nicht länger als 30 Zeichen sein.')


def send_description(lighteffect_id, conversation_id, description, socketio, conversations=None, use_large_model=True):
    messages = [{"role": "system", "content": instruction}]
    if conversations is not None:
        for conversation in conversations:
            messages.append({"role": "user", "content": conversation.description})
            messages.append({"role": "assistant", "content": conversation.answer})
    messages.append({"role": "user", "content": description})
    model = "gpt-4-1106-preview" if use_large_model else "gpt-3.5-turbo-1106"
    try:
        completion = openai.ChatCompletion.create(model=model, messages=messages, stream=True)

        answer = ''
        for chunk in completion:
            if chunk.choices[0].finish_reason == 'length':
                raise ExceedsLimitsError
            if chunk.choices[0].finish_reason != 'stop':
                if 'delta' in chunk.choices[0] and 'content' in chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    answer += content
                    socketio.emit(f"/lighteffects/{lighteffect_id}/conversations/{conversation_id}/answer/chunk",
                                  {'id': conversation_id, 'chunk': content, 'lighteffectId': lighteffect_id})

        code_pattern_search = re.compile(r".*```.{0,10}\n(?P<code>.*)```\n.*", flags=re.DOTALL).search(answer)
        code = code_pattern_search.group(
            "code") if code_pattern_search is not None and "code" in code_pattern_search.groupdict().keys() else None

        return ChatResponse(answer, code)
    except InvalidRequestError as e:
        if e.code == 'context_length_exceeded':
            raise ExceedsLimitsError
        else:
            raise TranslationError
    except:
        raise TranslationError


class TranslationError(BaseException):
    pass


class ExceedsLimitsError(TranslationError):
    pass


class ChatResponse:
    def __init__(self, answer, code):
        self.answer = answer
        self.code = code


def get_chat_name_suggestion(description, answer):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=[{"role": "system", "content": instruction},
                                                        {"role": "user", "content": description},
                                                        {"role": "assistant", "content": answer},
                                                        {"role": "user", "content": instruction_naming}])
    print(completion)
    return completion.choices[0].message.content
