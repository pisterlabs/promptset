import logging
import re
import time
import openai
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger()

MODEL_ENGINE = "gpt-3.5-turbo-0301"
MAX_TOKENS = 3000

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create file handler for DEBUG level
file_handler = logging.FileHandler('app.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)-12s %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.addHandler(TqdmLoggingHandler(level=logging.WARNING))

logger.info("spg logging started")

class GptSrtTranslator():

    API_KEY = None
    MODEL_ENGINE = None

    skip_square_brackets = True
    # [Cheering]
    skip_all_caps = True
    # LAUGHING

    ignore_asterisks = True
    # * There is a house in New Orleans *
    ignore_note_sings = True
    # ♪ There is a house in New Orleans ♪

    # Use the following characters to detect non-english all caps strings
    all_caps_regex = r"^[A-ZÁÉÍÓÖŐÚÜŰ,.!?\- ]{3,}$"

    def __init__(self, **kwargs) -> None:
        '''
        Initializes a SubtitleTranslator object with the specified parameters.

        Args:
            **kwargs: Keyword arguments for the SubtitleTranslator object. Optional arguments include:
                - api_key: A string representing the OpenAI API key. Defaults to the class attribute API_KEY.
                - slice_length: An integer representing the number of lines sent for translation in one step. Defaults to 10.
                - relax_time: How many seconds to wait between chatgpt request. Defaults to 1.
                - max_tokens: max number of tokens to use in a single go. Defaults to the class attribute MAX_TOKENS.
                - model_engine: which openai model language to use. Defaults to the class attribute MODEL_ENGINE.
                - input_language: language of the original subtitle. Defaults to "english".
                - output_language: language of the target subtitle. Defaults to "hungarian".
                - subtitle_line_max_length: add a line break if a subtitle line is longer than max . Defaults to 50.
                - input_file: Source of translation. Defaults to an empty string.
                - output_file: Target of translation. Defaults to "output.srt".

        Returns:
            None.
        '''
        openai.api_key = kwargs.get("api_key", self.API_KEY)

        self.srt = {}
        self.srt_index = {}
        self.srt_index = {}

        self.slice_length = kwargs.get("slice_length", 10)
        self.relax_time = kwargs.get("relax_time", 0.5)
        self.max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        self.model_engine = kwargs.get("model_engine", MODEL_ENGINE)

        self.input_language = kwargs.get("input_language", "english")
        self.output_language = kwargs.get("output_language", "hungarian")

        self.subtitle_line_max_length = kwargs.get("subtitle_line_max_length", 40)

        self.input_file = kwargs.get("input_file", "")
        self.output_file = kwargs.get("output_file", "output.srt")

        logger.info("Starting translation")
        logger.info("Input srt file: %s", self.input_file)
        logger.info("Output srt file: %s", self.output_file)

        if self.input_file:
            self.load_srt()

        if logger.isEnabledFor(logging.DEBUG):
            with open('01-original.txt', mode='w', encoding="utf8") as file:
                file.write("")
            with open('02-translated.txt', mode='w', encoding="utf8") as file:
                file.write("")

    def load_srt(self) -> None:
        self.log("Loading srt")
        with open(self.input_file, 'r', encoding="utf8") as f:
            srt_text = f.read()
            # Split the text at every integer which is followed by a timestamp
            if srt_text == '\n\n\n':
                parts = '\n\n\n'
            else:
                parts = re.split(r'(\d+)\n(\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d)', srt_text)
            if len(parts) <= 1:
                logger.error("Empty srt file: %s", self.input_file)
                return False

            # Remove any empty parts
            parts = [part for part in parts if part.strip(' ')]

            # Remove BOM
            if parts[0] == '\ufeff':
                del parts[0]

            # Load srt into an object
            self.srt = {}
            index = 1
            for i in range(0,len(parts), 3):
                try:
                    timestamp = parts[i+1].strip()
                    if parts[i+2].strip() == '':
                        original = '...'
                    else : 
                        original = parts[i+2].strip()
                except:
                    continue
                try:

                    # skip all caps subtitles
                    if self.skip_all_caps:
                        match = re.match(self.all_caps_regex, original.strip())
                        if match:
                            logger.debug("Skipping all caps: %s", original.strip())
                            continue

                    # skip subtitles in square brackets
                    if self.skip_square_brackets:
                        if original.strip().startswith("[") and original.strip().endswith("]"):
                            logger.debug("Skipping lines in square brackets: %s", original.strip())
                            continue

                    # skip parts in sqauer brackets
                    if self.skip_square_brackets and "[" in original:
                        logger.debug("Skipping text in square brackets: %s", original.strip())
                        original = re.sub(r'\[.*?\]', '', original)  # remove square brackets and text inside them
                        original = re.sub(r'\s+', ' ', original)  # remove duplicate spaces

                    if original == '...':
                        original == ''
                    self.srt[index] = {
                        "index": index,
                        "timestamp": timestamp,
                        "original": original.replace('"', '').strip(",").strip("."),
                        "translated": ""
                    }
                    time_index = self.srt[index]["timestamp"].split(" --> ")[0]
                    self.srt_index[time_index] = index
                except KeyError:
                    logger.error("Index not found in SRT: %s", index)

                index += 1

        self.log(f"Loaded {len(self.srt)} subtitles")

    def get_translatable_text(self, start:int, buffer:int=5) -> str:
        # create a simplified text structure so chatgpt will be able process it
        total = ""
        index = start

        while True:
            if index > len(self.srt):
                break

            # Skip musical parts indicated with: *
            if self.ignore_asterisks:
                if self.srt[index]["original"].strip().startswith("*"):
                    index = index + 1
                    continue
            # Skip musical parts indicated with: ♪
            if self.ignore_note_sings:
                if "♪" in self.srt[index]["original"]:
                    index = index + 1
                    continue

            clean_subtitle = self.srt[index]['original'].replace('\n', ' ') + "\n"
            total += f"[{self.srt[index]['timestamp']}] {clean_subtitle}"

            if index > start + self.slice_length:
                # wait for an end of a sentence
                if total.strip()[-1] in ".?!:\"\'":
                    # end of last line is an end of a sentence
                    break

                if index+1 in self.srt and  self.srt[index+1]['original'][0].isupper():
                    # next line starts with a capital letter, this line is probably and end of a sentence
                    break

                if index  > start + self.slice_length + buffer:
                    # checked forward until the max, stop adding extra lines
                    break

            index = index + 1
            if index not in self.srt or index > len(self.srt):
                break

        # remove all html tags
        soup = BeautifulSoup(total, "html.parser")
        return (
            index + 1,          # Next index to translate
            soup.get_text()     # Translateable text
        )

    def break_subtitle_line(self, text):
        """Breaks a subtitle line into two lines if it is longer than the specified maximum length."""
        if len(text) <= self.subtitle_line_max_length:
            return text

        mid = len(text) // 2
        left = text[:mid]
        right = text[mid:]
        last_space = left.rfind(' ')
        first_space = right.find(' ')

        if last_space == -1 and first_space == -1:
            return text

        if mid-last_space < first_space:
            new_text = left[:last_space].strip() + '\n' + (left[last_space+1:] + right).strip()
        else:
            new_text = (left + right[:first_space]).strip() + "\n" + right[first_space+1:].strip()

        return new_text

    def save_translated_text(self, text):
        # process text received from chatgpt
        for line in text.split('\n'):
            # skip empty lines
            if len(line) == 0:
                continue

            pattern = r"\[(.*) -->.*\] (.*)"
            match = re.search(pattern, line)

            if match:
                timestamp = match.group(1)
                translated_subtitle = match.group(2)

                # break dialogs into two lines
                if translated_subtitle.startswith("-") and translated_subtitle[2:-2].find("-") > 0:
                    second_hyphen = translated_subtitle.find("-", translated_subtitle.find("-") + 1)
                    new_text = translated_subtitle[:second_hyphen] + "\n-" + translated_subtitle[second_hyphen+1:]
                    translated_subtitle = new_text
                else:
                    # break long text into two lines
                    translated_subtitle = self.break_subtitle_line(translated_subtitle)

                if timestamp in self.srt_index:
                    subtitle_index = self.srt_index[timestamp]
                    self.srt[subtitle_index]["translated"] = translated_subtitle
                else:
                    logger.warning("Timestamp was not found when saving translated text: %s", timestamp)

    def translate(self):
        # translate the subtitle, show a progress bar during translation
        # create title for progress bas, find episode number in string
        match = re.search(r's\d+e\d+', self.input_file, re.IGNORECASE)
        if match:
            title = match.group()
        else:
            title = "video"

        self.log("Starting translation")

        index = 1
        progress_subtitle = tqdm(total=len(self.srt), bar_format='{l_bar}{bar:40}{r_bar}', desc=title.ljust(10))

        while index < len(self.srt):
            logger.info("Slice: %d, %d pieces, total %d", index, self.slice_length, len(self.srt))

            index, text_to_translate = self.get_translatable_text(index)

            if len(text_to_translate) < 10:
                # No more lines
                progress_subtitle.update(progress_subtitle.total-progress_subtitle.n)
                break

            if logger.isEnabledFor(logging.DEBUG):
                with open('01-original.txt', mode='a', encoding="utf8") as file:
                    new_string = ''
                    for line in text_to_translate.split('\n'):
                        if ']' in line:
                            idx = line.index(']') + 1
                            line = line[:idx].strip() + '\n' + line[idx:].strip()
                        new_string += line + '\n'

                    file.write(new_string.strip()+"\n")
                    file.write("-"*40 + "\n")

            translated_text = None
            error_found = None

            relax_delay = 10

            while (error_found or translated_text is None) and relax_delay <= 640:

                translated_text = self.chat_gpt_translate(text_to_translate)
                error_found = False
                if translated_text is None:
                    logger.error("Error during translation, wating for %d sec to overcome rate limitation...", relax_delay)
                    error_found = True
                elif translated_text.count("\n") < 2:
                    # too few lines were returned
                    logger.error("Short string was returned, wating for %d sec to overcome rate limitation...", relax_delay)
                    error_found = True
                
                if relax_delay >= 80:
                    break

                if error_found:
                    time.sleep(relax_delay)
                    logger.error("Trying again...")
                    relax_delay += relax_delay
                else:
                    self.save_translated_text(translated_text)

                    if logger.isEnabledFor(logging.DEBUG):
                        with open('02-translated.txt', mode='a', encoding="utf8") as file:
                            new_string = ''
                            for line in translated_text.split('\n'):
                                if ']' in line:
                                    idx = line.index(']') + 1
                                    line = line[:idx].strip() + '\n' + line[idx:].strip()
                                new_string += line + '\n'

                            file.write(new_string.strip()+"\n")
                            file.write("-"*40 + "\n")


            progress_subtitle.update(index-progress_subtitle.n)

            self.save_srt()

        self.log("Translation completed")
        progress_subtitle.close()

    def save_srt(self):
        srt_content = ""

        for index, subtitle in self.srt.items():
            srt_content += f"{index}\n"
            srt_content += f"{subtitle['timestamp']}\n"
            srt_content += f"{subtitle['translated']}\n"
            srt_content += "\n"

        with open(self.output_file, 'w', encoding="utf8") as file:
            file.write(srt_content)

    def chat_gpt_translate(self, text) -> str:
        original_line_count = text.strip().count('\n')+1
        self.log(f"Sent {original_line_count} lines")

        prompt='''You are a program responsible for translating subtitles.
Your task is to output the specified target language based on the input text.
Please do not create the following subtitles on your own.
Please do not output any text other than the translation.
You will receive the subtitles as lines of text to be translated.
Please always keep the timestamp at the beginning of the lines intact and
always put the translated text into the line matching the original timestamp.
If you need to merge the subtitles with the following line, simply repeat the translation.
Be concise.\n'''
        prompt += f"Original language: {self.input_language}\n"
        prompt += f"Target language: {self.output_language}\n"
        prompt += f"{text}"

        logger.debug("Sent %d lines for translation", original_line_count)
        logger.debug("Prompt:\n\n%s\n", prompt)

        # Generate a response
        try:
            completion = openai.ChatCompletion.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model_engine,
                max_tokens=self.max_tokens,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=100
            )
        except Exception as e:
            logger.error("Unsuccesful OpenAI operation, see debug log")
            logger.debug("Unsuccesful OpenAI operation. Error: %s", e)
            return None

        response = completion.choices[0]["message"]["content"].strip()
        response_line_count = response.count('\n')+1
        logger.debug("Returned %d lines", response.count('\n')+1)
        logger.debug("Translation:\n\n%s\n\n", response)
        if response_line_count < original_line_count:
            logger.warning("Missing %d line(s)", original_line_count - response_line_count)
        elif response_line_count > original_line_count:
            logger.info("Extra %d line(s)", response_line_count - original_line_count)

        time.sleep(self.relax_time)

        return response

    def log(self, message):
        tqdm.write(message)
