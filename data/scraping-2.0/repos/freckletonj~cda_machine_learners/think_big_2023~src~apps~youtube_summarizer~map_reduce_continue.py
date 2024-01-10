import tiktoken
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from apps import our_openai as openai


MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=
    """I want to you to act as a note-taking assistant. Create a well formatted summary of the following transcript.
    
    
    "{text}"
    
    
    Write a short summary here:
    """
)

CONTINUE_PROMPT = PromptTemplate(
    input_variables=["text", "last_response"],
    template=
    """I want to you to act as a note-taking assistant. Create a well formatted summary of the following transcript.
    
    
    "{text}"
    
    
    Continue your short summary here:
    {last_response}"""
)
CHAPTER_MAP_PROMPT = PromptTemplate(
    input_variables=["text", "chapter"],
    template=
    """I want to you to act as a note-taking assistant. Create a well formatted summary of the following transcript.
    
    
    "{text}"
    
    
    The current topic is '{chapter}'. Write your short summary here:
    """
)

CHAPTER_CONTINUE_PROMPT = PromptTemplate(
    input_variables=["text", "chapter", "last_response"],
    template=
    """I want to you to act as a note-taking assistant. Create a well formatted summary of the following transcript.
    
    
    "{text}"
    
    
    The current topic is '{chapter}'. Continue your short summary here:
    {last_response}"""
)
encoder = tiktoken.get_encoding("p50k_base")


class MapReduceContinue:
    def __init__(self, debug=False):
        self.debug = debug
        self.encoding = "p50k_base"
        self.encoder = encoder
        self.ai_model = 'text-curie-001'
        self.PROMPT_TOKENS = len(encoder.encode(CHAPTER_CONTINUE_PROMPT.template))
        self.TOKEN_LIMIT = 2000
        self.RESPONSE_TOKENS = 450
        self.OVERLAP_TOKENS = 124
        self.CONTINUE_TOKENS = self.OVERLAP_TOKENS // 2
        self.MAX_TOKENS = self.TOKEN_LIMIT - self.PROMPT_TOKENS - self.OVERLAP_TOKENS - self.RESPONSE_TOKENS - self.CONTINUE_TOKENS

        self._text_splitter = CharacterTextSplitter(
            length_function=lambda text: len(encoder.encode(text)),
            chunk_overlap=self.OVERLAP_TOKENS,
            chunk_size=self.MAX_TOKENS + self.OVERLAP_TOKENS,
        )

        self.responses = []

    def _call_api(self, prompt):
        return openai.Completion.create(
            model=self.ai_model,
            prompt=prompt,
            temperature=0.6,
            max_tokens=self.RESPONSE_TOKENS,
            top_p=0.4,
            frequency_penalty=0.5,
            presence_penalty=0
        )

    def _split_text(self, text):
        # Calculate the encoded length of the input text
        encoded_length = len(self.encoder.encode(text))

        # If the encoded length is less than or equal to MAX_TOKENS, no need to split
        if encoded_length <= self.MAX_TOKENS:
            return [text]

        # Calculate the number of chunks needed
        num_chunks = encoded_length // self.MAX_TOKENS + 1

        # Find approximate size of each chunk
        chunk_size = len(text) // num_chunks

        chunks = []
        start = 0

        # Split the text into chunks at whitespace positions
        for _ in range(num_chunks - 1):
            end = start + chunk_size

            # Move the end index to the nearest whitespace
            while end < len(text) and text[end] in [' ', '\n', '\t']:
                end += 1

            # Add the chunk to the list of chunks
            chunks.append(text[start:end])

            # Update the start index for the next iteration
            start = end - self.OVERLAP_TOKENS // 3

        # Add the last chunk
        chunks.append(text[start:])

        return chunks

    def _map_reduce(self, text, initial_prompt, continue_prompt, chapter=None):
        text_chunks = self._split_text(text)
        if self.debug:
            print(f"Split text into {len(text_chunks)} chunks")
        responses = []
        for i, chunk in enumerate(text_chunks):
            kwargs = {'text': chunk}
            if chapter is not None:
                kwargs['chapter'] = chapter
            if i == 0:
                prompt = initial_prompt.format(**kwargs)
            else:
                kwargs['last_response'] = responses[-1][self.CONTINUE_TOKENS * -4:]
                prompt = continue_prompt.format(**kwargs)
            if self.debug:
                print(f"Prompt {i}:\n{prompt}\n")
            response = self._call_api(prompt)
            if self.debug:
                print(f"Response {i}:\n{response}\n")
            responses.append(response.choices[0].text)
            self.responses.append({
                'text': chunk,
                'response': response.choices[0].text,
                'prompt': prompt,
            })
            if response.choices[0].finish_reason == 'length':
                print(f"Response {i} was stopped because of length")

        return ''.join(responses).strip()

    def summarize(self, text, chapter=None):
        if chapter is None:
            return self._map_reduce(text, MAP_PROMPT, CONTINUE_PROMPT)
        else:
            return self._map_reduce(text, CHAPTER_MAP_PROMPT, CHAPTER_CONTINUE_PROMPT, chapter=chapter)
