import dateparser
from langchain.text_splitter import RecursiveCharacterTextSplitter


MAX_LENGTH = 1000
doc_spliter = RecursiveCharacterTextSplitter(chunk_size=MAX_LENGTH, chunk_overlap=100)
# doc_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=MAX_LENGTH, chunk_overlap=100)


class Event:
    def __init__(self, timestamp, title, content):
        self.timestamp = dateparser.parse(timestamp)
        self.title = title
        if self.title is None:
            self.title = ''
        self.content = content
        if self.content is None:
            self.content = ''
        self.abstract = None

    def add_abstract(self, parser, len_abstract: int = 50):
        content = f'标题：{self.title}\n内容：{self.content}'
        if len(content) < len_abstract:
            self.abstract = content
        elif len(content) < MAX_LENGTH:
            self.abstract = parser.generate_abstract(content)
        else:
            content_docs = doc_spliter.create_documents([content])
            sub_abstracts = list()
            for sub_content in content_docs:
                abstract = parser.generate_abstract(sub_content.page_content)
                sub_abstracts.append(abstract)
            self.abstract = parser.generate_abstract('\n'.join(sub_abstracts))

    def __repr__(self):
        return f'{self.timestamp} {"" if self.title is None else self.title}'


class MedicalRecord:
    def __init__(self, id):
        self.id = id
        self.records = list()
        self.docs = None

    def add_record(self, parser, timestamp, title, content):
        event = Event(timestamp, title, content)
        event.add_abstract(parser)
        self.records.append(event)
        self.records = sorted(self.records, key=lambda x: x.timestamp)

    def create_docs(self):
        # For semantic searching
        event_doc = '\n\n'.join([f'时间：{e.timestamp}\n标题：{e.title}\n内容：\n{e.content}' for e in self.records])
        self.docs = doc_spliter.create_documents([event_doc])

    def __repr__(self):
        return f'{self.id} [{self.records[0].timestamp.date()} ~ {self.records[-1].timestamp.date()}]'
