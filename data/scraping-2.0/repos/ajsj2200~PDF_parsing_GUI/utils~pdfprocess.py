import fitz
import re
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
import tempfile
import os
import streamlit as st


def split_text_at_period(text, max_chars, buffer=10):
    """Split the text based on the maximum character limit and specific patterns."""
    parts = []
    figure_table_pattern = re.compile(r'(Figure \d+:|Table \d+:)')
    number_parenthesis_pattern = re.compile(r'\d+\),')

    # Remove (cid:숫자) pattern from the text
    text = re.sub(r'\(cid:\d+\)', '', text)

    text = text.replace('\n', ' ')
    text = text.replace('•', '\n\n•')
    text = re.sub(r'\([^)]*\)', '', text)

    while text:
        if len(text) <= max_chars:
            parts.append(text)
            break

        figure_table_match = figure_table_pattern.search(text)

        if figure_table_match and figure_table_match.start() < max_chars:
            if figure_table_match.start() > 0:  # Ensure we're not at the very start
                parts.append(
                    text[:figure_table_match.start()].strip() + '\n\n')
                text = text[figure_table_match.start():]
                continue  # Move to next iteration to process the Figure/Table part
            else:
                part = text[:figure_table_match.end() + 1] + '\n\n'
                text = text[figure_table_match.end() + 1:]
        else:
            split_point = text.rfind('.', 0, max_chars)

            if split_point != -1:
                # Check within a buffer ahead for the number pattern.
                if number_parenthesis_pattern.search(text[split_point:split_point+buffer]):
                    split_point = text.rfind('.', 0, split_point)

                part = text[:split_point + 1] + '\n\n'
                text = text[split_point + 2:]
            else:
                part = text[:max_chars]
                text = text[max_chars:]

        parts.append(part)

    return parts


@st.cache_data
def load_pdf_content(file_buffer, max_chars=1000):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file_path = temp_file.name

    # Write the content of the PDF to the temporary file
    with open(temp_file_path, 'wb') as f:
        f.write(file_buffer.read())

    # Use the temporary file path with the PDF loader
    loader = PDFMinerPDFasHTMLLoader(temp_file_path)
    data = loader.load()[0]
    soup = BeautifulSoup(data.page_content, 'html.parser')

    content = soup.find_all('div')

    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px', st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text, cur_fs))

    cur_idx = -1
    semantic_snippets = []

    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
            metadata = {'heading': s[0],
                        'content_font': 0, 'heading_font': s[1]}
            metadata.update(data.metadata)
            semantic_snippets.append(
                Document(page_content='', metadata=metadata))
            cur_idx += 1
            continue

        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_font'] = max(
                s[1], semantic_snippets[cur_idx].metadata['content_font'])
            continue

        # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
        # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
        metadata = {'heading': s[0],
                    'content_font': 0, 'heading_font': s[1]}
        metadata.update(data.metadata)
        semantic_snippets.append(
            Document(page_content='', metadata=metadata))
        cur_idx += 1

    # 목차 추출
    doc = fitz.open(temp_file_path)
    outline = [x[1] for x in doc.get_toc()]

    # 중복된 목차가 있다면 _1, _2, ...를 붙여준다.
    for i in range(len(outline)):
        for j in range(i+1, len(outline)):
            if outline[i] == outline[j]:
                outline[j] += f'_{j-i}'

    MAX_CHARS = max_chars  # or any number you choose
    mds = ''
    h = None
    h_count = 0

    for snippet in semantic_snippets:
        if snippet.metadata['heading'] != h:
            h_count = 0
            h = snippet.metadata['heading']
            mds += f'## {h}\n\n'

        content_parts = split_text_at_period(
            snippet.page_content, MAX_CHARS)
        for part in content_parts:
            mds += part + '\n'
            h_count += len(part)

    # 'Figure 숫자):' 패턴을 찾아서 '\nFigure 숫자):'로 바꿔준다.
    mds = re.sub(r'Figure (\d+)\):', '\n\nFigure \\1):', mds)

    # 'Table 숫자):' 패턴을 찾아서 '\nTable 숫자):'로 바꿔준다.
    mds = re.sub(r'Table (\d+)\):', '\n\nTable \\1):', mds)

    # '숫자. 문자' 패턴을 찾아서 '\n숫자. 문자'로 바꿔준다.
    # mds = re.sub(r'\d+\.+\D', '\n\n', mds)

    mds = mds.replace('•', '\n\n•')
    cleaned_outline = [re.sub(r'^[\d\.]+\s+', '', item)
                       for item in outline]

    temp_file.close()
    os.remove(temp_file_path)

    return mds, cleaned_outline


def process_pdf(file_buffer, max_chars=1000):
    return load_pdf_content(file_buffer, max_chars)
