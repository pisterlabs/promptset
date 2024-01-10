from pathlib import Path
from typing import List

import pandas as pd
from html2text import html2text
from langchain.text_splitter import TokenTextSplitter

from .types import Doc, Text


def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import fitz

    file = fitz.open(path)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    file.close()
    return texts


def parse_pdf(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    pdfFileObj.close()
    return texts


def parse_txt(
    path: Path, doc: Doc, chunk_chars: int, overlap: int, html: bool = False
) -> List[Text]:
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if html:
        text = html2text(text)
    # yo, no idea why but the texts are not split correctly
    text_splitter = TokenTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]
    return texts


def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
            split += line
            if len(split) > chunk_chars:
                texts.append(
                    Text(
                        text=split[:chunk_chars],
                        name=f"{doc.docname} lines {last_line}-{i}",
                        doc=doc,
                    )
                )
                split = split[chunk_chars - overlap :]
                last_line = i
    if len(split) > overlap:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


def _csv_to_timetable_text(df: pd.DataFrame) -> str:
    # Rename the first (unnamed) column to 'day'
    df = df.rename(columns={df.columns[0]: "day"})

    # time_slot from timetable
    time_slots = df.columns[1:]
    time_slots_list = list(time_slots)
    all_time_slots = [
        "7-8am",
        "8-9am",
        "9-10am",
        "10-11am",
        "11-12pm",
        "12-1pm",
        "1-2pm",
        "2-3pm",
        "3-4pm",
        "4-5pm",
        "5-6pm",
        "6-7pm",
    ]

    timetable_text = (
        "Below is the timetable for Kenyatta University,"
        " Computer Science Course for 3rd year 1st Semester:\n\n"
    )

    for _, row in df.iterrows():
        day = row["day"]
        timetable_text += f"On {day}:\n"

        start_time = None
        previous_content = None

        for idx, time_slot in enumerate(time_slots_list):
            cell_content = row[time_slot]

            # Check if the time slots are consecutive
            consecutive = False
            if start_time:
                if all_time_slots.index(time_slot) - all_time_slots.index(
                    start_time
                ) == idx - time_slots_list.index(start_time):
                    consecutive = True

            if pd.notna(cell_content):
                if not previous_content:
                    start_time = time_slot
                    previous_content = cell_content
                elif cell_content != previous_content or not consecutive:
                    # Print the previous content
                    end_time = time_slots_list[idx - 1].split("-")[1]
                    merged_time_slot = f'{start_time.split("-")[0]}-{end_time}'
                    details = previous_content.split(" ")
                    unit = f"{details[0]} {details[1]}"
                    venue = details[2] if len(details) == 3 else "Venue not provided"
                    timetable_text += f"- From {merged_time_slot}, the unit is {unit}, held in {venue}.\n"

                    # Set the current content as the starting content
                    start_time = time_slot
                    previous_content = cell_content

            else:
                # If there's no content but there's a previous content, print it
                if previous_content:
                    end_time = time_slots_list[idx - 1].split("-")[1]
                    merged_time_slot = f'{start_time.split("-")[0]}-{end_time}'
                    details = previous_content.split(" ")
                    unit = f"{details[0]} {details[1]}"
                    venue = details[2] if len(details) == 3 else "Venue not provided"
                    timetable_text += f"- From {merged_time_slot}, the unit is {unit}, held in {venue}.\n"

                    # Reset the starting time and previous content
                    start_time = None
                    previous_content = None

        # Handle the last time slot
        if previous_content:
            end_time = time_slots_list[-1].split("-")[1]
            merged_time_slot = f'{start_time.split("-")[0]}-{end_time}'
            details = previous_content.split(" ")
            unit = f"{details[0]} {details[1]}"
            venue = details[2] if len(details) == 3 else "Venue not provided"
            timetable_text += (
                f"- From {merged_time_slot}, the unit is {unit}, held in {venue}.\n"
            )

        timetable_text += "\n"

    return timetable_text


def parse_timetable_csv(
    path: Path, doc: Doc, chunk_chars: int, overlap: int
) -> List[Text]:
    """Parse a CSV document into chunks representing a timetable for Computer Science."""

    df = pd.read_csv(path)
    timetable_text = _csv_to_timetable_text(df)

    # Check if the timetable text exceeds the chunk size
    texts: List[Text] = []
    start = 0
    while start < len(timetable_text):
        end = start + chunk_chars
        if end > len(timetable_text):
            end = len(timetable_text)
        else:
            # try to split at a newline to maintain meaningful boundaries
            last_newline = timetable_text.rfind("\n", start, end)
            if last_newline != -1:
                end = last_newline

        texts.append(
            Text(
                text=timetable_text[start:end],
                name=f"{doc.docname} - Timetable Part {len(texts) + 1}",
                doc=doc,
            )
        )
        start = end

    return texts


def read_doc(
    path: Path,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
) -> List[Text]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".html"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True)
    elif str_path.endswith(".csv"):  # Handle CSV files
        return parse_timetable_csv(path, doc, chunk_chars, overlap)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap)
