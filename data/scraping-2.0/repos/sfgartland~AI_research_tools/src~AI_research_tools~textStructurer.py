from difflib import SequenceMatcher
from typing import Callable
from openai import OpenAI
import regex

from rich import print


def getStructureForSection(transcriptSection):
    client = OpenAI()
    # TODO Check if I could switch over to json response here?
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a college professor structuring a transcript. Keeping the original content",
            },
            {"role": "user", "content": transcriptSection},
            {
                "role": "user",
                "content": "Output to markdown. Edit the previous message into paragraphs with headings. Keep the original content. Return only the last sentence of each paragraph. Wrap each sentence in square brackets.",
            },
        ],
    )

    return response


def structureSection(transcriptSection, structure):
    # Finds all last sentences of paragraphs and headings
    parsEndings = [
        match.groups()
        for match in regex.finditer("(#+[^\[\]]+)?\n\[([^\[\]]+)\]", structure)
    ]
    print(len(parsEndings))
    output = []
    for heading, parEnding in parsEndings:
        # Gets fuzzy matched sentence from body of transcript
        match = regex.search(
            f"({regex.escape(parEnding)})" + "{e<=2}", transcriptSection
        )
        if match:
            paragraph = transcriptSection[: match.end()]
            paragraph = regex.sub("^\s", "", paragraph)  # Cleans up paragraph
            output.append([heading, paragraph])  # Adds paragraph with heading to output
            transcriptSection = transcriptSection[
                match.end() :
            ]  # Removes processed parts of transcript

    return output


def getStructureFromGPT(
    transcriptPath, progressCallback: Callable[[int, int], None] = None
):
    transcript = open(transcriptPath, "r", encoding="utf-8").read()
    # Split transcript into chunks of 6000 characters for processing
    chunks, chunk_size = len(transcript), 6000
    transcriptChunks = [
        transcript[i : i + chunk_size] for i in range(0, chunks, chunk_size)
    ]

    structuredSections = []
    responses = []
    if progressCallback:
        progressCallback(0, len(transcriptChunks))
    for index, transChunk in enumerate(transcriptChunks):
        # If there are already processed section, remove the last and include it in this one
        prevChunk = transcriptChunks[index - 1]
        if len(structuredSections) > 0:
            spilloverIndex = prevChunk.index(structuredSections[-1][1])
            transChunk = prevChunk[spilloverIndex:] + transChunk
            structuredSections = structuredSections[:-1]
        res = getStructureForSection(transChunk)
        responses.append(res)
        structure = res.choices[0].message.content
        structuredSection = structureSection(transChunk, structure)

        # GPT does not really include the last bit of text in the last paragraph,
        # overrule this and add the complete manuscript if it is the last block
        if index == len(transcriptChunks) - 1 and len(structuredSection) > 0:
            finalSpilloverIndex = transChunk.index(structuredSection[-1][1])
            structuredSection[-1][1] = transChunk[finalSpilloverIndex:]

        [
            structuredSections.append(x) for x in structuredSection
        ]  # Adds all the sections to the main output
        if progressCallback:
            progressCallback(index + 1, len(transcriptChunks))

    return structuredSections, responses


def checkSimilarityToOriginal(original, structured):
    with open(original, "r", encoding="utf-8") as f:
        original = f.read()
    with open(structured, "r", encoding="utf-8") as f:
        structured = f.read()
    originalStripped = regex.sub("\s", "", original)
    structuredStripped = regex.sub("(#+.+)|\n|\s", "", structured)

    return SequenceMatcher(None, originalStripped, structuredStripped).ratio()


def sectionListToMarkdown(sectionList):
    return "\n".join([f"{section[0]}\n{section[1]}\n" for section in sectionList])
