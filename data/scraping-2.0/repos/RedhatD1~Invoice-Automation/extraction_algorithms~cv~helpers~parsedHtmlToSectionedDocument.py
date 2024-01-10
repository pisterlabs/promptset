from langchain.docstore.document import Document


# This module creates the sections grabbed from the parsed html
# The Document class is used to represent a document with content and associated metadata.
def convert(parsed_html: list, html: Document) -> list:
    cur_idx = -1
    # The current idx position in our snippet object

    sectioned_document = []
    # This will contain the organized snippets where there are formed together
    # based on their relative font size in the document

    # Assumption: headings have higher font size than their respective content
    for snippet in parsed_html:
        # if the current snippet's font size > previous section's heading => it is a new heading
        if not sectioned_document or snippet[1] > sectioned_document[cur_idx].metadata['heading_font']:
            metadata = {'heading': snippet[0], 'content_font': 0, 'heading_font': snippet[1]}
            # Since we saved the heading and its fontsize as a tuple,
            # Its first index contains the heading text,
            # and the second index contains the fontsize of the heading,
            # We also initialized the content_font of the heading to 0

            metadata.update(html.metadata)  # Here data is inherited from a langchain document library.

            sectioned_document.append(Document(page_content='', metadata=metadata))
            cur_idx += 1
            continue

        # if (current snippet's font size <= previous section's content)
        # -> content belongs to the same section
        if not sectioned_document[cur_idx].metadata['content_font'] \
                or snippet[1] <= sectioned_document[cur_idx].metadata['content_font']:

            # checks if the content font size is not yet set or
            # whether the current snippet's font size (snippet[1]) is
            # lower than or equal to the font size of the previous section's content
            # if these conditions are true, then the new snippet is part of the previous section
            sectioned_document[cur_idx].page_content += snippet[0]
            sectioned_document[cur_idx].metadata['content_font'] \
                = max(snippet[1], sectioned_document[cur_idx].metadata['content_font'])
            continue

        else:
            # else if current snippet's font size > previous section's content
            # but less than previous section's heading than also make a new section
            metadata = {'heading': snippet[0], 'content_font': 0, 'heading_font': snippet[1]}
            metadata.update(html.metadata)
            sectioned_document.append(Document(page_content='', metadata=metadata))
            cur_idx += 1
    return sectioned_document
