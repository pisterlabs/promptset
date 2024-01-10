from langchain.text_splitter import (
	MarkdownTextSplitter,
	RecursiveCharacterTextSplitter,
	TextSplitter,
)


def get_splitter_for(mimetype: str = 'text/plain') -> TextSplitter:
	kwargs = {
		'chunk_size': 2000,
		'chunk_overlap': 200,
		'add_start_index': True,
		'strip_whitespace': True,
		'is_separator_regex': True,
		'keep_separator': True,
	}

	mt_map = {
		'text/markdown': MarkdownTextSplitter(**kwargs),
		'application/json': RecursiveCharacterTextSplitter(separators=['{', '}', r'\[', r'\]', ',', ''], **kwargs),  # noqa: E501
		# processed csv, does not contain commas
		'text/csv': RecursiveCharacterTextSplitter(separators=['\n', ' ', ''], **kwargs),
		# remove end tags for less verbosity, and remove all whitespace outside of tags
		'application/xml': RecursiveCharacterTextSplitter(separators=['\n', ' ', ''], **kwargs),
		'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ' ', ''], **kwargs),  # noqa: E501
		'application/vnd.ms-excel.sheet.macroEnabled.12': RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ' ', ''], **kwargs),  # noqa: E501
	}

	if mimetype in mt_map.keys():
		return mt_map[mimetype]

	# all other mimetypes
	return RecursiveCharacterTextSplitter(
		separators=['\n\n', '\n', r'\.', r'\?', '!', ';', r'\|+', ' ', ''],
		**kwargs
	)


__all__ = [ 'get_splitter_for' ]
