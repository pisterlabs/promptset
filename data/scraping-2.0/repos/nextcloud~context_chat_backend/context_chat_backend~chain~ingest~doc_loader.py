from logging import error as log_error
import re
import tempfile
from typing import BinaryIO

from fastapi import UploadFile
from pandas import read_csv, read_excel
from pypandoc import convert_text
from pypdf import PdfReader
from langchain.document_loaders import (
	UnstructuredPowerPointLoader,
	UnstructuredEmailLoader,
)


def _temp_file_wrapper(file: BinaryIO, loader: callable, sep: str = '\n') -> str:
	raw_bytes = file.read()
	tmp = tempfile.NamedTemporaryFile(mode='wb')
	tmp.write(raw_bytes)

	docs = loader(tmp.name)

	tmp.close()
	if not tmp.delete:
		import os
		os.remove(tmp.name)

	return sep.join(map(lambda d: d.page_content, docs))


# -- LOADERS -- #

def _load_pdf(file: BinaryIO) -> str:
	pdf_reader = PdfReader(file)
	return '\n\n'.join([page.extract_text().strip() for page in pdf_reader.pages])


def _load_csv(file: BinaryIO) -> str:
	return read_csv(file).to_string(header=False, na_rep='')


def _load_epub(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'epub').strip()


def _load_docx(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'docx').strip()


def _load_ppt_x(file: BinaryIO) -> str:
	return _temp_file_wrapper(file, lambda fp: UnstructuredPowerPointLoader(fp).load()).strip()


def _load_rtf(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'rtf').strip()


def _load_rst(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'rst').strip()


def _load_xml(file: BinaryIO) -> str:
	data = file.read().decode('utf-8')
	data = re.sub(r'</.+>', '', data)
	return data.strip()


def _load_xlsx(file: BinaryIO) -> str:
	return read_excel(file).to_string(header=False, na_rep='')


def _load_odt(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'odt').strip()


def _load_email(file: BinaryIO, ext: str = 'eml') -> str | None:
	# NOTE: msg format is not tested
	if ext not in ['eml', 'msg']:
		return None

	# TODO: implement attachment partitioner using unstructured.partition.partition_{email,msg}
	# since langchain does not pass through the attachment_partitioner kwarg
	def attachment_partitioner(
		filename: str,
		metadata_last_modified: None = None,
		max_partition: None = None,
		min_partition: None = None,
	):
		...

	return _temp_file_wrapper(
		file,
		lambda fp: UnstructuredEmailLoader(fp, process_attachments=False).load(),
	).strip()


def _load_org(file: BinaryIO) -> str:
	return convert_text(file.read(), 'plain', 'org').strip()


# -- LOADER FUNCTION MAP -- #

_loader_map = {
	'application/pdf': _load_pdf,
	'application/epub+zip': _load_epub,
	'text/csv': _load_csv,
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document': _load_docx,
	'application/vnd.ms-powerpoint': _load_ppt_x,
	'application/vnd.openxmlformats-officedocument.presentationml.presentation': _load_ppt_x,
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': _load_xlsx,
	'application/vnd.oasis.opendocument.spreadsheet': _load_xlsx,
	'application/vnd.ms-excel.sheet.macroEnabled.12': _load_xlsx,
	'application/vnd.oasis.opendocument.text': _load_odt,
	'text/rtf': _load_rtf,
	'text/x-rst': _load_rst,
	'application/xml': _load_xml,
	'message/rfc822': _load_email,
	'application/vnd.ms-outlook': _load_email,
	'text/org': _load_org,
}


def decode_source(source: UploadFile) -> str | None:
	try:
		if _loader_map.get(source.headers.get('type')):
			return _loader_map[source.headers.get('type')](source.file)

		return source.file.read().decode('utf-8')
	except Exception as e:
		log_error(f'Error decoding source file ({source.filename}): {e}')
		return None
