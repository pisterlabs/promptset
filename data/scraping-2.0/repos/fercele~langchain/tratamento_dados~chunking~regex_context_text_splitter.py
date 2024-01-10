from typing import List
from langchain.schema import Document
import re
from io import StringIO
import os


"""
Este módulo é exploratório, e visa subsidiar um processo genérico e automático para extração de dados de com reaproveitamento
dos separadores como conteúdo de contexto dos próprios chunks.
Visa contornar limitações dos text splitter padrão do langchain.
Útil para dados de documentos semi-estruturados, que permite agrupar seções semanticamente relacionadas, como cláusulas de contratos,
ou perguntas-respostas de um FAQ.

TODO - TRANSFORMAR EM CLASSE que herde do TextSplitter básico do LangChain
"""

def split_text_including_separator(original_doc:Document, regex:str, prefixo_as_contexto:bool = False, max_chunk_size:int = None, metadata_to_content:List[str] = None) -> List[Document]:
    """
    Splits a string by a regex, and includes the regex match at the beginning of the following split.
    """
    result:List[Document] = []
    last_split_end = None

    current_chunk = StringIO() 
    current_match = None   
    context_prefix = ""

    for match in re.finditer(regex, original_doc.page_content, re.IGNORECASE | re.MULTILINE):

        if last_split_end == None:
            #Primeira iteração
            if match.start() > 0:
                if prefixo_as_contexto:
                    #O conteudo antes do primeiro match é considerado contexto de todos os posteriores
                    context_prefix = original_doc.page_content[:match.start()] + "\n"
                else:
                    #Adiciona todo o conteúdo anterior ao primeiro match como chunk separad
                    temp_chunk = StringIO()
                    temp_chunk.write(original_doc.page_content[:match.start()])
                    add_chunk_to_result(original_doc, result, temp_chunk)
                
                if metadata_to_content is not None: 
                    #Adiciona os metadados selecionados ao conteúdo do chunk
                    for metadata_key in metadata_to_content:
                        if metadata_key in original_doc.metadata:
                            context_prefix += f"{original_doc.metadata[metadata_key]}\n"
        
        else:

            if max_chunk_size is None:
                current_chunk.write(context_prefix)
                current_chunk.write(current_match)
                current_chunk.write(original_doc.page_content[last_split_end:match.start()])
                add_chunk_to_result(original_doc, result, current_chunk)
            else:
                content_to_add = current_match + original_doc.page_content[last_split_end:match.start()]
                current_chunk_content = current_chunk.getvalue()

                if len(current_chunk_content) == 0:
                    current_chunk.write(context_prefix)
                    current_chunk.write(content_to_add)
                else:
                    if len(current_chunk_content) + len(content_to_add) > max_chunk_size:
                        add_chunk_to_result(original_doc, result, current_chunk)

                        current_chunk.write(context_prefix)
                        current_chunk.write(content_to_add)
                    else:
                        current_chunk.write(content_to_add)
        
        current_match = match.group(0)
        last_split_end = match.end()

    if current_match is None:
        #Não houve nenhum match para a expressão de separador fornecida - retorna lista com o próprio doc de entrada
        result.append(original_doc)
        return result
    
    #Trata o trecho referente ao último match
    current_chunk_content = current_chunk.getvalue()
    if current_chunk_content != "":
    
        if last_split_end < len(original_doc.page_content): 
            #Faltou ler do último separador em diante
            content_to_add = current_match + original_doc.page_content[match.end():]

            if len(current_chunk_content) + len(content_to_add) > max_chunk_size:
                add_chunk_to_result(original_doc, result, current_chunk)

                current_chunk.write(context_prefix)
                current_chunk.write(content_to_add)

            else:
                current_chunk.write(content_to_add)

        add_chunk_to_result(original_doc, result, current_chunk)

    elif last_split_end < len(original_doc.page_content): 
        #Faltou ler do último separador em diante
        content_to_add = current_match + original_doc.page_content[match.end():]

        current_chunk.write(context_prefix)
        current_chunk.write(content_to_add)

        add_chunk_to_result(original_doc, result, current_chunk)

         
    return result


def add_chunk_to_result(original_document:Document, result:List[Document], current_chunk:StringIO, metadata_adicional=None):
    #Cria um novo documento com o conteudo do StringIO e com os metadados do documento original
    document_chunk = Document(page_content=current_chunk.getvalue(), metadata=dict(original_document.metadata))
    if metadata_adicional is not None:
        document_chunk.metadata.update(metadata_adicional)

    result.append(document_chunk)

    #Limpa o StringIO para ser reutilizado
    current_chunk.truncate(0)
    current_chunk.seek(0)

CHUNK_SEPARATOR = "\n||||||||||||||||||||||||||||||||||\n"


def save_chunks(result:List[Document], chunk_file_path:str):

    if os.path.exists(chunk_file_path):
        print(f"Removing {chunk_file_path}")
        os.remove(chunk_file_path)
    
    with open(chunk_file_path, "w") as f:
        for chunk in result:
            f.write(str(chunk.metadata) + "\n")
            f.write(chunk.page_content)
            f.write(CHUNK_SEPARATOR)





