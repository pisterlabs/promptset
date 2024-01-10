import logging
import os
import urllib
from threading import Lock

from fastapi import File, Form, Body, Query, UploadFile
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
from server.knowledge_base.kb_api import update_video_path
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile)
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import Json
import json
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_file_repository import get_file_detail
from typing import List, Dict, Tuple
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi


class DocumentWithScore(Document):
    score: float = None


# 创建一个字典来存储每个文件路径的锁
save_file_to_kb_locks = {}


def search_docs(query: str = Body(..., description="用户输入", examples=["你好"]),
                knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(SCORE_THRESHOLD,
                                              description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                              ge=0, le=1),
                ) -> List[DocumentWithScore]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return []
    docs = kb.search_docs(query, top_k, score_threshold)
    data = [DocumentWithScore(**x[0].dict(), score=x[1]) for x in docs]

    return data


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    '''
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    '''

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容

            # 获取文件路径的锁，如果不存在则创建一个
            lock = save_file_to_kb_locks.setdefault(file_path, Lock())

            with lock:  # 使用锁来防止并发写入
                if (os.path.isfile(file_path)
                        and not override
                        and os.path.getsize(file_path) == len(file_content)
                ):
                    # TODO: filesize 不同后的处理
                    file_status = f"文件 {filename} 已存在。"
                    logger.warn(file_status)
                    return dict(code=404, msg=file_status, data=data)

                with open(file_path, "wb") as f:
                    f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


def _download_files_in_thread(link: Tuple[str, str],
                              knowledge_base_name: str,
                              override: bool):
    '''
    通过多线程将链接的字幕保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    '''

    def save_download_file(filename: str, url: str, knowledge_base_name: str, override: bool) -> dict:
        '''
        下载单个文件。
        '''
        try:
            logger.info("=============" + filename + "=================")
            logger.info("=============" + url + "=================")
            filename = filename + ".txt"
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = get_youtube_subtitles_with_interval(url)  # 读取上传文件的内容
            if file_content is None:
                file_status = f"文件 {filename} 下载失败。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)
            # 获取文件路径的锁，如果不存在则创建一个
            lock = save_file_to_kb_locks.setdefault(file_path, Lock())

            with lock:
                # 保证本地文件和数据库数据一致性：先写文件，成功再修改数据库。
                # 因为用户任何时候都需要访问数据库，所以数据库的数据必须保证是对的。
                # 用户不直接访问本地文件，系统需要经过数据库访问本地文件，所以本地文件可以暂时有问题，但是数据库不能有任何问题。
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                # 尝试更新视频路径
                kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
                # 更新数据库中的video_path
                kb.update_video_path_in_kb(url)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"filename": link[0], "url": link[1], "knowledge_base_name": knowledge_base_name, "override": override}]
    for result in run_in_thread_pool(save_download_file, params=params):
        yield result


# 似乎没有单独增加一个文件上传API接口的必要
# def upload_files(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件")):
#     '''
#     API接口：上传文件。流式返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
#     '''
#     def generate(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     return StreamingResponse(generate(files, knowledge_base_name=knowledge_base_name, override=override), media_type="text/event-stream")


# TODO: 等langchain.document_loaders支持内存文件的时候再开通
# def files2docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件"),
#                 save: bool = Form(True, description="是否将文件保存到知识库目录")):
#     def save_files(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     def files_to_docs(files):
#         for result in files2docs_in_thread(files):
#             yield json.dumps(result, ensure_ascii=False)


def upload_docs(files: List[UploadFile] = File(None, description="上传文件，支持多文件"),
                knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
                override: bool = Form(False, description="覆盖已有文件"),
                to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
                chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
                chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
                zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
                docs: Json = Form({}, description="自定义的docs，需要转为json字符串",
                                  examples=[{"test.txt": [Document(page_content="custom doc")]}]),
                not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
                link: Json = Form(Tuple[str, str], description="视频链接，用于索引视频和下载字幕"),
                ) -> BaseResponse:
    '''
    API接口：上传文件，并/或向量化
    '''
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    if files is not None:
        # TODO：如果还有视频文件，调用 kb.update_video_path_in_kb() 修改kb和数据库，可能需要去 _save_files_in_thread() 里修改
        for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
            filename = result["data"]["file_name"]
            if result["code"] != 200:
                failed_files[filename] = result["msg"]

            if filename not in file_names:
                file_names.append(filename)

    if link is not None:
        for result in _download_files_in_thread(link, knowledge_base_name=knowledge_base_name, override=override):
            filename = result["data"]["file_name"]
            if result["code"] != 200:
                failed_files[filename] = result["msg"]

            if filename not in file_names:
                # 保存成功的filename
                file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def delete_docs(knowledge_base_name: str = Body(..., examples=["samples"]),
                file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
                delete_content: bool = Body(False),
                not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
                ) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


def update_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: Json = Body({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    '''
    更新知识库文档
    '''
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    # 拿到kb_service里的某个kb服务实例，比如：faiss_kb_service, ...（都继承kb_service（base文件下））
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # TODO: 可以开多线程进行 summary

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
):
    '''
    下载知识库文档
    '''
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    '''
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    '''

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            kb.create_kb()
            kb.clear_vs()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    return StreamingResponse(output(), media_type="text/event-stream")


def get_youtube_subtitles_with_interval(youtube_url, interval=10):
    try:
        logger.info("============= in youtube loader=================")
        # 从YouTube URL中提取视频ID
        video_id = youtube_url.split("v=")[1]

        # 获取字幕
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        subtitles_with_interval = []

        for i in range(0, len(transcript), interval):
            segment_start = int(transcript[i]['start'])
            segment_end = int(transcript[min(i + interval, len(transcript) - 1)]['start'] +
                              transcript[min(i + interval, len(transcript) - 1)]['duration'])

            # 拼接文本和时间戳
            subtitle_text = " ".join(entry['text'] for entry in transcript[i:i + interval])
            timestamp = f"({segment_start} - {segment_end})"
            subtitles_with_interval.append(f"{timestamp} {subtitle_text}")

        file_content = "\n".join(subtitles_with_interval).encode('utf-8')
        logger.info("============= end youtube loader=================")
        return file_content

    except Exception as e:
        return None
#
#
# # # 用法示例
# # youtube_url = "https://www.youtube.com/watch?v=o5MutYFWsM8"
# # subtitles = get_youtube_subtitles_with_interval(youtube_url)
# #
# # if subtitles:
# #     for i, subtitle in enumerate(subtitles):
# #         print(f"Subtitle {i + 1}: {subtitle}\n")
# # else:
# #     print("无法获取字幕。")
