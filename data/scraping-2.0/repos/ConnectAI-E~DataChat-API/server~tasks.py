import os
from urllib.parse import urlparse

import httpx
from functools import cached_property
from tempfile import NamedTemporaryFile
from time import time
from datetime import timedelta, datetime
from celery import Celery
from app import app
from models import save_document, save_embedding, purge_document_by_id, ObjID
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.schema import Document


LARK_HOST = 'https://open.feishu.cn'


def create_celery_app(app=None):
    """
    Create a new Celery object and tie together the Celery config to the app's
    config. Wrap all tasks in the context of the application.

    :param app: Flask app
    :return: Celery app
    """

    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])

    celery.conf.update(app.config.get("CELERY_CONFIG", {}))
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery



celery = create_celery_app(app)

celery.conf.beat_schedule = {
    "sync_feishudoc": {
        "task": "celery_app.sync_feishudoc",
        "schedule": timedelta(hours=1), # 定时1hours执行一次
        # "schedule": timedelta(seconds=5), # 定时2hours执行一次
        # "schedule": 10.0, # 每10秒执行一次
        # "schedule": crontab(minute='*/1'), # 定时每分钟执行一次
        # 使用的是crontab表达式
        "args": (False) # 函数传参的值
    },
    "sync_feishuwiki": {
        "task": "celery_app.sync_feishuwiki",
        "schedule": timedelta(seconds=3900), # 定时1hours执行一次，避免任务一起执行，占资源
        "args": (False) # 函数传参的值
    },
    "sync_yuque": {
        "task": "celery_app.sync_yuque",
        "schedule": timedelta(seconds=3700), # 定时1hours执行一次，避免任务一起执行，占资源
        "args": (False) # 函数传参的值
    },
    "sync_notion": {
        "task": "celery_app.sync_notion",
        "schedule": timedelta(seconds=3800),  # 定时1hours执行一次，避免任务一起执行，占资源
        "args": (False)  # 函数传参的值
    }
}


LOADER_MAPPING = {
    "pdf": (PyMuPDFLoader, {}),
    "word": (UnstructuredWordDocumentLoader, {}),
    "excel": (UnstructuredExcelLoader, {}),
    "markdown": (UnstructuredMarkdownLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    "html": (UnstructuredHTMLLoader, {}),
    "sitemap": (SitemapLoader, {}),
    "default": (UnstructuredFileLoader, {}),
}


def embedding_single_document(doc, fileUrl, fileType, fileName, collection_id, openai=False, uniqid='', version=0):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")
    # 初始化加载器
    # text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents([doc])
    # 先生成document_id，等向量保存完成之后，再保存文档
    document_id = ObjID.new_id()
    # document_ids.append(document_id)
    try:
        doc_result = embeddings.embed_documents([d.page_content for d in split_docs])
        for chunk_index, doc in enumerate(split_docs):
            save_embedding(
                collection_id, document_id,
                chunk_index, len(doc.page_content),
                doc.page_content,
                doc_result[chunk_index],  # embed
            )
        save_document(
            collection_id, fileName or fileUrl, fileUrl, len(split_docs), fileType,
            uniqid=uniqid, version=version,
            document_id=document_id,
        )
        return document_id
    except Exception as e:
        # 出错的时候移除
        purge_document_by_id(document_id)


def get_status_by_id(task_id):
    return celery.AsyncResult(task_id)


def embed_query(text, openai=False):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")

    return embeddings.embed_query(text)


class Lark(object):
    def __init__(self, app_id=None, secret_key=None, app_secret=None, verification_token=None, validation_token=None, encript_key=None, encrypt_key=None, host=LARK_HOST, **kwargs):
        self.app_id = app_id
        self.app_secret = app_secret or secret_key
        self.encrypt_key = encrypt_key or encript_key
        self.verification_token = verification_token or validation_token
        self.host = host

    @cached_property
    def _tenant_access_token(self):
        # https://open.feishu.cn/document/ukTMukTMukTM/ukDNz4SO0MjL5QzM/auth-v3/auth/tenant_access_token_internal
        url = f'{self.host}/open-apis/auth/v3/tenant_access_token/internal'
        result = self.post(url, json={
            'app_id': self.app_id,
            'app_secret': self.app_secret,
        }).json()
        if "tenant_access_token" not in result:
            raise Exception('get tenant_access_token error')
        return result['tenant_access_token'], result['expire'] + time()

    @property
    def tenant_access_token(self):
        token, expired = self._tenant_access_token
        if not token or expired < time():
            # retry get_tenant_access_token
            del self._tenant_access_token
            token, expired = self._tenant_access_token
        return token

    def request(self, method, url, headers=dict(), **kwargs):
        if 'tenant_access_token' not in url:
            headers['Authorization'] = 'Bearer {}'.format(self.tenant_access_token)
        return httpx.request(method, url, headers=headers, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)


class LarkWikiLoader(object):
    def __init__(self, space_id, **kwargs):
        app.logger.info("debug %r", kwargs)
        self.kwargs = kwargs
        self.client = Lark(**kwargs)
        self.space_id = space_id

    def get_info(self):
        url = f"{self.client.host}/open-apis/wiki/v2/spaces/{self.space_id}"
        return self.client.get(url).json()

    def get_spaces(self):
        page_token = ''
        url = f"{self.client.host}/open-apis/wiki/v2/spaces?page_size=50&page_token={page_token}"
        res = self.client.get(url).json()
        while True:
            for item in res.get('data', {}).get('items', []):
                yield item
            if not res.get('data', {}).get('has_more'):
                break
            page_token = res['data']['page_token']

    def get_nodes(self, parent_node_token=''):
        page_token = ''
        while True:
            url = f"{self.client.host}/open-apis/wiki/v2/spaces/{self.space_id}/nodes?page_size=50&page_token={page_token}&parent_node_token={parent_node_token}"
            res = self.client.get(url).json()
            for item in res.get('data', {}).get('items', []):
                yield item
                if item['has_child']:
                    for child_item in self.get_nodes(parent_node_token=item['node_token']):
                        yield child_item
            if not res.get('data', {}).get('has_more'):
                break
            page_token = res['data']['page_token']


class LarkDocLoader(object):
    def __init__(self, fileUrl, document_id, **kwargs):
        app.logger.info("debug %r", kwargs)
        self.client = Lark(**kwargs)
        self.fileUrl = fileUrl
        if not document_id:
            t = fileUrl.split('?')[0].split('/')
            document_id = t.pop()
            type_ = t.pop()
            # https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node
            # https://xxx.feishu.cn/docx/ExGmdqrg4oz2evx7SRuciY78nRe
            # https://xxx.feishu.cn/wiki/V0LuwIeWCiL3yWkq0zBcn1g0nua
            if type_ == 'wiki':
                url = f"{self.client.host}/open-apis/wiki/v2/spaces/get_node?token={document_id}"
                res = self.client.get(url).json()
                if 'data' not in res or 'node' not in res['data']:
                    app.logger.error("error get node %r", res)
                    if res['code'] == 131006:
                        raise Exception('「企联 AI 飞书助手」应用权限配置不正确，请检查以后重新配置')
                    raise Exception('「企联 AI 飞书助手」无该文档访问权限')
                document_id = res['data']['node']['obj_token']
                type_ = res['data']['node']['obj_type']

            if type_ not in ['docx', 'doc']:
                app.logger.error("unsupport type %r", type_)
                raise Exception('「企联 AI 飞书助手」无该文档访问权限')
                # raise Exception(f'unsupport type {type_}')
        self.document_id = document_id
        # TODO (文档只有所有者可以订阅) 查询订阅状态
        # url = f"{self.client.host}/open-apis/drive/v1/files/{document_id}/get_subscribe?file_type={type_}"
        # res = self.client.get(url).json()
        # if not res.get('data', {}).get('is_subscribe'):
        #     url = f"{self.client.host}/open-apis/drive/v1/files/{document_id}/subscribe?file_type={type_}"
        #     res = self.client.post(url).json()
        #     app.logger.info("debug subscribe %r", res)

    def load(self):
        # https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/raw_content
        # https://open.feishu.cn/open-apis/docx/v1/documents/:document_id/raw_content
        url = f"{self.client.host}/open-apis/docx/v1/documents/{self.document_id}/raw_content"
        res = self.client.get(url).json()
        if 'data' not in res or 'content' not in res['data']:
            app.logger.error("error get content %r", res)
            raise Exception('「企联 AI 飞书助手」无该文档访问权限')
            # raise Exception(f'error get content for document')
        return Document(page_content=res['data']['content'], metadata=dict(
            fileUrl=self.fileUrl,
            document_id=self.document_id,
            revision_id=self.version,
            title=self.title,
        ))

    @property
    def version(self):
        return self.file_info.get('revision_id', 0)

    @property
    def title(self):
        return self.file_info.get('title', '')

    @cached_property
    def file_info(self):
        url = f"{self.client.host}/open-apis/docx/v1/documents/{self.document_id}"
        res = self.client.get(url).json()
        app.logger.info("debug file_info %r %r", url, res)
        return res.get('data', {}).get('document', {})


class YuqueDocLoader(object):

    def __init__(self, fileUrl, **kwargs):
        self.fileUrl = fileUrl
        # https://www.yuque.com/yuque/developer/doc
        self.fileUrl = fileUrl
        temp = fileUrl.split('?')[0].split('/')
        self.namespace = '/'.join(temp[-3:-1])
        self.slug = temp[-1]
        self.config = kwargs

    def load(self):
        # -L To follow redirect with Curl
        # curl -L -X "POST" "https://www.yuque.com/api/v2/..." \
        #      -H 'User-Agent: your_name' \
        #      -H 'X-Auth-Token: your_token' \
        #      -H 'Content-Type: application/json' \
        #      -d $'{}'
        # GET /repos/:namespace/docs/:slug
        url = f"https://www.yuque.com/api/v2/repos/{self.namespace}/docs/{self.slug}"
        client = httpx.Client(follow_redirects=True)
        res = client.get(url, headers={'X-Auth-Token': self.config.get('token')}).json()
        if 'data' not in res or 'body' not in res['data']:
            app.logger.error("error get content %r", res)
            raise Exception('「企联 AI 语雀助手」无该文档访问权限')
            # raise Exception(f'error get content for document')
        markdown_content = res['data']['body'].encode()
        with NamedTemporaryFile(delete=False) as f:
            f.write(markdown_content)
            f.close()
            # 拿到markdown_content，然后使用markdown loader重新解析一遍真实内容
            loader = UnstructuredMarkdownLoader(f.name)
            docs = loader.load()
            os.unlink(f.name)
            # 这里只有单个文件
            return Document(
                page_content='\n'.join([d.page_content for d in docs]),
                # ● id - 文档编号● slug - 文档路径● title - 标题● book_id - 仓库编号，就是 repo_id
                # ● ormat - 描述了正文的格式 [lake , markdown]● body - 正文 Markdown 源代码
                # content_updated_at - 文档内容更新时间
                metadata=dict(
                    fileUrl=self.fileUrl,
                    id=res['data']['id'],
                    slug=res['data']['slug'],
                    title=res['data']['title'],
                    # 唯一ID，用于区分
                    uniqid=f"{self.namespace}/{self.slug}",
                    modified=datetime.fromisoformat(res['data']['content_updated_at'].split('.')[0]),
                )
            )

class NotionDocLoader(object):

    def __init__(self, fileUrl, **kwargs):
        # https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151?pvs=4
        # https://www.notion.so/b1-8beaa48d081e44e69000cd789726a151
        self.fileUrl = fileUrl
        self.page_id, self.title = self.extract_ids(fileUrl)[0], self.extract_ids(fileUrl)[1]
        self.config = kwargs

    # notion文档的标题在链接里面，id也需要加入分号分割，需要单独做一个操作
    def extract_ids(self, link):
        # 解析URL
        parsed_url = urlparse(link)
        # 从URL的路径中提取ID
        path = parsed_url.path
        parts = path.split('/')
        ids = parts[-1]
        # 获取后32位
        id = ids[-32:-24] + '-' + ids[-24:-20] + '-' + ids[-20:-16] + '-' + ids[-16:-12] + '-' + ids[-12:]
        title = ids[:-33]
        return id, title

    def retrieve_block_children(self):
        """
            这里的  url  是接受处理过的id号拼接为的接口地址
        """
        url = f"https://api.notion.com/v1/blocks/{self.page_id}/children"

        # notion的版本可能会有更新的问题
        headers = {
            "Authorization": self.config.get('token'),
            "Notion-Version": '2022-06-28'
        }


        blocks = []
        cursor = None
        while True:
            params = {}
            # if cursor:
            #     params["start_cursor"] = cursor
            res = httpx.get(url, headers=headers, params=params).json()

            blocks.extend(res.get("results", []))
            has_more = res.get("has_more", False)
            if not has_more:
                break
            cursor = res.get("next_cursor")
            # print(json.dumps(blocks))
        return blocks


    # 获取到  block  中  paragraph.rich_text[*].plain_text  的富文本内容
    def get_plain_text_from_rich_text(self, rich_text):
        return "".join([t['plain_text'] for t in rich_text])

    def get_text_from_block(self, block):
        text = ""
        # 这里加异常处理原因是，获取到的content中不一定有 block[block['type']]['rich_text']
        block = block[0]
        try:
            if block[block['type']]['rich_text']:
                text = self.get_plain_text_from_rich_text(block[block['type']]['rich_text'])
        except:
            block_type = block['type']
            if block_type == "unsupported":
                text = "[Unsupported block type]"

            elif block_type == "bookmark":
                text = block['bookmark']['url']

            elif block_type == "child_database":
                text = block['child_database']['title']

            elif block_type == "child_page":
                text = block['child_page']['title']

            # elif block_type in ["embed", "video", "file", "image", "pdf"]:
            #     text = get_media_source_text(block)

            elif block_type == "equation":
                text = block['equation']['expression']

            elif block_type == "link_preview":
                text = block['link_preview']['url']

            elif block_type == "synced_block":
                if 'synced_from' in block['synced_block']:
                    synced_with_block = block['synced_block']['synced_from']
                    text = f"This block is synced with a block with the following ID: {synced_with_block[synced_with_block['type']]}"
                else:
                    text = "Source sync block that another blocked is synced with."

            elif block_type == "table":
                text = f"Table width: {block['table']['table_width']}"

            elif block_type == "table_of_contents":
                text = f"ToC color: {block['table_of_contents']['color']}"

            elif block_type in ["breadcrumb", "column_list", "divider"]:
                text = "No text available"

            else:
                text = "[Needs case added]"

        string_data = f"{block['type']}: {text}"
        # 使用冒号分割字符串
        key, value = string_data.split(':')
        # 创建字典
        my_dict = {key.strip(): value.strip()}

        # 只返回正文的内容
        if 'paragraph' in my_dict and my_dict['paragraph'] != '':
            return my_dict['paragraph']
        else:
            return ''

        # return f"{block['type']}: {text}"

    def load(self):
        url = f"https://api.notion.com/v1/blocks/{self.page_id}/children"

        blocks, cursor = [], None
        headers = {
            "Authorization": self.config.get('token'),
            "Notion-Version": '2022-06-28'
        }
        params = {}
        if cursor:
            params["start_cursor"] = cursor
            res = httpx.get(url, headers=headers,params=params).json()
        blocks.extend(res.get("results", []))

        text = self.get_text_from_block(blocks)

        if res['results'] == '':
            # app.logger.error("error get content %r", res)
            raise Exception('「企联 AI Notion助手」无该文档访问权限')
            # raise Exception(f'error get content for document')
        return Document(
            page_content=text,
            metadata=dict(
                fileUrl=self.fileUrl,
                id=self.extract_ids(self.fileUrl)[0],
                title=self.extract_ids(self.fileUrl)[1],
                # 唯一ID，用于区分
                uniqid=f"{self.title}-{self.page_id}",
                modified=datetime.fromisoformat(res['result']['last_edited_time'].split('.')[0]),
            )
        )


