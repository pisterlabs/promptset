from typing import Any, List, Optional

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.prompts import PromptTemplate
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore


class Pipeline:
    def __init__(
        self,
        llm_model,
        embed_model_name: str = "sentence-transformers/all-MiniLM-l6-v2",
        embed_model: Optional[Any] = None,
        prompt_temp_path: str | None = "prompt_templates/default.txt",
        service_context: Optional[ServiceContext] = None,
        is_debug: bool = False,
    ) -> None:
        """
        Pipelineクラスのコンストラクタ

        Args:
            llm_model: LLMモデル
            embed_model_name (str, optional): Embeddingモデルの名前. Defaults to "sentence-transformers/all-MiniLM-l6-v2".
            embed_model (Optional[Any], optional): Embeddingsのモデル. Defaults to None.
            prompt_temp_path (str | None, optional): Prompt Templateのパス. Defaults to "prompt_templates/default.txt".
            is_debug (bool, optional): デバッグモードかどうか. Defaults to False.
        """
        self._prompt_template = self._load_prompt_template(prompt_temp_path)
        self.is_debug = is_debug
        self._storage_context = None
        self.vector_store_index = None
        self.node_parser = None
        self.callback_manager = None
        self._embed_model = None
        self._service_context = None

        if is_debug:
            # デバッグの設定
            from llama_index.callbacks import CallbackManager, LlamaDebugHandler

            self._llama_debug_handler = LlamaDebugHandler(
                print_trace_on_end=True
            )
            self.callback_manager = CallbackManager([self._llama_debug_handler])

        # Service Context の作成
        if service_context is not None and isinstance(
            service_context, ServiceContext
        ):
            self._service_context = service_context
        else:
            # Embeddings の設定
            self._embed_model = embed_model or HuggingFaceEmbeddings(
                model_name=embed_model_name
            )
            self._service_context = ServiceContext.from_defaults(
                llm=llm_model,
                embed_model=self._embed_model,
                callback_manager=self.callback_manager,
            )

    def _create_strage_context(self) -> None:
        """Storage Contextを作成する関数"""
        self._storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )

    def _load_prompt_template(self, prompt_temp_path: str) -> PromptTemplate:
        """Prompt Templateを読み込む関数

        Args:
            prompt_temp_path (str): Prompt Templateのパス

        Returns:
            PromptTemplate: 読み込まれたPrompt Template
        """
        with open(prompt_temp_path, "r") as f:
            return PromptTemplate(f.read())

    def read_document(
        self,
        docs_dir_path: str,
        required_exts: List[str] = [".pdf", ".txt"],
        recursive: bool = True,
    ) -> List[str]:
        """SimpleDirectoryReaderを使用してドキュメントを読み込む関数

        Args:
            docs_dir_path (str): ドキュメントのディレクトリのパス
            required_exts (List[str], optional): 読み込むドキュメントの拡張子のリスト. Defaults to [".pdf", ".txt"].
            recursive (bool, optional): 再帰的に読み込むかどうか. Defaults to True.
        """
        try:
            # SimpleDirectoryReaderを作成し、ドキュメントを読み込む
            reader = SimpleDirectoryReader(
                input_dir=docs_dir_path,
                required_exts=required_exts,
                recursive=recursive,
            )
            docs = reader.load_data()
        except FileNotFoundError as e:
            # ファイルが見つからなかった場合
            print(f"Error while reading documents: {e}")
            docs = []
        except Exception as e:
            # その他のエラーが発生した場合
            print(f"Error while reading documents: {e}")
            docs = []

        return docs

    def vectorize_documents(self, docs: List[str]) -> None:
        """ドキュメントをベクトル化する関数

        Args:
            docs (List[str]): ドキュメントのリスト
        """
        try:
            # VectorStoreIndexを作成し、ドキュメントをベクトル化する
            self.vector_store_index = VectorStoreIndex.from_documents(
                docs,
                storage_context=self._storage_context,
                service_context=self._service_context,
            )
        except Exception as e:
            # エラーが発生した場合
            print(f"Error while vectorizing documents: {e}")
            self.vector_store_index = None

    def read_vector_index(self, index_dir_path: str) -> None:
        """ベクトルインデックスを読み込む関数

        Args:
            index_dir_path (str): ベクトルインデックスが保存されたディレクトリへのパス
        """
        try:
            # StorageContextを作成する
            self._storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(
                    persist_dir=index_dir_path
                ),
                vector_store=SimpleVectorStore.from_persist_dir(
                    persist_dir=index_dir_path
                ),
                index_store=SimpleIndexStore.from_persist_dir(
                    persist_dir=index_dir_path
                ),
            )

            # ベクトルインデックスを読み込む
            self.vector_store_index = load_index_from_storage(
                self._storage_context, service_context=self._service_context
            )
        except FileNotFoundError as e:
            # ファイルが見つからなかった場合
            print(f"Error while reading vector index: {e}")
            self.vector_store_index = None
        except Exception as e:
            # その他のエラーが発生した場合
            print(f"Error while reading vector index: {e}")
            self.vector_store_index = None

    def generate_llm_response_for_prompt_temp(
        self,
        prompt: str,
        prompt_template: str | None = None,
    ) -> str:
        """Prompt TemplateとPromptを用いて、LLMに指示を出す関数

        Args:
            prompt (str): Prompt
            prompt_template (str): Prompt Template. Defaults to None.

        Returns:
            str: LLMによる回答
        """
        if self._service_context.llm is None:
            # LLMが読み込まれていない場合
            raise ValueError(
                "You need to load LLM model before generating response."
            )

        if prompt_template is None:
            template = self._prompt_template
        else:
            # Prompt Templateを作成する
            template = PromptTemplate(prompt_template)

        try:
            prompt = template.format(prompt_text=prompt)
            # LLMに指示を出す
            response = self._service_context.llm.complete(
                prompt=prompt,
            )
        except Exception as e:
            # エラーが発生した場合
            print(f"Error while generating response: {e}")
            response = ""

        return response

    def generate_answer_from_vector_index(self, prompt: str) -> str:
        """ベクトルインデックスを用いて、Promptに対する回答を生成する関数

        Args:
            prompt (str): Prompt

        Returns:
            str: Promptに対する回答
        """
        if self.vector_store_index is None:
            # ベクトルインデックスが読み込まれていない場合
            raise ValueError(
                "You need to read documents or vector index before generating answer."
            )

        try:
            # QueryEngineを作成する
            query_engine = self.vector_store_index.as_query_engine(
                response_mode="tree_summarize",
                text_qa_template=self._prompt_template,
                service_context=self._service_context,
            )

            # Promptに対する回答を生成する
            answer = query_engine.query(prompt)
        except Exception as e:
            # エラーが発生した場合
            print(f"Error while generating answer: {e}")
            answer = ""

        return answer
