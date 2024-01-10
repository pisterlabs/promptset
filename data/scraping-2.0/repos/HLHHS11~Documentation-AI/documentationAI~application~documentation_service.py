from typing import Dict
import asyncio
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from openai import InvalidRequestError

from documentationAI.utils.topological_sort import topological_sort
from documentationAI.domain.repositories.interfaces import IDocumentRepository
from documentationAI.domain.services.analyzer import IAnalyzerHelper, IPackageAnalyzer
from documentationAI.domain.models.document import Document
from documentationAI.domain.services.prompt_generator.documentation import DocumentationPromptGenerator, DocumentationPromptGeneratorContext
from documentationAI.domain.models.symbol import ISymbolId


class DocumentationService:

    def __init__(
            self,
            package_analyzer: IPackageAnalyzer,
            document_repository: IDocumentRepository,
            helper: IAnalyzerHelper,
            prompt_generator: DocumentationPromptGenerator
    ):
        self.package_analyzer = package_analyzer
        self.document_repository = document_repository
        self.helper = helper
        self.prompt_generator = prompt_generator

    async def generate_package_documentation(
        self,
        project_root_dir: str,
        package_root_dir: str,
        package_name: str
    ) -> None:
        
        # for debugging
        self._counter = 0

        self._package_root_dir = package_root_dir
        # パッケージ解析サービスを利用して，シンボルの依存関係を取得するとともに，解決順序を取得する
        self._dependencies_map = self.package_analyzer.generate_dag(package_root_dir, package_name)
        self._resolved = topological_sort(self._dependencies_map)

        # 参照の関係も取得するため，逆向きのDAGを生成する
        self._reversed_dependencies_map = self.package_analyzer.generate_reversed_dag_from_dag(self._dependencies_map)

        # progressは現状"pending", "processing", "fulfilled", "rejected"の4つの値をとることにする
        self._progress_map: Dict[ISymbolId, str] = {symbol_id: "pending" for symbol_id in self._dependencies_map.keys()}

        # 解決された順番にしたがってドキュメンテーション生成を行う。
        initial_process_symbol_ids: list[ISymbolId] = []
        for symbol_id in self._resolved:
            if self._can_be_processed(symbol_id):
                initial_process_symbol_ids.append(symbol_id)
            else:
                # 一度でもelseに入ったなら，その後の要素はすべて何らかの依存先をもつ=初期には処理できない。そのためこれ以上ループを回しても意味がない
                break
        tasks = [self._exec_documentation_chain(symbol_id) for symbol_id in initial_process_symbol_ids]
        await asyncio.gather(*tasks)

        return


    async def _documentation(self, symbol_id: ISymbolId):
        # TODO: 解析対象のファイルで「存在しないモジュールをインポートしている」場合，ここでエラーになる。なぜなら，dependencies_mapは存在するモジュールのシンボル情報を解析して保存した辞書だから。
        required_symbol_ids = self._dependencies_map[symbol_id]
        # 1. シンボルのソース定義を取得
        print(f"[STARTED]: Generating document for {symbol_id.stringify()}...")
        symbol_def = self.helper.get_symbol_def(symbol_id, self._package_root_dir)

        # 2. 依存先シンボルのドキュメントを取得
        required_symbol_docs: list[Document] = []
        for required_symbol_id in required_symbol_ids:
            required_symbol_doc = self.document_repository.get_by_symbol_id(required_symbol_id)
            if required_symbol_doc: # TODO: 見つからなければ何をするのか，具体的に考えておくこと
                required_symbol_docs.append(required_symbol_doc)
        
        # 3. AIに投げるための質問文を生成
        context = DocumentationPromptGeneratorContext.from_dict({
            "symbol_id": symbol_id,
            "symbol_def": symbol_def,
            "required_symbol_docs": required_symbol_docs,
        })
        prompt = self.prompt_generator.generate(context)

        # 4. AIにプロンプトを投げて，返ってきた返答をドキュメントとして保存
        chat = ChatOpenAI(temperature = 0.25)
        messages = [HumanMessage(content = prompt)]
        
        try:
            response = await asyncio.to_thread(chat, messages)
        # TODO: 例外処理をもっと丁寧に書く
        except InvalidRequestError as e:
            print(e)
            # トークン数の問題でエラーが発生したので，GPT-3.5 16k contextを利用して再度トライする旨を表示
            print(f"A token limit error occurred for {symbol_id.stringify()}, so we will try again with GPT-3.5 16k context.")
            # GPT-3.5 16k contextを使用して再度トライ
            try:
                chat = ChatOpenAI(temperature = 0.25, model = "gpt-3.5-turbo-16k")
                response = await asyncio.to_thread(chat, messages)
                print("Successfully generated documentation with GPT-3.5 16k context.")
            except InvalidRequestError as e:
                print(f"An error occurred and skipped generating documentation for {symbol_id.stringify()}.")
                self.document_repository.save(Document(
                    symbol_id = symbol_id,
                    dependencies = required_symbol_ids,
                    content = "",
                    succeeded = False
                ))
                return
        
        response_text = response.content

        # 4. のデバッグ用。実際にはリクエストを投げず，適当なテキストを返すようにする。
        # await asyncio.to_thread(time.sleep, 1)
        # self._counter += 1
        # response_text = f"{self._counter}テストレスポンス。これはテストです。"

        generated_document = Document(  # TODO: よしなに生成する。これ専用にファクトリメソッドを作ってもいい
            symbol_id = symbol_id,
            dependencies = required_symbol_ids,
            content = response_text,
            succeeded = True
        )
        self.document_repository.save(generated_document)
        
        print(f"[FINISHED]: Generated document for {symbol_id.stringify()}.")
        return
    

    # シンボルのドキュメント生成が可能かどうかを判定するprivateメソッド。依存先シンボルのドキュメントがすべて生成済み（または依存先がない）ことを確認する
    def _can_be_processed(self, symbol_id: ISymbolId) -> bool:
        for dependency in self._dependencies_map[symbol_id]:
            if self._progress_map[dependency] != "fulfilled":
                return False
        return True
    

    # ドキュメント生成を連鎖的に実行するprivateメソッド。
    # 受け取った`symbol_id`のドキュメントが未生成であることをチェックしてから，ドキュメントを生成
    # さらに生成後には，`symbol_id`に依存しているシンボルのうち，ドキュメント生成が可能な状態のものを再帰的に処理する。
    async def _exec_documentation_chain(self, symbol_id: ISymbolId):
        if self._progress_map[symbol_id] == "pending":
            self._progress_map[symbol_id] = "processing"
            await self._documentation(symbol_id)
            self._progress_map[symbol_id] = "fulfilled"

            symbols_to_be_processed: list[ISymbolId] = []
            for each_reference in self._reversed_dependencies_map[symbol_id]:
                if self._can_be_processed(each_reference):
                    symbols_to_be_processed.append(each_reference)
                else:
                    continue
            
            tasks = [self._exec_documentation_chain(symbol_id) for symbol_id in symbols_to_be_processed]
            await asyncio.gather(*tasks)


if __name__ == "__main__":

    async def debug():
        import os
        from documentationAI.domain.implementation.python.package_analyzer import PythonPackageAnalyzer
        from documentationAI.domain.implementation.python.module_analyzer import PythonModuleAnalyzer
        from documentationAI.domain.implementation.python.helper import PythonAnalyzerHelper
        from documentationAI.domain.implementation.python.symbol import PythonSymbolId
        from documentationAI.interfaces.repository.document_repository_impl.sqlite import SQLiteDocumentRepositoryImpl


        helper = PythonAnalyzerHelper()
        module_analyzer = PythonModuleAnalyzer(helper)
        package_analyzer = PythonPackageAnalyzer(module_analyzer, helper)
        prompt_generator = DocumentationPromptGenerator()
        class MockDocumentRepository(SQLiteDocumentRepositoryImpl):
            def __init__(self):
                super().__init__(
                    sqlite_db_path = os.path.join(os.path.dirname(__file__), "../../", "test/test_documentation_service_db.sqlite"),
                    helper = PythonAnalyzerHelper()
                )
            def save(self, document: Document) -> None:
                super().save(document)
                print("\n========saved========")
                print(document.get_content())
            def get_by_symbol_id(self, symbol_id: PythonSymbolId) -> Document|None:  # type: ignore
                return super().get_by_symbol_id(symbol_id)
        
        document_repository = MockDocumentRepository()

        documentation_service = DocumentationService(
            package_analyzer,
            document_repository,
            helper,
            prompt_generator
        )

        project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        package_root_dir = os.path.join(project_root_dir, "documentationAI")
        package_name = "documentationAI"

        await documentation_service.generate_package_documentation(project_root_dir, package_root_dir, package_name)

    asyncio.run(debug())