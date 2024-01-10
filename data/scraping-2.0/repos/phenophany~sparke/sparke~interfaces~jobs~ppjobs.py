import os
from uuid import uuid4
import sparke.types as t

from sparke.config import System, Settings
from sparke.db.system import SysDB
from sparke.config import Component
from sparke.ingest import Producer
from sparke.tools.embedding_functions import OpenAIEmbeddingFunction


class SparkePackageJob(Component):
    _path: str
    _sysdb: SysDB
    _tenant_id: str
    _topic_ns: str
    _collection_cache: dict

    def __init__(self, system: System):
        super().__init__(system)
        self._settings = system.settings
        self._sysdb = self.require(SysDB)
        # self._manager = self.require(SegmentManager)
        # self._telemetry_client = self.require(Telemetry)
        self._producer = self.require(Producer)
        self._tenant_id = system.settings.tenant_id
        self._topic_ns = system.settings.topic_namespace
        self._collection_cache = {}

    def createsparkepackage(self):
        """Creates a SparkePackageJob object."""
        # create a hidden folder in this directory, create a sqlite database in this directory
        oe = OpenAIEmbeddingFunction(api_key='sk-xxx')



        pass

    def createtemplatefolder(self, path: str) -> dict:
        """Creates a template folder in the given path."""
        # create a folder in the given path
        # whether the folder is hidden or not is up to the user
        # os.listdir(path)
        try:
            os.mkdir(path + "/sparke", 0o777)
            os.mkdir(path + "/sparke/pdf", 0o777)
            os.mkdir(path + "/sparke/word", 0o777)
            os.mkdir(path + "/sparke/excel", 0o777)
            os.mkdir(path + "/sparke/ppt", 0o777)
            os.mkdir(path + "/sparke/other", 0o777)
            os.mkdir(path + "/sparke/multimedia", 0o777)
            os.mkdir(path + "/sparke/output", 0o777)
        except OSError:
            print("Creation of the directory %s failed" % path)

        # create a database in this directory
        _db_file = path + "/sparke/sparke.db"
        if not os.path.exists(_db_file):
            os.makedirs(os.path.dirname(_db_file), exist_ok=True)

        self._settings.persist_directory = path + "/sparke"

        self._sysdb.create_tables()

        return {
            "msg": "success",
            "content": "Template folder created successfully.The template folder is located at " + path + "/sparke",
        }


# settings = Settings(
#         # chroma_sysdb_impl="sparke.db.works.sqlite.SqliteDB",
#         # chroma_producer_impl="sparke.db.works.sqlite.SqliteDB",
#         # chroma_consumer_impl="sparke.db.works.sqlite.SqliteDB",
#         # chroma_segment_manager_impl="chromadb.segment.impl.manager.local.LocalSegmentManager",
#         allow_reset=True,
#         is_persistent=True,
#         persist_directory="d:\\test",
#     )
#
# system = System(settings)
# spp = SparkePackageJob(system)
# spp.createtemplatefolder("d:\\test")
