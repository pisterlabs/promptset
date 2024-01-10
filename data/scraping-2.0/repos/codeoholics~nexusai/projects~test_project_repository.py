import os
from unittest import TestCase

from aiclients import openai_client
from projects import sourcecode_service, project_repository, document_reader
from projects.document_reader import extract_text_from_file, identify_insights_from_filename, \
    extract_file_content_from_s3_url
import app_loader
from projects.project_repository import calculate_similarity_percentile
from projects.project_seeder import seed_projects_data, get_resource_folder_path
from db.vector_utils import string_to_vector

from shared import logger

log = logger.get_logger(__name__)


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")
        app_loader.init_app()

    def test_calculate_similarity(self):
        testcases = [[0, 100] ,[0.5,75],[1, 50] ,[1.5, 25],[2, 0]]

        for testcase in testcases:
            res = calculate_similarity_percentile(testcase[0],2)
            log.info(f"testcase {testcase[0]} res {res}")
            self.assertTrue(res == testcase[1])

    def test_similar_summary(self):
        # 2023-12-17 21:38:32,410 - projects.sourcecode_service - INFO - [('23', 0.0)]
        # =====================================
        # None
        # 2023-12-17 21:38:33,057 - projects.sourcecode_service - INFO - [('23', 0.8071515771874657)]

        project = project_repository.find_project_by_id("3129b55f-d3ef-48ca-bdcb-dfa065de5445")

        summary_contents = document_reader.extract_file_content_from_s3_url(project["summary_file"])

        log.info(project["summary_file"])
        # log.info(summary_contents)

        ##summary_embeddings = openai_client.create_openai_embedding(summary_contents)
        summary_embeddings = string_to_vector(summary_contents)
        result = project_repository.insert_embeddings_to_project("23", "summary", summary_embeddings)
        response = project_repository.find_similar_and_check_plagiarism("summary", summary_embeddings)
        print("=====================================")
        log.info(response)
        resources_dir = get_resource_folder_path()

        full_path = os.path.join(resources_dir, "ar.md")
        summary_contents = extract_text_from_file(full_path)
        # log.info(summary_contents)

        summary_embeddings = string_to_vector(summary_contents)
        log.info(summary_embeddings)
        response = project_repository.find_similar_and_check_plagiarism("summary", summary_embeddings)
        print("=====================================")
        log.info(response)

        # full_path = os.path.join(resources_dir, "summary5.pdf")
        summary_contents = extract_text_from_file(full_path)
        # # assert response is a map with all the keys title description institute categories theme domain
        # self.assertTrue("title" in response)
        # # title should contain home automation lower case
        # self.assertTrue("Home Automation" in response["title"])
        # self.assertTrue("description" in response)
