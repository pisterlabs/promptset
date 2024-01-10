import unittest
import json

from objective_utils import get_objective, get_embedding_fn, calculate_confidence
from agent_files.objective_defs import CMO_queries, PRO_queries

from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from agent_files.objectives import CatalogMaintenanceObjective, PeriodicRevisitObjective
from agent_files.objectives import SearchObjective, ScheduleFillerObjective
from agent_files.objectives import QualityWindowObjective, Payload, StateVector


class ObjectivesTestCase(unittest.TestCase):
    def setUp(self):
        self.embedding_fn = get_embedding_fn()
        
    def test_cmo_queries(self):
        for cmo_q in CMO_queries:
            objective, conf = get_objective(task=cmo_q,
                                      embedding_fn=self.embedding_fn)
            #print(f"{str(objective.name)} - {conf}")
            self.assertEqual(str(objective.name), "CatalogMaintenanceObjective")

    def test_pro_queries(self):
        for pro_q in PRO_queries:
            objective, _ = get_objective(task=pro_q,
                                      embedding_fn=self.embedding_fn)
            #print(f"{str(objective.name)} - {conf}")
            self.assertEqual(str(objective.name), "PeriodicRevisitObjective")

    def test_confidence(self):
        test_cosine_sims = [1, 0.999, 0.99, 0.9]
        confidence_percentage = calculate_confidence(test_cosine_sims)
        self.assertEqual(f"{confidence_percentage:.2f}", "0.07")

    def test_pydantic(self):
        test_response1 = """{
        "objective_def_name": "CatalogMaintenanceObjective",
        "end_time_offset_minutes": 20,
        "objective_name": "Catalog Maintenance Objective",
        "priority": 10,
        "sensor_name": "RME01",
        "classification_marking": "U",
        "data_mode": "TEST"
        }
        """
        parser_1 = PydanticOutputParser(pydantic_object=CatalogMaintenanceObjective)
        parsed_response1 = parser_1.parse(test_response1)
        self.assertEqual(parsed_response1.sensor_name, "RME01")
        test_response2 = """{
        "objective_def_name": "PeriodicRevisitObjective",
        "target_id": 28884,
        "sensor_name": "RME00",
        "revisits_per_hour": 2,
        "data_mode": "TEST",
        "classification_marking": "U",
        "objective_name": "Periodic Revisit Objective"
        }
        """
        parser_2 = PydanticOutputParser(pydantic_object=PeriodicRevisitObjective)
        parsed_response2 = parser_2.parse(test_response2)
        self.assertEqual(parsed_response2.target_id, 28884)
        test_response3 = """{
        "objective_def_name": "SearchObjective",
        "objective_name": "Search Objective",
        "target_id": 37775,
        "sensor_name": "RME15",
        "initial_offset": 60,
        "final_offset": 60,
        "frame_overlap_percentage": 0.3,
        "objective_start_time": "2023-08-21T18:47:19.059212",
        "objective_end_time": "2023-08-21T18:57:19.059284",
        "data_mode": "TEST",
        "classification_marking": "U"
        }
        """
        parser_3 = PydanticOutputParser(pydantic_object=SearchObjective)
        parsed_response3 = parser_3.parse(test_response3)
        self.assertEqual(parsed_response3.target_id, 37775)

        ## Testing Validators for errors
        test_err_response_1 = """{
        "objective_def_name": "SearchObjective",
        "objective_name": "Search Objective",
        "target_id": null,
        "sensor_name": "",
        "initial_offset": 60000,
        "final_offset": -100,
        "frame_overlap_percentage": -1,
        "objective_start_time": "2023-08-21T18:47:19.059212",
        "objective_end_time": "2023-08-21T18:57:19.059284",
        "data_mode": "",
        "classification_marking": ""
        }
        """
        with self.assertRaises(OutputParserException) as context:
            parsed_err_response_1 = parser_3.parse(test_err_response_1)
        test_err_2 = """{
        "objective_def_name": "PeriodicRevisitObjective",
        "target_id": "",
        "sensor_name": "",
        "revisits_per_hour": 2,
        "data_mode": "",
        "classification_marking": "",
        "objective_name": "Periodic Revisit Objective"
        }
        """
        with self.assertRaises(OutputParserException) as context:
            parsed_err_2 = parser_2.parse(test_err_2)
        test_err_3 = """{
        "objective_def_name": "CatalogMaintenanceObjective",
        "end_time_offset_minutes": 20,
        "objective_name": "Catalog Maintenance Objective",
        "priority": 10,
        "sensor_name": "",
        "classification_marking": "",
        "data_mode": ""
        }
        """
        with self.assertRaises(OutputParserException) as context:
            parsed_err_3 = parser_1.parse(test_err_3)
        ## ScheduleFillerObjective Tests
        test_response4 = """{
        "objective_def_name": "ScheduleFillerObjective",
        "objective_name": "Schedule Filler Objective",
        "sensor_name": "RME90",
        "objective_start_time": "2023-08-21T18:47:19.059212",
        "scheduling_density": 30.0,
        "data_mode": "TEST",
        "classification_marking": "U"
        }
        """
        parser_4 = PydanticOutputParser(pydantic_object=ScheduleFillerObjective)
        parsed_response4 = parser_4.parse(test_response4)
        self.assertEqual(parsed_response4.sensor_name, "RME90")
        test_err_4 = """{
        "objective_def_name": "ScheduleFillerObjective",
        "objective_name": "Schedule Filler Objective",
        "sensor_name": "",
        "objective_start_time": "2023-08-21T18:47:19.059212",
        "scheduling_density": 30.0,
        "data_mode": "",
        "classification_marking": ""
        }
        """
        with self.assertRaises(OutputParserException) as context:
            parsed_err_4 = parser_4.parse(test_err_4)
        ## QWO Tests
        state_vector = '''{
            "timestamp": "2023-07-04T00:00:00.000Z",
            "x_kilometers": 3.43988467e04,
            "y_kilometers": -2.51038896e04,
            "z_kilometers": -5.14207398e02,
            "x_dot_kilometers_per_second": 1.8,
            "y_dot_kilometers_per_second": 2.46858629,
            "z_dot_kilometers_per_second": -2.07930829e-02
            }
        '''
        payload = '''{
            "satNo": 28884,
            "priority": 2,
            "state_vector": {
                "timestamp": "2023-07-04T00:00:00.000Z",
                "x_kilometers": 3.43988467e04,
                "y_kilometers": -2.51038896e04,
                "z_kilometers": -5.14207398e02,
                "x_dot_kilometers_per_second": 1.8,
                "y_dot_kilometers_per_second": 2.46858629,
                "z_dot_kilometers_per_second": -2.07930829e-02
                },
            "window_start": "2023-07-04T22:00:00.000Z",
            "window_end": "2023-07-04T23:00:00.000Z",
            "position_accuracy": 1.0,
            "velocity_accuracy": 5
            }
        '''
        parser_state_vec = PydanticOutputParser(pydantic_object=StateVector)
        parsed_state_vec = parser_state_vec.parse(state_vector)
        self.assertEqual(parsed_state_vec.x_dot_kilometers_per_second, 1.8)
        parser_payload = PydanticOutputParser(pydantic_object=Payload)
        parsed_payload = parser_payload.parse(payload)
        self.assertEqual(parsed_payload.satNo, 28884)
        qwo_example = '''{
        "sensor_name": "RME37",
        "objective_start_time": "2023-07-04T00:00:00.000Z",
        "payload_list": [
            {
                "satNo": 28884,
                "priority": 2,
                "state_vector": {
                    "timestamp": "2023-07-04T00:00:00.000Z",
                    "x_kilometers": 3.43988467e04,
                    "y_kilometers": -2.51038896e04,
                    "z_kilometers": -5.14207398e02,
                    "x_dot_kilometers_per_second": 1.8,
                    "y_dot_kilometers_per_second": 2.46858629,
                    "z_dot_kilometers_per_second": -2.07930829e-02
                    },
                "window_start": "2023-07-04T22:00:00.000Z",
                "window_end": "2023-07-04T23:00:00.000Z",
                "position_accuracy": 1.0,
                "velocity_accuracy": 5
            }
        ],
        "data_mode": "TEST",
        "classification_marking": "U",
        "scheduling_density": 5.0,
        "hours_to_plan": 24,
        "objective_name": "Quality Window Objective",
        "objective_start_time": "2023-07-04T00:00:00.000Z",
        "objective_end_time": "2023-07-04T00:00:00.000Z",
        "priority": 1
        }'''
        parser_qwo = PydanticOutputParser(pydantic_object=QualityWindowObjective)
        parsed_qwo = parser_qwo.parse(qwo_example)
        self.assertEqual(parsed_qwo.sensor_name, "RME37")
        qwo_error = '''{
        "sensor_name": "",
        "objective_start_time": "2023-07-04T00:00:00.000Z",
        "payload_list": [
            {
                "satNo": 28884,
                "priority": 2,
                "state_vector": {
                    "timestamp": "2023-07-04T00:00:00.000Z",
                    "x_kilometers": 3.43988467e04,
                    "y_kilometers": -2.51038896e04,
                    "z_kilometers": -5.14207398e02,
                    "x_dot_kilometers_per_second": 1.8,
                    "y_dot_kilometers_per_second": 2.46858629,
                    "z_dot_kilometers_per_second": -2.07930829e-02
                    },
                "window_start": "2023-07-04T22:00:00.000Z",
                "window_end": "2023-07-04T23:00:00.000Z",
                "position_accuracy": 1.0,
                "velocity_accuracy": 5
            }
        ],
        "data_mode": "",
        "classification_marking": "",
        "scheduling_density": 5.0,
        "hours_to_plan": 24,
        "objective_name": "Quality Window Objective",
        "objective_start_time": "2023-07-04T00:00:00.000Z",
        "objective_end_time": "2023-07-04T00:00:00.000Z",
        "priority": 1
        }'''
        with self.assertRaises(OutputParserException) as context:
            parsed_err_qwo = parser_qwo.parse(qwo_error)
