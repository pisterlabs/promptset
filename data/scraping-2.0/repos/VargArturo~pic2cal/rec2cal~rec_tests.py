import unittest
from sentence_transformers import SentenceTransformer
from rec2cal.utils import make_data, load_np_data
from os import path
import os
from rec2cal.data_paths import data_path, rep_json
import openai
from rec2cal.openai_api import OPTools
import json



class TestEmbedder(unittest.TestCase):
    
    def setUp(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def test_shape(self):
        """
        Test that the embedding utils output the right shape.
        """
        sentences = [
            "Hi I am john",
            "The end of the world is near its all over",
            "I like pizza"
        ]
        embeddings = self.model.encode(sentences)
        self.assertEqual(embeddings.shape, (3, 384))

    def test_empty(self):
        """
        Tests the empty list edge case.
        """
        embeddings = self.model.encode([])
        self.assertEqual(embeddings.shape, (0,))


class TestLoadData(unittest.TestCase):
    
    def setUp(self):
        self.train_file = path.join(data_path, "regr_model_2_vectors_train.npz")
        self.vec_X_train, self.y_train  = load_np_data(self.train_file)

    def test_shape_xy(self):
        """
        Test that the saved embeddings have matching shapes for
        X and y.
        """
        self.assertEqual(self.vec_X_train.shape[0], self.y_train.shape[0])

    def test_embed_dim(self):
        """
        Checks the final embedding dimensions are correct.
        """
        self.assertEqual(self.vec_X_train.shape[-1], 384)

        
class OpenAITools(unittest.TestCase):
    
    def setUp(self):
        our_model = "ada:ft-personal-2022-12-25-20-56-27"
        openai.api_key =  os.getenv("OPEN_AI_API_KEY")

        recipies = json.load(open(rep_json, "rb"))

        our_model = "ada:ft-personal-2022-12-25-20-56-27"

        self.openai_object_api_train = OPTools(
            recipies=recipies,
            partition="train",
            model=our_model
        )
        
        self.example = self.openai_object_api_train.dataset[0]
        
    def test_xy_type(self):
        str_1 = self.example["prompt"]
        str_2 = self.example["completion"]
 
        self.assertTrue(isinstance(str_1, str))
        self.assertTrue(isinstance(str_2, str))
   
    def test_xy_format(self):
        str_x = self.example["prompt"]
        lst_y = self.example["completion"].split(" ")
        
        self.assertEqual(len(lst_y), 4)
        cal = lst_y[-1].replace('.', '', 1).isdigit()
        
        self.assertTrue(cal)
        
        self.assertEqual(lst_y[1], "total")
        self.assertEqual(lst_y[2], "calories:")
        
        self.assertTrue("\n\n###\n\n" in str_x)
        
    def test_get_calories(self):
        prompt = self.openai_object_api_train.dataset[0]["prompt"]
        cal = self.openai_object_api_train.get_calories(prompt)
        self.assertTrue(isinstance(cal, float)  or cal is None)
        
        prompt = self.openai_object_api_train.dataset[-1]["prompt"]
        cal = self.openai_object_api_train.get_calories(prompt)
        self.assertTrue(isinstance(cal, float)  or cal is None)
        
                              
class TestMakedData(unittest.TestCase):
    
    def setUp(self):
        self.recipies = json.load(open(rep_json, "rb"))
        self.partition_keys = ['train', 'val', 'test']
        self.X_train, self.y_train  = make_data(self.recipies, "train")
        self.X_test, self.y_test  = make_data(self.recipies, "test")
        self.X_val, self.y_val = make_data(self.recipies, "val")
    
    def test_number_in_partition(self):
        self.assertEqual(len(self.y_train), 35867)
        self.assertEqual(len(self.y_val), 7687)
        self.assertEqual(len(self.y_test), 7681)
        
    def test_partitions(self):
        
        partition_keys = []
        for x in self.recipies:
            partition_keys += [x['partition']]
        
        a = set(partition_keys)
        b = set(self.partition_keys)
        
        self.assertEqual(a, b)
    
    def test_xy_type(self):
        self.assertTrue(isinstance(self.X_train[0], str))
        self.assertTrue(isinstance(self.y_test[0], float))

        self.assertTrue(isinstance(self.X_train[-1], str))
        self.assertTrue(isinstance(self.y_test[-1], float))


if __name__ == '__main__':
    unittest.main()