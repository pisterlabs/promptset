from django.test import TestCase
from django.db import connections
from ..models.VectorEngine import VectorEngine
from ..models.KBitem import * 
from ..models.Query import *
import psycopg2
from dotenv import load_dotenv
import os
import statistics
from langchain.vectorstores.pgvector import PGVector
load_dotenv()
from django.core.management import call_command
import json
import uuid


class testQuery(): 
    '''
    Basic class to encapsulate a URI we intend to match and the queries we will use to try and match it 
    '''
    def __init__(self, URI, releventQueries):
        self.URI = URI
        self.relevantQueries = releventQueries


class UUIDEncoder(json.JSONEncoder):
    '''
    Class to help with dumping and loading UUID data to/from json files
    '''
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

class Helper: 
    '''
    Helper class used to encapsulate basic helper functions used by all tests
    '''

    mainDBConnection = psycopg2.connect(
        host="localhost",
        database = 'Athena',
        user = 'postgres',
        password = os.environ.get("PGVECTOR_PASSWORD", "postgres")
        ) 
    

    def emptyVectorDB(): 
        '''
        Method to empty all test items from the pg vector database. Langchain stores all embeddings for different collections in the same table (our production table) so we 
        must clear all items from tests from the table when tests are complete" 
        '''
        cursor = Helper.mainDBConnection.cursor()
        sql_empty_query = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s"
        cursor.execute(sql_empty_query, ("a5cc26f9-a46b-41d6-938c-d748fed2487f",))
        Helper.mainDBConnection.commit()
        cursor.close()
    

    def createVectorEngine(collectionName): 
        '''
        Method to create a VectorEngine instance that connects to the test pg vcector database with select embedding functions 
        '''
        vectorEngine = VectorEngine(collectionName)
        vectorEngine.vectorDatabase = PGVector(
                collection_name=vectorEngine.vectorCollectionName,
                connection_string=vectorEngine.CONNECTION_STRING,
                embedding_function=vectorEngine.embeddings,
        )

        return vectorEngine
    
    def dumpKBItemDB(file): 
        '''
        Method to dump test KBItem table to json file
        '''
        with open(file, 'w') as f: 
            call_command('dumpdata', 'kbItems', '--indent', '4', stdout=f)
    
    def loadKBItemDB(file): 
        '''
        Load test KBItem json file to test database 
        '''
        call_command('loaddata','/Users/andrewpassero/Documents/Athena/' + file)

    def loadTestVectorDB(file):
        '''
        Load json file containing PGVector embeddings to the test database 
        '''
        # Read the data from the JSON file
        with open(file, 'r') as f:
            dict_rows = json.load(f)

        # Get a cursor from the database connection
        cursor = Helper.mainDBConnection.cursor()

        for row in dict_rows:
            # Convert the UUID strings back to UUID objects
            row['collection_id'] = uuid.UUID(row['collection_id'])
            row['uuid'] = uuid.UUID(row['uuid'])

            # Convert dict to JSON string
            if 'cmetadata' in row and isinstance(row['cmetadata'], dict):
                row['cmetadata'] = json.dumps(row['cmetadata'])
            
            # Build the INSERT query
            columns = ', '.join(row.keys())
            placeholders = ', '.join(['%s'] * len(row))
            query = f"INSERT INTO langchain_pg_embedding ({columns}) VALUES ({placeholders})"

            # Execute the INSERT query
            cursor.execute(query, list(row.values()))

        # Commit the transaction
        Helper.mainDBConnection.commit()

        # Close the cursor
        cursor.close()

    
    def saveTestVectorDB(file): 
        cursor = Helper.mainDBConnection.cursor()
        cursor.execute("SELECT * FROM langchain_pg_embedding WHERE collection_id = 'a5cc26f9-a46b-41d6-938c-d748fed2487f'")
        # cursor.execute("SELECT * FROM langchain_pg_embedding WHERE collection_id = '947d59e7-cc04-44b8-8521-b15c66f10bf5'")
        rows = cursor.fetchall()

        column_names = [desc[0] for desc in cursor.description]

        dict_rows = [dict(zip(column_names, row)) for row in rows]

        with open(file, 'w') as f:
            json.dump(dict_rows, f, cls=UUIDEncoder)  # Use the custom encoder

        cursor.close()
    


class ParseDocumentText(TestCase): 
    '''
    Test case for scraping document content from url
    '''
    def setUp(self): 
        '''
        Load KBItem table containing document URLS and empty content
        '''
        Helper.loadKBItemDB('BaselineTestDBs/kbitemsBaseline.json')
        
        for obj in KBItem.objects.all(): 
            obj.userTags = ""
            obj.itemContent = ""
        

    def test_parse(self):
        '''
        Assign each item to correct child class, parseURL and confirm content exists, save to kbItem table
        '''
        for item in KBItem.objects.all(): 
            URI = item.URI
            id = item.id 
            item.refresh_from_db()
            if hasattr(item,'imagekbitem'):
                item = ImageKBItem(URI = URI, id = id)
                print("Testing " + item.URI + " as an ImageItem")
            elif hasattr(item,'textkbitem'):
                item = TextKBItem(URI = URI, id = id)
                print("Testing " + item.URI + " as an TextImage")
            else: 
                print("Testing " + item.URI + " NON TYPE")
                break 
            
            item.parseURI() 

            try: 
                self.assertGreater(len(item.itemContent),0)
                print(item.itemContent)
                print("\n" + item.userTags)
            except AssertionError: 
                print("Kbitem ID: " + str(item.id) + " Failed to Parse")

            item.save() 
            print("")
    
    def tearDown(self) -> None:
        Helper.dumpKBItemDB('kbitemsTESTS.json')
    


class EmbedModelTest(TestCase): 
    '''
    Test case for embedding document text content to a PGVector table
    '''
    def setUp(self): 
        self.newVectorEngine = Helper.createVectorEngine(collectionName="VectorTest")
        # self.newVectorEngine = Helper.createVectorEngine(collectionName="KBitem")
        Helper.loadKBItemDB('BaselineTestDBs/kbitemsBaseline.json')
    
    def queryVectorByID(self,itemID):
        '''
        Helper method to retreive items from PGVector table with kbItem ID
        '''
        cursor = Helper.mainDBConnection.cursor()
        sql_query = """SELECT * FROM langchain_pg_embedding 
                    WHERE collection_id = %s 
                    AND cmetadata->>'kbItemID' = %s"""
        cursor.execute(sql_query, ("a5cc26f9-a46b-41d6-938c-d748fed2487f", itemID))
        # cursor.execute(sql_query, ("947d59e7-cc04-44b8-8521-b15c66f10bf5", itemID))

        rows = cursor.fetchall()
        cursor.close() 

        return rows 



    def testEmbedItems(self,chunk_size = 100): 
        '''
        Test to save KBItem document data to a vector and ensure it is properly saved 
        '''
        Helper.emptyVectorDB()
        for item in KBItem.objects.all(): 
            item.vectorEngine = self.newVectorEngine
            item.createVector(chunk_size=chunk_size)


            rows = self.queryVectorByID(str(item.id)) 

            try: 
                self.assertGreater(len(rows), 0)
                print("SUCCESS VECTORS RETRIEVED")
            except AssertionError: 
                print("FAILED TO FIND SAVED EMBEDDINGS")
            
            retrievedID = (rows[0][-3])['kbItemID']

            try: 
                self.assertEquals(retrievedID, item.id)
                print("SUCCESS VECTORS MATCH IDS")
            except AssertionError: 
                print(" FAILED RETURNED NON MATCHING ITEM " +str(item.id) + " != " + str(retrievedID) )

        
    def tearDown(self) -> None:
        Helper.saveTestVectorDB("VectorDBTEST.json")
        Helper.emptyVectorDB()
    

class QueryPerformanceTest(TestCase): 
    '''
    Test case to evaulate search performance. All URI's in the test set are given custom queries that should retrieve the given query from the databse.
    This "needle in a haystack test" allows us to determine the accuracy for which the model returns the subjectively desired document. This data set consists
    of test cases that are similar and should class. 

    Accuracy as of 7/28/23 is around 55% for top matching item. 
    
    NOTE: There are certainly generic queries that should will match similar documents and lead to a decrease in accuracy 
    '''
    
    def setUp(self): 
        '''
        Hand created queries for each URL in the kbItems test set
        '''
        self.allURIs = []
        
        URI = "https://twitter.com/edkrassen/status/1679971231280365568?s=12&t=luE0t2rXCmHkF-x3pwIZ9g"
        queries = ["xAI", "artificial Intelligence", "elon musk new company", "understand the universe", "Twitter new company name"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/Curj4UXgKQr/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["fix slice", "golf drills to fix slice", "golf close club face", "rotate the club face", "golf drills"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CujAuYyAZ20/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["ChatGPT", "artifical intelligence", "code interpretation", "chatgpt predict stock prices"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CubW3EQg3y6/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["ChatGPT", 'artificial intelligence', "chatgpt up to date data", "LLM search web"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CurVzXtgMfD/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["fix slice", "golf drills to fix slice", "golf drills", "fix early turn"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://thestayathomechef.com/how-to-cook-steak/"
        queries = ["recipes", "how to cook steak", "cooking", "cook steak"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CuuH982gUeM/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["men and women", "relationships", "men express what they think", 'relationship communication']
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CuUlf0TO_wJ/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["men and women", "relationships", "things to talk about with partner", "questions to ask partner"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CsRmI9QtN-z/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["recipes", "cooking", "chicken burrito", "how to cook chicken", "chicken meals"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/Cu2XZsvrw4r/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["kids study", "reward for effort no talent", "effort is better than intelligence"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CupiyZBMgOK/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["recipes", "sandwhich recipe", "chicken bacon ranch", "things to cook", "cooking"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/Cu2D3CyrGHl/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["chicken burritos", "low calories meal", "recipes", "cooking", "things to cook"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CvAtU0HRrpk/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "rolling golf club", "club face path", "fix golf swing"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://twitter.com/MarioNawfal/status/1679878494963023876?s=20"
        queries = ["META twitter", "twitter competitor", "user activity on threads", "meta rival"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/Cthyx1cuRAc/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "getting stuck at impact"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)
    
        URI = "https://twitter.com/GuyDealership/status/1678533370958032896?s=20	"
        queries = ["car supply","lack of financing", "high interest rates", "hard to get credit"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CrrdZjrNc-_/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "golf set up", "trail elbow","allow elbow to bend"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://twitter.com/WallStreetSilv/status/1679588837301923841?s=20"
        queries = ["blockchain", "bitcoin", "centralized digital currency", "government freeze funds"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CvFl3vyR-JO/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "slice fix", "get rid of slice", "hit ball straighter"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CsPAreZuh4T/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "lower body movement", "roate hips", "faster club speed"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CszDLDdsa_u/?igshid=MTc4MmM1YmI2Ng%3D%3D"
        queries = ["golf drills", "slice fix", "get rid of slice", "swing in to out", "hit straight ball"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.healthline.com/health/fitness-exercise/best-chest-exercises"
        queries = ["workouts", "chest workouts", "chest day excercises", "bench press", ]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.verywellfit.com/20-great-exercises-to-work-your-shoulders-1231032"
        queries = ["workouts", "shoulder workouts", "shoulder day excercises", "shoulder press", "arm workout"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.healthline.com/health/fitness/leg-workout"
        queries = ["workouts", "leg workouts", "leg day excercises", "squat", "quad excercises"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CsYXeFGrYV6/?igshid=MTc4MmM1YmI2Ng%3D%3D	"
        queries = ["golf drills", "shallow golf club", "golf swing thoughts", "move hands down"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        URI = "https://www.instagram.com/reel/CrHGvy4uPHM/?igshid=MTc4MmM1YmI2Ng%3D%3D	"
        queries = ["fishing", "catching carp", "how to catch fish"]
        URI = testQuery(URI, queries)
        self.allURIs.append(URI)

        # URI = ""
        # queries = []
        # URI = testQuery(URI, queries)
        # self.allURIs.append(URI)

        
        Helper.loadKBItemDB("BaselineTestDBs/kbitemsBaseline.json")
        Helper.loadTestVectorDB("BaselineTestDBs/VectorDBBaseline.json")
        self.newVectorEngine = Helper.createVectorEngine(collectionName="VectorTest")

    
    def test_performance(self):
        '''
        Iterate through every query for every URI in the test set, do a similarity search and determine if the top matching item matches the URI 
        the test was looking for. 
        '''
        totalTestCases = 0 
        correctTetCases = 0 
        for resourceQueries in self.allURIs: 
            print("\nTESTS FOR URI: " + resourceQueries.URI)
            matchScores = [] 
            for query in resourceQueries.relevantQueries: 
                totalTestCases += 1
                qObject = Query(query)
                qObject.vectorEngine = self.newVectorEngine
                sortedKbItems = qObject.getMatchedDocs()
                topMatchItem = sortedKbItems[0]

                topMatchScore = topMatchItem[1]
                topMatchURI = topMatchItem[0].URI.strip()

                try:
                    self.assertEqual(topMatchURI, resourceQueries.URI.strip())
                    print("QUERY STRING: " + query + " SCORE = " + str(topMatchScore) + " Next Closest Score: " + str(sortedKbItems[1][1] if len(sortedKbItems)>1 else 0))
                    matchScores.append(topMatchScore)
                    correctTetCases += 1

                except AssertionError:
                    print("QUERY STRING: " + query + " FAIL: MATCHED " + topMatchURI+ " SCORE = " + str(topMatchScore) + " Next Closest Score: " + str(sortedKbItems[1][1] if len(sortedKbItems)>1 else 0) + "Matching " + sortedKbItems[1][0].URI)

        accuracy = correctTetCases / totalTestCases * 100 
        print(round(accuracy,2))
    
    def tearDown(self):
        Helper.emptyVectorDB()


    
        



            
            

    
    
