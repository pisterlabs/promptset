import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime
import lancedb
import pandas as pd

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def ai_completion(prompt, engine='gpt-3.5-turbo', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore' ).decode()
    while True:
        try:
            response = openai.Completion.createChatCompletion(
                model=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response[ 'choices' ][0][ 'text' ].strip()
            text = re.sub( '[\r\n]+', '\n', text )
            text = re.sub( '[\t ]+', ' ', text )
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists( 'gpt3_logs' ):
                os.makedirs( 'gpt3_logs' )
            save_file( 'gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text )
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print( 'Error communicating with OpenAI:', oops )
            sleep(1)

initialization_data = {
    'unique_id': '2c9a93d5-3631-4faa-8eac-a99b92e45d50', 
    'vector': [-0.07254597, -0.00345811,  0.038447  ,  0.025837  , -0.01153462,
         0.05443505,  0.04415885, -0.03636164,  0.04025393,  0.07552634,
         0.05359982,  0.00822271, -0.01921194,  0.09719925, -0.05354664,
         0.06897003,  0.01113722,  0.06425729,  0.04223888, -0.05898998,
        -0.01620383,  0.01389384,  0.02873985, -0.00392985, -0.02874645,
         0.02680893, -0.01051578, -0.0792539 , -0.03293172, -0.00302758,
        -0.03745122, -0.02573149, -0.00473748, -0.04199643, -0.03275133,
         0.00779039,  0.00624639,  0.06108246, -0.03870484,  0.06269313,
        -0.06609031, -0.01554973, -0.04453023, -0.00073963,  0.01021871,
        -0.02984073,  0.00474442,  0.00195324, -0.02518238, -0.00426692,
         0.00750736,  0.10541135,  0.08878568,  0.05580394, -0.01232905,
        -0.04016594,  0.04829635, -0.05689557, -0.01863352,  0.03308525,
         0.06468356, -0.03367596,  0.03575945, -0.02212196, -0.01714826,
        -0.00585904, -0.09612011, -0.00102483,  0.06920582,  0.05855923,
        -0.04266937, -0.03763324, -0.02187943, -0.00141346, -0.086646  ,
         0.02106668,  0.00786448,  0.04093482, -0.00187637,  0.02952651,
        -0.03702659, -0.02844533,  0.00322303, -0.02380866, -0.05954637,
         0.07149482, -0.0065098 ,  0.06807149, -0.00099369,  0.05040864,
         0.04761266,  0.01862198, -0.05431763,  0.00940712, -0.00970824,
        -0.02216387,  0.024306  ,  0.03772607, -0.01540066,  0.03771403,
         0.01400787, -0.09354229, -0.06321603, -0.09549774,  0.00895245,
        -0.01175102,  0.03934404,  0.00956635, -0.04152715,  0.04295438,
         0.02825363,  0.02063269,  0.02212336, -0.06888197,  0.01428573,
         0.04887657,  0.00304061,  0.03196091,  0.03902192,  0.02360773,
        -0.02807535,  0.01558309,  0.02165642,  0.01129555,  0.0567826 ,
        -0.00659211, -0.01081236,  0.01809447,  0.00318123, -0.01214105,
        -0.05691559, -0.01717793,  0.05293235,  0.01663713,  0.04678147,
        -0.02094   , -0.05482098,  0.05463412,  0.00163532,  0.00956752,
        -0.03624124, -0.02359207,  0.01571903, -0.01502842,  0.03324307,
         0.01896691,  0.02235259,  0.02551061, -0.02953271,  0.05505196,
        -0.03115846, -0.01975026, -0.05484571, -0.01757487, -0.01038232,
        -0.06098176, -0.01663185, -0.06602633, -0.00643233,  0.00167366,
        -0.04243006,  0.01024193, -0.02288529, -0.06190364,  0.03787598,
         0.03914008, -0.04915332,  0.0182827 ,  0.0136188 ,  0.02917461,
         0.03118066, -0.03110682, -0.04193405, -0.01370175, -0.03901035,
         0.00850587,  0.01056607, -0.00084098, -0.01737773,  0.00836137,
         0.01500763,  0.00917414, -0.07946376,  0.02008886,  0.04600394,
         0.01271509, -0.01654603, -0.04405601,  0.01442427,  0.00967625,
         0.01212494,  0.01189141,  0.03507042, -0.00291006,  0.04226362,
        -0.0958102 ,  0.04722575, -0.02520623, -0.00780957, -0.01983704,
        -0.02350736, -0.03137485,  0.00325953,  0.10679087, -0.08251372,
         0.02922777, -0.05723861, -0.05683867, -0.04093323, -0.04769454,
        -0.02704669, -0.04450696,  0.03854201,  0.05599346, -0.07225747,
        -0.01060745, -0.01285277, -0.02004824,  0.00567907, -0.01130959,
         0.03845671, -0.06483931, -0.00013804,  0.00342195, -0.00497795,
         0.03194252,  0.06014316,  0.07774884, -0.02778566, -0.06470748,
         0.02103901,  0.02202238,  0.02044025,  0.10802107,  0.00356093,
        -0.01817842,  0.09661267, -0.05937773, -0.08208849, -0.05190327,
        -0.0302214 ,  0.05572621, -0.06395542, -0.03078226,  0.00083952,
         0.09572925, -0.04516173, -0.0123177 ,  0.09613901, -0.05666108,
        -0.00537586,  0.04220096,  0.00019196,  0.00295547, -0.07350546,
        -0.00707971, -0.01553643, -0.05214835,  0.00311794,  0.00742682,
        -0.02943217,  0.06675503,  0.04113274, -0.0809793 ,  0.03398148,
         0.01721729,  0.03014007, -0.04178908,  0.01025263,  0.03336379,
         0.05700357,  0.10388609,  0.00663307, -0.05146715, -0.02173147,
        -0.02297893, -0.01923811,  0.03292958,  0.0521661 ,  0.03923552,
         0.01330443,  0.02524009,  0.06507587, -0.01531762, -0.04601574,
         0.0499142 ,  0.06374968,  0.06080135, -0.08060206,  0.03382473,
        -0.03596291, -0.06714796, -0.08815136,  0.02092835,  0.10282409,
         0.07779143, -0.01839681, -0.03541641,  0.00666599,  0.0029895 ,
        -0.08307225, -0.06535257,  0.01114002, -0.06142527, -0.01779631,
         0.04441926,  0.02008377,  0.03211711, -0.02073815, -0.01346437,
         0.02578364, -0.01888524,  0.03310522, -0.02017466,  0.0198052 ,
        -0.01019527, -0.02200533, -0.02650121, -0.02987311, -0.04946938,
        -0.05915657, -0.0779579 ,  0.03368903,  0.01859711,  0.02692219,
         0.04209578, -0.01279042, -0.00151735, -0.03290961,  0.00719433,
        -0.05409581,  0.04818217, -0.00339916,  0.01444317, -0.04898094,
        -0.02065373, -0.04324449, -0.01409152, -0.02882394,  0.0129813 ,
        -0.03886433, -0.08824961,  0.02457459, -0.03383131,  0.04405662,
         0.03947931,  0.02983763,  0.00124698,  0.01098392,  0.05948395,
         0.08565806,  0.02848131, -0.00725272, -0.04415287, -0.03293212,
        -0.01364554, -0.09744117, -0.05662472,  0.03124948, -0.04624591,
        -0.00605065, -0.06229377,  0.08636316, -0.03645795,  0.08642905,
         0.03093746, -0.08031843,  0.01407037,  0.09892832,  0.03219265,
         0.02964027, -0.00517425, -0.03442131, -0.01141241, -0.06644958,
        -0.07285954,  0.00890575, -0.01360151,  0.00057073, -0.08988309,
         0.00797763,  0.0176619 ,  0.00745209, -0.07096376,  0.07894821,
        -0.08301938,  0.0990236 ,  0.03789177, -0.01905026,  0.0547296 ,
        -0.06224509,  0.01964617,  0.08179896, -0.0852924 ,  0.00475453,
        -0.01451678,  0.03582037, -0.04732088, -0.041508  ,  0.05553002,
        -0.00753875, -0.02849884,  0.04659286, -0.05146529, -0.0661836 ,
        -0.00761966,  0.01581906,  0.02444271, -0.01438573, -0.03466942,
        -0.06876651, -0.02311521, -0.00312491,  0.03457906, -0.04614082,
         0.03010868,  0.0206049 ,  0.08378315, -0.03001363, -0.00827654,
         0.01580172, -0.04855691,  0.00014473, -0.01702366,  0.06371997,
         0.00924862, -0.01441237,  0.0184262 ,  0.03586025,  0.07453281,
        -0.01822053,  0.00263505, -0.07093351, -0.02956585,  0.0937797 ,
        -0.03792839,  0.03657963, -0.01717029,  0.0077794 ,  0.06886019,
         0.04470135,  0.04228634,  0.06212147, -0.05456647, -0.02041842,
         0.02251387,  0.06653161, -0.00503211,  0.03463385, -0.02718318,
         0.00118317, -0.02953942, -0.04361469,  0.01001209,  0.01472133,
        -0.07398187,  0.00152049, -0.02058817, -0.03011479, -0.03247686,
        -0.03999605,  0.00089937,  0.06058171, -0.1016895 ,  0.07500667,
         0.03293885, -0.05828201, -0.01353116,  0.06867946, -0.03266895,
        -0.02314214,  0.03284731,  0.02857622,  0.05733896,  0.05395727,
         0.06677917, -0.01256167,  0.01832761,  0.01509516,  0.08785269,
        -0.01094873, -0.09930896, -0.00904166,  0.01920987,  0.01392063,
        -0.03855692,  0.04157091, -0.05284394,  0.01217607, -0.00495155,
        -0.02351189,  0.03753581,  0.03075539,  0.0635642 ,  0.05873286,
         0.00987345,  0.05255824, -0.08698288,  0.10400596, -0.00647114,
        -0.00831464,  0.0055213 ,  0.01613558, -0.10711982,  0.00563591,
         0.03591603,  0.00221161, -0.01541905, -0.0879847 , -0.05289326,
        -0.04107964, -0.04039652],
    'speaker': 'USER', 
    'time': 1695146425.0193892, 
    'message': 'this is a test.', 
    'timestring': 'Tuesday, September 19, 2023 at 02:00PM '
}

import pyarrow as pa
class LanceTable:
    def __init__(self):
        # Initialize lancedb
        self.db = lancedb.connect( "/tmp/fresh-lancedb" )
        
        # self.schema = pa.schema([
        #     pa.field("unique_id", pa.string()),
        #     pa.field("vector", pa.list_(pa.float32())),
        #     pa.field("speaker", pa.string()),
        #     pa.field("time", pa.float64()),
        #     pa.field("message", pa.string()),
        #     pa.field("timestring", pa.string()),
        # ])


        # Create the table with the defined schema
        panda_data_frame = pd.DataFrame([ initialization_data ])
        table_name = "lance-table"
        if table_name in self.db.table_names():
            print( "table %s already exists" % table_name )
            self.db.drop_table(table_name)  # Drop the table if it already exists
            self.table = self.db.create_table( table_name, panda_data_frame )
        else:
            print( "creating table: %s" % table_name )
            self.table = self.db.create_table( table_name, panda_data_frame )
        
        # Insert the provided data into the table
        # self.table_initialized = False
        # self.table = None
        print(json.dumps(initialization_data, indent=4))
        # Ensure 'embedded_user_input' is a numpy array
        # embedded_user_input = np.array(initialization_data['vector'])

        # # Flatten the array
        # flattened_input = embedded_user_input.flatten().tolist()
        # initialization_data[ "vector" ] = flattened_input
        # dataframe = pd.DataFrame([ initialization_data ])
        # arrow_table = pa.Table.from_pandas(dataframe, panda_data_frame)

        # self.table.add( arrow_table )
        # self.table.add( dataframe )
        
    def add(self, unique_id_arg, embedded_message, speaker, timestamp, message, timestring ):
        # Ensure 'embedded_user_input' is a numpy array
        # embedded_user_input = np.array( embedded_message )

        # Flatten the array
        # flattened_input = embedded_user_input.flatten().tolist()
        # embedded_user_input = flattened_input
        
        # embedded_user_input = np.array(embedded_message['vector'])

        # Flatten the array
        # flattened_input = embedded_user_input.flatten().tolist()
        # embedded_message[ "vector" ] = flattened_input
        
        data = {
            "unique_id": unique_id_arg,
            "vector": embedded_message,
            "speaker": speaker,
            "time": timestamp,
            "message": message,
            "timestring": timestring
        }
        
        # print( data )
        dataframe = pd.DataFrame([ data ])
        # arrow_table = pa.Table.from_pandas(dataframe, panda_data_frame )

        # self.table.add( arrow_table )
        self.table.add( dataframe )
        
lanceTable = LanceTable()

import tensorflow_hub as hub
# Load the Universal Sentence Encoder
encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

if __name__ == '__main__':
    openai.api_key = open_file('/home/adamsl/linuxBash/pinecone_chat_dave_s/key_openai.txt')
    
    while True:
        # user_input = input('\n\nUSER: ')
        user_input = "hi"
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        unique_id = str(uuid4())       
        # embedded_user_input = encoder([ user_input ]).numpy() # Convert the text into vector form
        # embedded_user_input = gpt3_embedding( user_input )
        embedded_user_input = lanceTable.table.embedding_functions[ 'vector' ].function.compute_query_embeddings( user_input )[ 0 ]
        speaker   = 'USER'
        message = user_input 
        # embedded_user_input = np.array( embedded_user_input )
        # flattened_input = [float(item) for item in embedded_user_input.flatten().tolist()]

        # Insert User's input to lancedb
        lanceTable.add( unique_id, embedded_user_input, speaker, timestamp, message, timestring )

        query_builder = lanceTable.LanceVectorQueryBuilder( lanceTable.table, embedded_user_input, 'vector' )
        
        # Search for relevant message unique ids in lancedb
        # results = lanceTable.table.search( embedded_user_input ).limit( 30 ).to_df()
        
        results = query_builder.to_arrow()
        
        dataframe = results.to_pandas()
        
        print ( dataframe )
        
        chance_to_quit = input( "Press q to quit: " )
        if chance_to_quit == "q":
            break
        
        break
        
        # print ( results )
        # conversation = "\n".join(results['message'].tolist())
        
        # prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', user_input)
        # ai_completion_text = ai_completion(prompt)
        # timestamp = time()
        # timestring = timestamp_to_datetime(timestamp)
        # embedded_ai_completion = gpt3_embedding(ai_completion_text)
        # unique_id = str(uuid4())
        # speaker   = 'RAVEN'
        # thetimestamp = timestamp
        # message = ai_completion_text 
        # timestring = timestring
        
        # Insert AI's response to lancedb
        # lanceTable.table.add([( unique_id, embedded_ai_completion, speaker, timestamp, timestring )])
        
        # print('\n\nRAVEN: %s' % ai_completion_text)
