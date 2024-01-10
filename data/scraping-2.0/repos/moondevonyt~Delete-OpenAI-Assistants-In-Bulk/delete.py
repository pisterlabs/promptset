from openai import OpenAI 
import time 
import schedule

client = OpenAI(api_key='YOUR KEY HERE BRO')

# list all of our assistants 
def list_assistants(client, limit=100):
    return client.beta.assistants.list(order='desc', limit=str(limit))

# function that deletes and assistant
def delete_assistant(client, assistant_id):
    try:
        response =client.beta.assistants.delete(assistant_id=assistant_id)
        print(f'Deleted: {assistant_id} thanks moon!')
        return response 
    except Exception as e:
        print(f'errror delteing {assistant_id}: {e}')

# what about saving some assistants
do_not_delete_ids = {
    'asst_XovpYGPkfxslvkhsBjigeWE2',
    'asst_lMBUsMM13krUpgRWOp4KUn13',
    'asst_t6gTygM3lVvAeWXuul4YaTuA',
    'sjklsjfdskj'
}

def bot():
    # get the list of assistants 
    my_assistants = list_assistants(client)

    # delete all of the assistants that are not in the do_not_delete_ids set
    for assistant in my_assistants.data:
        if assistant.id not in do_not_delete_ids:
            delete_assistant(client, assistant.id)
            time.sleep(.2)
        else:
            print(f'Not deleting: {assistant.id}')


schedule.every(2).seconds.do(bot)   

while True:
    try:
        schedule.run_pending()
    except:
        print('+++++ maybe an internet problem.. code failed. sleeping 10')
        time.sleep(10)