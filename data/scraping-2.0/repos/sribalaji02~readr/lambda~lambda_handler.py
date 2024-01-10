import requests
import json
from bs4 import BeautifulSoup
import openai
import boto3
import threading
import time
from users import *

lock = threading.Lock()
lex_client = boto3.client('lex-runtime', region_name='', aws_access_key_id='', aws_secret_access_key='')
client = boto3.client('dynamodb', region_name='', aws_access_key_id='', aws_secret_access_key='')
check = ''
temp = ''
worklist = []
def lambda_handler(event, context):
    global worklist
    global client, check, temp
    cache_ret = client.scan(
    ExpressionAttributeNames={
        '#E': 'email',
        '#B': 'isActive',
    },
    ProjectionExpression='#E, #B',
    TableName='readr_cache',
    )

    for j in range(0, cache_ret['Count']):
        if len(worklist) == 0:
            if cache_ret['Items'][j]['isActive']['BOOL'] == True:
                worklist.append(Users(cache_ret['Items'][j]['email']['S'], len(worklist)))
                continue
        elif len(worklist) >= 1:
            if cache_ret['Items'][j]['isActive']['BOOL'] == True:
                email = cache_ret['Items'][j]['email']['S']
                if email not in [u.email for u in worklist]:
                    worklist.append(Users(email, len(worklist)))

    help = ''
    
    def getBookName(user_count):
        worklist[user_count].start = 0
        worklist[user_count].end = 5
        worklist[user_count].book_ext = event['currentIntent']['slots']['book_ext']
        worklist[user_count].book_ext = worklist[user_count].book_ext.lower()
        worklist[user_count].book_name = event['currentIntent']['slots']['book_name']
        worklist[user_count].currbook.append(worklist[user_count].book_name)
        worklist[user_count].book_dict=worklist[user_count].scraped_book_dict(worklist[user_count].book_name, worklist[user_count].book_ext)
        bookname = ''
        for name in range(worklist[user_count].start, worklist[user_count].end):
            bookname += str(name+1) + ". " + str(worklist[user_count].book_dict[name]['Name'])
            bookname += "\n"
            worklist[user_count].remember_index.append(str(name))
        worklist[user_count].set_val(str(f"Here are 5 {str(worklist[user_count].book_name)} books that you'd be intrested in :\n" + bookname))
        
            
    def getNextBook(user_count):
        worklist[user_count].start += 5
        worklist[user_count].end += 5
        bookname = ''
        for name in range(worklist[user_count].start, worklist[user_count].end):
            bookname += str(name+1) + ". " + str(worklist[user_count].book_dict[name]['Name'])
            bookname += "\n"
            worklist[user_count].remember_index.append(str(name))
        worklist[user_count].set_val(str(f"Here are another 5 {str(worklist[user_count].book_name)} books that you'd be intrested in :\n" + bookname))
            
            
    def getBookLink(user_count):
        book_index=int(event['currentIntent']['slots']['book_index']) - 1
        
        if str(book_index) in worklist[user_count].remember_index:
            book_link=worklist[user_count].get_link(str(worklist[user_count].book_dict[book_index]['href']))
            worklist[user_count].set_val((f"Click any of the below link to initiate download for {str(worklist[user_count].book_dict[book_index]['Name'])} : \n" + book_link +"\nThanks for using Readr."))
            client.update_item(
                TableName='readr_downloads',
                Key={
                    'email': {
                            'S': "{}".format(worklist[user_count].get_email()),
                    }
                },
                ExpressionAttributeNames={
                    '#N': 'name',
                },
                ExpressionAttributeValues={
                    ':n': {
                        'L': [{'S': str(worklist[user_count].book_dict[book_index]['Name'])}],
                    },
                },
                UpdateExpression='SET #N = list_append(#N, :n)',
                ReturnValues="UPDATED_NEW"
            )
            client.update_item(
                TableName='readr_downloads',
                Key={
                    'email': {
                            'S': "{}".format(worklist[user_count].get_email()),
                    }
                },
                ExpressionAttributeNames={
                    '#O': 'link',
                },
                ExpressionAttributeValues={
                    ':o': {
                        'L': [{'S': str(book_link)}],
                    }
                },
                UpdateExpression='SET #O = list_append(#O, :o)',
                ReturnValues="UPDATED_NEW"
            )
        else :
            val=("Enter a valid book index number.")
            
    if event['currentIntent']['name'] == "getBookName":
        check += 'INT1 :'
        for i in range(len(worklist)):
            if worklist[i].sesid == i:
                if worklist[i].get_userid() != event['userId']:
                    worklist[i].set_userid(event['userId'])
                    break

        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                check += str(worklist[i].sesid) 
                check += ', '
                time.sleep(1)
                getBookName(i)
                return worklist[i].return_structured_response(worklist[i].get_val())
                break
            else:
                continue
          
    elif event['currentIntent']['name'] == "getNextBook":
        check += 'INT2 :'
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                check += str(worklist[i].sesid)
                check += ', '
                time.sleep(1)
                getNextBook(i)
                return worklist[i].return_structured_response(worklist[i].get_val())
                break
            else:
                continue
    
    elif event['currentIntent']['name'] == "getNextBookSet":
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                time.sleep(1)
                getNextBook(i)
                return worklist[i].return_structured_response(worklist[i].get_val())
                break
            else:
                continue
    
            
    elif event['currentIntent']['name'] == "getBookLink":
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                time.sleep(1)
                getBookLink(i)
                return worklist[i].return_structured_response(worklist[i].get_val())
                lex_client.close()
            else:
                continue
            
    elif event['currentIntent']['name'] == "elaborateBook":
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                time.sleep(1)
                book_index=int(event['currentIntent']['slots']['book_index_elab']) - 1
                return worklist[i].return_structured_response(worklist[i].gpt_response([book_index],1))
            else:
                continue
    
    elif event['currentIntent']['name'] == "compareBook":
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                time.sleep(1)
                book_one=int(event['currentIntent']['slots']['book_one']) - 1
                book_two=int(event['currentIntent']['slots']['book_two']) - 1
                return worklist[i].return_structured_response(worklist[i].gpt_response([book_one,book_two],2))
            else:
                continue
    
    elif event['currentIntent']['name'] == "getHelp":
        for i in range(len(worklist)):
            if worklist[i].get_userid() == '':
                if worklist[i].sesid == i:
                    worklist[i].set_userid(event['userId'])
                    break
            else:
                continue
        help += "1. Use 'Hi' or 'Hello' to start a conversation with Readr.\n"
        help += "2. Use 'I need a book' or 'Can you recommend some books' for eliciting request from Readr.\n"
        help += "3. Use 'Next' to switch to a series of next 5 books.\n"
        help += "4. Use 'Download book (your required book's index number)' or 'I want book (your required book's index number)' to make Readr. provide you download links for your books.\n"
        help += "5. Use 'I need some details about book (any book index)' or 'Compare books' to provide details or compare book(s) respectively.\n"
        val = help
        for i in range(len(worklist)):
            if event['userId'] == worklist[i].get_userid():
                return worklist[i].return_structured_response(help)
            else:
                continue

    return {
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': 'Fulfilled',
            'message': {
                'contentType': 'PlainText',
                'content': "Len of worklist:" + str(len(worklist)) + ", " + str(check) + "Network Error, Please try again."
            }
        }
    }