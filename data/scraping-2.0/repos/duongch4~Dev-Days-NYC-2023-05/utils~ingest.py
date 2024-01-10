from datasets import load_dataset_builder, load_dataset, Image
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import random
import json
from decimal import Decimal as D
from datetime import datetime
import mysql.connector
from dotenv import load_dotenv
import openai
from openai.embeddings_utils import get_embedding
import requests


load_dotenv()

S2_HOST = os.getenv('S2_HOST')
S2_USER = os.getenv('S2_USER')
S2_PASS = os.getenv('S2_PASS')
S2_DB = os.getenv('S2_DB')
openai.api_key = os.getenv('OPENAI_API_KEY')

def init_connection():
    return mysql.connector.connect(user=S2_USER, password=S2_PASS, host=S2_HOST, database=S2_DB)

dataset = "katanaml-org/invoices-donut-data-v1"
cache_path = 'static/datasets' # relative cache path to utils/ingest.py file
pmt_status_list = ['PIF', 'PARTIAL', 'NA']
pmt_method = ['CASH', 'CREDIT', 'WIRE']

ds = load_dataset(dataset, split='train', cache_dir=cache_path).cast_column("image", Image(decode=False))

# print(ds[0]['image'])

def lookup_invoice_no(invoice_no):
    if invoice_no in db:
        return True
    else:
        return False

def load_item_as_is(invoice):
    if invoice['collected'] == True:
        invoice['collected_amt'] == ''

def load_item_modified(invoice):
    invoice['collected_amt'] == ''

def reset_modifier_counter():
    modifier_counter = 0
    return modifier_counter

def parse_item_rebuild(invoice):
    parsed_invoice = {}
    # if invoice['image']:
    #     print(invoice['image'])
    #     parsed_invoice['image'] = invoice['image']['bytes']
    inv_dump = json.loads(invoice['ground_truth'])
    print(inv_dump)
    if inv_dump['gt_parse'].keys() & {'header', 'items', 'summary'} and inv_dump['gt_parse']['header'].keys() & {'invoice_no', 'invoice_date'}:
        new_header = {}
        new_items = []
        new_summary = {}
        orig_header = inv_dump['gt_parse']['header']
        orig_items = inv_dump['gt_parse']['items']
        
        if 'invoice_no' in orig_header:
            if inv_dump['gt_parse']['header']['invoice_no'] in ['16692901', '45181097', '38557956', '17545286']:
                return 'skipped'
            for key in orig_header:
                if key == 'invoice_date':
                    reformat_str = orig_header['invoice_date'].replace('/', '-')
                    date_obj = datetime.strptime(reformat_str, '%m-%d-%Y').date()
                    new_header['invoice_date'] = str(date_obj)
                    print(new_header['invoice_date'])
                else:
                    new_header[key] = orig_header[key]
        else:
            new_header['invoice_no'] = str(random.randrange(00000000,99999999))
            for key in orig_header:
                if key == 'invoice_date':
                    reformat_str = orig_header['invoice_date'].replace('/', '-')
                    date_obj = datetime.strptime(reformat_str, '%m-%d-%Y').date()
                    new_header['invoice_date'] = str(date_obj)
                    print(new_header['invoice_date'])
                else:
                    new_header[key] = orig_header[key]

        if new_header['invoice_no'] == 'Date':
            new_header['invoice_no'] = str(random.randrange(00000000,99999999))

        for i in orig_items:
            if 'item_qty' in i:
                new_i = {}
                #print('item: {}'.format(i))

                if 'item_desc' in i:
                    new_i['item_desc'] = i['item_desc']
                
                if 'item_qty' in i:
                    new_i['item_qty'] = i['item_qty'].replace(',', '.').replace('.00', '')

                if not 'item_net_worth' in i:
                    if 'total_net_worth' in i:
                        new_i['item_net_worth'] = i['total_net_worth'].replace(' ', '').replace(',', '.')
                    else:
                        new_i['item_net_worth'] = str(float(new_i['item_qty']) * float(i['item_net_price'].replace(' ', '').replace(',', '.')))

                elif 'item_net_worth' in i:
                    new_i['item_net_worth'] = i['item_net_worth'].replace(' ', '').replace(',', '.')
                else:
                    print('No item net worth')
                    exit(1)

                if not 'item_net_price' in i:
                    item_nw = i['item_net_worth'].replace(' ', '').replace(',', '.')
                    if int(new_i['item_qty']) >= 1: 
                        new_i['item_net_price'] = str(float(item_nw) / float(new_i['item_qty']))
                    else:
                        new_i['item_net_price'] = i['item_net_worth']
                else:
                    new_i['item_net_price'] = i['item_net_price'].replace(' ', '').replace(',', '.')
                
                
                if 'item_vat' in i:
                    new_i['item_tax'] = i['item_vat']
                
                if 'item_gross_worth' in i:
                    new_i['item_gross_worth'] = i['item_gross_worth'].replace(' ', '').replace(',', '.')
                elif 'total_gross_worth' in i:
                    new_i['item_gross_worth'] = i['total_gross_worth'].replace(' ', '').replace(',', '.')

                new_items.append(new_i)
        if 'summary' in inv_dump['gt_parse']:
            orig_summary = inv_dump['gt_parse']['summary']
            for key in orig_summary:
                if key == 'total_net_worth':
                    new_summary[key] = orig_summary[key].replace(' ', '').replace(',', '.').replace('$', '')
                elif key == 'total_vat':
                    new_summary['total_tax'] = orig_summary[key].replace(' ', '').replace(',', '.').replace('$', '')
                elif key == 'total_gross_worth':
                    new_summary['total_gross_worth'] = orig_summary[key].replace(' ', '').replace(',', '.').replace('$', '')
            if 'total_gross_worth' not in orig_summary:
                new_summary['total_gross_worth'] = str(float(new_summary['total_net_worth']) + float(new_summary['total_tax']))
            if 'total_net_worth' not in orig_summary:
                new_summary['total_net_worth'] = str(float(new_summary['total_gross_worth']) - float(new_summary['total_tax']))
            if 'total_tax' not in orig_summary:
                new_summary['total_tax'] = str(float(new_summary['total_gross_worth']) - float(new_summary['total_net_worth']))
        else:
            items_total = 0.00
            tax_total = 0.00
            for i in new_items:
                items_total = items_total + float(i['item_net_worth'])
                if i['item_tax']:
                    tax_perc = D(i['item_tax'].replace('%', '')) / 100
                    item_tax_amt = float(i['item_net_worth']) * float(tax_perc)
                    tax_total = tax_total + item_tax_amt
            new_summary['total_net_worth'] = str(round(items_total, 2))
            new_summary['total_tax'] = str(round(tax_total, 2))
            new_summary['total_gross_worth'] = str(float(new_summary['total_net_worth']) + float(new_summary['total_tax']))
            print(new_summary)

        if new_summary['total_gross_worth'] == 'Total':
            new_summary['total_gross_worth'] = str(float(new_summary['total_net_worth']) + float(new_summary['total_tax']))
        
        parsed_invoice['header'] = new_header
        parsed_invoice['items'] = new_items
        parsed_invoice['summary'] = new_summary
        return parsed_invoice
    else:
        print('missing keys, skipping')
        return 'skipped'

def insert_invoice(cursor,parsed_invoice):
    pi = parsed_invoice
    print(pi)
    print(type(pi))
    query = ("INSERT INTO accounts_receivable "
             "(invoice_no, invoice_date, client, total_invoice_before_tax, total_invoice_tax, total_invoice_including_tax) "
             "VALUES (%(invoice_no)s, %(invoice_date)s, %(client)s, %(total_invoice_before_tax)s, %(total_invoice_tax)s, %(total_invoice_including_tax)s)")

    data = {
        'invoice_no': pi['header']['invoice_no'],
        'invoice_date': pi['header']['invoice_date'],
        'client': pi['header']['client'],
        'total_invoice_before_tax': pi['summary']['total_net_worth'],
        'total_invoice_tax': pi['summary']['total_tax'],
        'total_invoice_including_tax': pi['summary']['total_gross_worth']
    }
    print(pi['header']['invoice_no'])
    cursor.execute(query,data)

def insert_payment(cursor,parsed_invoice):
    pi = parsed_invoice
    query = ("INSERT INTO payments "
             "(invoice_no, pmt_date, pmt_type, pmt_amt) "
             "VALUES (%(invoice_no)s, %(pmt_date)s, %(pmt_type)s, %(pmt_amt)s)")
    rand_status = random.choice(pmt_status_list)
    pmt_date = datetime.today().strftime('%Y-%m-%d')
    if rand_status == 'PIF':
        pmt_amt = pi['summary']['total_gross_worth']
        pmt_type = random.choice(pmt_method)
    elif rand_status == 'PARTIAL':
        pmt_amt = round(random.uniform(0.00,float(pi['summary']['total_gross_worth'])))
        pmt_type = random.choice(pmt_method)
    else:
        print('No payment this time, skipping')
        query_update = ("UPDATE accounts_receivable "
                    "SET pmt_status = %(pmt_status)s"
                    "WHERE invoice_no = %(invoice_no)s")

        data_update = {
            'invoice_no': pi['header']['invoice_no'],
            'pmt_status': 'NA'
        }

        cursor.execute(query_update,data_update)
        return
    
    data = {
        'invoice_no': pi['header']['invoice_no'],
        'pmt_date': pmt_date,
        'pmt_type': pmt_type,
        'pmt_amt': pmt_amt
    }

    cursor.execute(query,data)

    query_update = ("UPDATE accounts_receivable "
                    "SET pmt_status = %(pmt_status)s"
                    "WHERE invoice_no = %(invoice_no)s")

    data_update = {
        'invoice_no': pi['header']['invoice_no'],
        'pmt_status': rand_status
    }

    cursor.execute(query_update,data_update)

def create_embeddings(cursor,parsed_invoice):
    pi = parsed_invoice
    query = ("INSERT INTO invoice_embeddings "
             "(invoice_no, text, vector) "
             "VALUES (%(invoice_no)s, %(text)s, JSON_ARRAY_PACK(%(vector)s)")
    
    embedding = json.dumps(get_embedding(str(pi), engine="text-embedding-ada-002"))
    print(embedding)
    data = {
        'invoice_no': pi['header']['invoice_no'],
        'text': str(pi),
        'vector': embedding[0]
    }

    cursor.execute(query, data)

def insert_to_db(cursor,parsed_invoice):
    inv_result = insert_invoice(cursor, parsed_invoice)
    print(inv_result)
    pmt_result = insert_payment(cursor, parsed_invoice)
    print(pmt_result)
    embed_result = create_embeddings(cursor, parsed_invoice)
    print(embed_result)

def main(ds):
    conn = init_connection()
    cursor = conn.cursor()
    for d in ds:
        parsed_invoice = parse_item_rebuild(d)
        if parsed_invoice != 'skipped':
            insert_to_db(cursor,parsed_invoice)
            conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main(ds)