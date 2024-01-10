from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

DriveFolder = {'bill':'1rJHRewitYJ1luUIrOSYXVz1sJIAKtF8F','product':'120Dq0RsLV2IrwMLiErnILVGURbeONlS1','invoice':'1QT7ipOqopPn8Ri_ZgZjYnjPx25qc2yUp'}
SpreadSheet = {'finance':'1pRqwos74q7zpk1kGc6DB_ZpzWRQ0OKQjzqg0-moPy48','materials':'1rv5koHjoPW9fm2Y8ql4QVLBTz6O5mRGnah2IS6n1NYM','products':'1JiNGR9dn-FEkAG13mUND6DEVijpcc8IuGFLbp78Iuz8','designs':'18XI5XZj4JjzvVMCyF6qyAv9ggQHSa2b7kbAhiDMFd5M'}
#Scopes = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
Scopes = ['https://www.googleapis.com/auth/drive']

SAMPLE_SPREADSHEET_ID = '1Biing8a_2ZqOPQd4k1Vec00rANom83oZtraQJT-Imfc'
SAMPLE_RANGE_NAME = 'Cash!A1:G'
EDIT_RANGE_NAME = 'Cash!A1:E'

adminID = '1651529355'  # Pallab Chat-ID

import asyncio
import time
import datetime
import pickle
from tqdm.contrib.telegram import tqdm, trange
import telegram
#from telegram import constants, Update, KeyboardButton, ReplyKeyboardMarkup, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
#from telegram.ext import ContextTypes, Updater, InlineQueryHandler, CommandHandler, MessageHandler, filters, CallbackContext, Application
#from telegram.constants import ParseMode
from telegram.ext import Updater, InlineQueryHandler, CommandHandler, MessageHandler, Filters
from telegram import ParseMode
import threading
import subprocess
from subprocess import PIPE, Popen
import numpy as np
import os
import openai
import cv2
from pyzbar.pyzbar import decode
import qrcode
import copy
from PIL import Image
openai.api_key_path = "./OpenAPI.token"


msg = """
Hi! Pallab have set me up for updating you about your House renovation account.
                                                - gadhuBot

You can type the following commands to get the corresponding updates:
    /start -> shows this text
    /balance -> shows current balance
    /transactions N -> shows last N number of transactions
    /credit amount remark -> add credit, update everyone
    /debit amount remark -> add debit, update everyone
    /clear -> delete chat history
"""

def envrun(env,script,argu):
    command = f"conda run -n {env} python {script} {argu}"
    return subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)

def remove_nested_keys(dictionary, key, child):
    if key in dictionary:
        del dictionary[key][child]

    for value in dictionary.values():
        if isinstance(value, dict):
            remove_nested_keys(value, key, child)

    return dictionary

def gencode(otype,keys,bits=0):
    if otype=="material":
        fbrc = keys[0]
        namelist = fbrc.split("-")
        if bits==0:
            cdn = ''.join([i[0].upper() for i in namelist])
        else:
            ix = bits%len(namelist)-1
            jx = bits//len(namelist)+2
            if ix==-1:
                ix = len(namelist)-1
                jx = jx-1
            cdn = ''.join([i[0].upper() for i in namelist])
            if jx<len(max(namelist)):
                cdn = cdn[:ix]+namelist[ix][:jx].capitalize()+cdn[ix+1:]
            else:
                cdn = ''.join([i[0].upper() for i in namelist])
                cdn = cdn+'X'*(jx-1)
    if otype=="product":
        matCode, prodName = keys
        matcdn = matCode[:matCode.rfind('-',0,matCode.rfind('-'))]
        frh = open('prodtype.list','rb')
        prodArr = np.array(pickle.load(frh))
        frh.close()
        sid = np.flatnonzero(prodArr[:,1]==prodName)[0]
        prodID = str(prodArr[sid,0])
        prodHRK = getProd_hrk(prodArr, prodID).split(' ')[:-1]
        frh = open('prodtype.dict','rb')
        prodDict = pickle.load(frh)
        subDict = prodDict
        frh.close()
        for prd in prodHRK:
            subDict = subDict[prd]
        keyID = list(subDict.keys()).index(prodName)
        cdn = ''.join([i[0].upper() for i in prodHRK])+str(keyID)+'-'+matcdn
    return cdn

def getProd_hrk(prodArray, crntID):
    if crntID=='0':
        return "outfit"
    else:
        sid = prodArray[:,0] == str(crntID)
        crntNAME = prodArray[sid,1][0].lower()
        return getProd_hrk(prodArray, prodArray[sid,2][0]) + ' ' + crntNAME

def get_spreadsheet(creds,getPOS,getID):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=getID,
                                    range=getPOS).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
            return

        print('Name, Major:')
        return values
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s' % (row[0], row[0]))
    except HttpError as err:
        print(err)

def get_index(creds,getPOS,getID):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        #result = sheet.values().get(spreadsheetId=getID,
        #                            range=getPOS).execute()

        result = service.spreadsheets().values().batchGet(
            spreadsheetId=getID, ranges=getPOS).execute()
        
        #values = result.get('values', [])
        values = result.get('valueRanges', [])

        #if not values:
        #    print('No data found.')
        #    return

        print('Name, Major:')
        print(values)
        return values
        #for row in values:
        #    print(row)
            # Print columns A and E, which correspond to indices 0 and 4.
            #print('%s, %s' % (row[0], row[0]))
    except HttpError as err:
        print(err)

def put_spreadsheet(creds,putLIST,putID,putPOS,write='append'):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    try:
        service = build('sheets', 'v4', credentials=creds)
        values = [putLIST]
        body = {'values': values}
        # Call the Sheets API
        sheet = service.spreadsheets()
        if write=='append':
            result = sheet.values().append(spreadsheetId=putID,
                                    range=putPOS, valueInputOption='USER_ENTERED', body=body).execute()
        else:
            result = sheet.values().update(spreadsheetId=putID,
                                    range=putPOS, valueInputOption='USER_ENTERED', body=body).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
            return

        print('Name, Major:')
        return values
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s' % (row[0], row[0]))
    except HttpError as err:
        print(err)

def upload_to_folder(creds,fileName,filePath,parent_id,permit='reader'):
    """Upload a file to the specified folder and prints file ID, folder ID
    Args: Id of the folder
    Returns: ID of the file uploaded

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {
            'name': fileName,
            'parents': [parent_id]
        }
        #media = MediaFileUpload(filePath,
        #                        mimetype='image/jpeg', resumable=True)
        media = MediaFileUpload(filePath,resumable=True)
        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata, media_body=media,
                                      fields='id').execute()
        user_permission = {
            'type': 'anyone',
            'role': '%s'%(permit)
        }
        file_id = file.get('id')
        service.permissions().create(fileId=file_id,
                                               body=user_permission,
                                               fields='id').execute()
        print(F'File ID: "{file.get("id")}".')
        return file.get('id')

    except HttpError as error:
        print(F'An error occurred: {error}')
        return None

def create_folder(creds,dirName,parent_id):
    """ Create a folder and prints the folder ID
    Returns : Folder Id

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {
            'name': dirName,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }

        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata, fields='id').execute()
        user_permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        file_id = file.get('id')
        service.permissions().create(fileId=file_id,
                                               body=user_permission,
                                               fields='id').execute()
        print(F'Folder ID: "{file.get("id")}".')
        return file.get('id')

    except HttpError as error:
        print(F'An error occurred: {error}')
        return None

def delete_file(creds, file_id):
    """Permanently delete a file, skipping the trash.

    Args:
        service: Drive API service instance.
        file_id: ID of the file to delete.
    """
    try:
        service = build('drive', 'v3', credentials=creds)
        service.files().delete(fileId=file_id).execute()
    except HttpError as error:
        print(F'An error occurred: {error}')

#async def launch_web_ui(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#    await update.message.reply_text(
#        "Material upload",
#        reply_markup=InlineKeyboardMarkup.from_button(
#            InlineKeyboardButton(
#                text="Open Form",
#                web_app=WebAppInfo(url="https://antant-boutique.github.io/Accountant/matform?material=CTN&size=10&price=20&"),
#            )
#        ),
#    )
    # For now, we just display google.com...
    #kb = [[InlineKeyboardButton("Open Form", web_app=WebAppInfo("https://antant-boutique.github.io/Accountant/matform?material=CTN&size=10&price=20&"))]]
    #await update.message.reply_text("Let's do this...", reply_markup=ReplyKeyboardMarkup(kb))
    #reply_markup = InlineKeyboardMarkup(kb)
    #update.message.reply_text('Click the button to open form:', reply_markup=reply_markup)

#async def web_app_data(update: Update, context: CallbackContext):
#    """Print the received data and remove the button."""
    # Here we use `json.loads`, since the WebApp sends the data JSON serialized string
    # (see webappbot.html)
#    print('atleast here')
#    data = json.loads(update.message.web_app_data.data)
#    print(data)
#    await update.message.reply_text("Your data was:")
#    for result in data:
#        await update.message.reply_text(f"{result['name']}: {result['value']}")


class Termibot:
    def __init__(self):
        self.TOKEN = '5783157460:AAH6gKOGQCJgS_sQg5W9piRtRRMaUkBDS7I'
        self.BOT = telegram.Bot(token=self.TOKEN)
        self.smsID = []
        self.dict = {}
        self.creds = None
        self.flow = None

    def cmdTRIGGER(self):
        self.updater = Updater(self.TOKEN, use_context=True)
        dp = self.updater.dispatcher
        try:
            dp.add_handler(MessageHandler(Filters.text, self.text_handler))
            dp.add_handler(MessageHandler(Filters.photo, self.photo_handler))
        except:
            pass
        self.updater.start_polling()
        self.updater.idle()

    def text_handler(self, update, context):
        txt = update.message.text
        USER_id = update.message.chat.id
        if USER_id not in self.dict:
            self.dict[USER_id]=len(self.dict)
            print(update.message.message_id)
            self.smsID.append([update.message.message_id])
        else:
            self.smsID[self.dict[USER_id]].append(update.message.message_id)
        CMD = txt.split(' ')[0]
        ARGS = txt.split(' ')[1:]
        self.cmd_handler(cmd=CMD,args=ARGS,USERid=USER_id)

    def photo_handler(self, update, context):
        txt = update.message.caption
        File = update.message.photo[-1].get_file()
        #path = File.download("output.jpg")
        USER_id = update.message.chat.id
        if USER_id not in self.dict:
            self.dict[USER_id]=len(self.dict)
            print(update.message.message_id)
            self.smsID.append([update.message.message_id])
        else:
            self.smsID[self.dict[USER_id]].append(update.message.message_id)
        CMD = txt.split(' ')[0]
        ARGS = txt.split(' ')[1:]+[File]
        self.cmd_handler(cmd=CMD,args=ARGS,USERid=USER_id)

    def cmd_handler(self,cmd,args,USERid):
        if cmd == '/start':
            self.start(userid=USERid)
        elif cmd == '/authenticate':
            self.authenticate(userid=USERid)
        elif cmd == '/token':
            code = args[0]
            self.savetoken(userid=USERid,CODE=code)
        elif cmd == '/balance':
            try:
                self.balance(userid=USERid)
            except:
                self.authenticate(userid=USERid)
        elif cmd == '/transactions':
            try:
                if len(args)==0:
                    self.transactions(userid=USERid)
                else:
                    self.transactions(userid=USERid,args=args[0])
            except:
                self.authenticate(userid=USERid)
        elif cmd == '/credit':
            try:
                amnt=float(args[0])
                rmrk=' '.join(args[1:])
            except:
                amnt=float(args[-1])
                rmrk=' '.join(args[:-1])
            try:
                self.credit(amount=amnt,remark=rmrk)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/purchase':
            try:
                paid_amnt=float(args[-3])
                due_amnt=float(args[-2])
                mtrl=' '.join(args[:-3])
                distri=args[-1]
            except:
                paid_amnt=float(args[-2])
                mtrl=' '.join(args[:-2])
                due_amnt=0
                distri=args[-1]
            try:
                self.purchase(paid=paid_amnt,material=mtrl,due=due_amnt,dist=distri)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/clrdue':
            try:
                self.clrdue(ARG=args)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/expense':
            try:
                self.expense(ARG=args)
                #amnt=float(args[0])
                #distri=args[1]
                #rmrk=' '.join(args[2:])

            #try:
                #if 'gst' in rmrk.lower():
                #    self.expense(amount=amnt,remark=rmrk)
                #else:
                #    self.expense(amount=amnt,remark=rmrk,dist=distri)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/sold':
            try:
                self.sold(ARG=args)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/addtxtl':
            try:
                #SRC,FBRC,CLR,LNTH,PRC = args
                #PRC = str(eval(PRC))
                #self.addtxtl(source=SRC,fabric=FBRC,color=CLR,length=LNTH,price=PRC)
                self.addtxtl(ARG=args,userid=USERid)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/txtladd':
            pass

        elif cmd == '/matfind':
            try:
                self.matfind(ARG=args,userid=USERid)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/details' or cmd == '/detail':
            try:
                CDN = args[-1]
                self.details(cdn=CDN,userid=USERid)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/design':
            try:
                CDN,WRK,PRC = args
                self.design(matcdn=CDN,workrmrk=WRK,price=PRC)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/plan':
            try:
                CDN,LEN = args
                self.plan(matcdn=CDN,length=LEN,userid=USERid)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/group':
            try:
                RMRK = args[-1]
                ARGS = args[:-1]
                self.group(mats=ARGS,rmrk=RMRK)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/upcat':
            #try:
            CDN,CAT,QTY,PRC = args
            self.upcat(matcdn=CDN,prodcat=CAT,quantity=QTY,profit=PRC,userid=USERid)
            #except:
            #    self.authenticate(userid=USERid)

        elif cmd == '/pfit':
            CDN,PRC = args
            self.pfit(prodcdn=CDN,saleprc=PRC)

        elif cmd == '/gptdes':
            CDN,CAT = args
            self.openAI_description(CDN,CAT,userid=USERid)

        elif cmd == '/gptname':
            CDN,CAT = args
            self.openAI_name(CDN,CAT,userid=USERid)

        elif cmd == '/prodcost':
            try:
                if len(args)>0:
                    DATE = args[0]
                else:
                    DATE = str(datetime.datetime.now().strftime('%d/%m/%Y'))
                self.prodcost(DATE)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/rmvprod':
            try:
                prod, typ = args
                self.rmvprod(child=prod,parent=typ)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/addprod':
            try:
                prod, typ = args
                self.addprod(child=prod,parent=typ)
            except:
                self.authenticate(userid=USERid)

        elif cmd == '/upname':
            MDL=args[0]
            NAME=' '.join(args[1:])
            self.upname(mdlno=MDL,name=NAME)

        elif cmd == '/updes':
            MDL=args[0]
            DES=' '.join(args[1:])
            self.updes(mdlno=MDL,description=DES)

        elif cmd == '/sell':
            self.sendPAYlink(ARG=args,userid=USERid)

        elif cmd == '/genINVC':
            self.genINVOICE(ARG=args,userid=USERid)

        elif cmd == '/genfINVC':
            self.genfINVOICE(ARG=args,userid=USERid)

        elif cmd == '/updINVC':
            self.updINVOICE(ARG=args,userid=USERid)

        elif cmd == '/customer':
            self.custINFO(ARG=args,userid=USERid)

        elif cmd == '/rmvtkn':
            self.rmvtkn()

        elif cmd == '/search':
            self.search()

        elif cmd == '/list':
            key = args[0]
            argl = args[1:]
            self.list(key1=key,key2=argl)

        elif cmd == '/edit':
            key = args[0]
            argl = args[1:]
            self.edit(key1=key,key2=argl)

        elif cmd == '/upmedia':
            parentType = args[1]
            if parentType == 'bill':
                self.addbillIMG(ARG=args,userid=USERid)
            if parentType == 'product':
                self.addprodIMG(ARG=args,userid=USERid)
            #fileObj = args[0]
            #parent = DriveFolder[args[1]]
            #child = args[2]
            #fileObj.download(child)
            #upload_to_folder(creds=self.creds,fileName=child,folder_id=parent)

        elif cmd == '/del':
            filetype = args[0]
            if filetype == 'bill':
                self.rmvbillIMG(ARG=args,userid=USERid)
            if filetype == 'product':
                self.rmvprodIMG(ARG=args,userid=USERid)
            if filetype == 'invoice':
                self.rmvinvcIMG(ARG=args,userid=USERid)

        elif cmd == '/cbalance':
            date = args[0]
            self.cbalance(month=date)

        elif cmd == '/debit':
            try:
                amnt=float(args[0])
                rmrk=' '.join(args[1:])
            except:
                amnt=float(args[-1])
                rmrk=' '.join(args[:-1])
            try:
                self.debit(amount=amnt,remark=rmrk)
            except:
                self.authenticate(userid=USERid)
        elif cmd == '/clear':
            self.cls(userid=USERid)
        elif cmd == '/exit':
            if str(USERid)==adminID:
                self.EXIT(userid=USERid)
            else:
                TXT = 'command not found !'
                self.sendUPDATE(CHATID=USERid,txt=TXT)
        elif cmd[0] == '/':
            TXT = 'command not found !'
            self.sendUPDATE(CHATID=USERid,txt=TXT)
        else:
            TXT = "Sorry I can only read '/commands', not 'texts'."
            self.sendUPDATE(CHATID=USERid,txt=TXT)

    def start(self,userid):
        self.sendUPDATE(CHATID=userid,txt=msg)

    def authenticate(self,userid):
        creds = None
        if os.path.exists('./token.json'):
            try:
                print('inside try')
                creds = Credentials.from_authorized_user_file('token.json', Scopes)
                self.creds = creds
            except:
                print('removing token')
                creds = None
                os.remove('token.json')
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except:
                    os.remove('token.json')
            else:
                self.flow = InstalledAppFlow.from_client_secrets_file(
                    'Antant_credentials.json', Scopes, redirect_uri='urn:ietf:wg:oauth:2.0:oob')
                auth_url, _ = self.flow.authorization_url(prompt='consent')
                LINK = "{}".format(auth_url)
                self.sendLINK(CHATID=userid,link=LINK,txt="Click here!")
                #self.sendUPDATE(CHATID=userid,txt="Click "+"<a href='%s'>Here</a>"%(TXT),parse_mode=ParseMode.HTML)

    def rmvtkn(self):
        os.remove('token.json')

    def savetoken(self,userid,CODE):
        # Save the credentials for the next run
        token = self.flow.fetch_token(code=CODE)
        creds = self.flow.credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        self.authenticate(userid)

    #def txtladd(self):
    #    WebAppInfo("https://google.com")
    #    #web_app=WebAppInfo("file:///E:/Antant_WSL/Ubuntu-20.04-antant/rootfs/home/antant/Test/formcam.html")

    def balance(self,userid='all'):
        values=get_spreadsheet(creds=self.creds)
        bal=values[1][6]
        TXT="current balance: %.2f/-"%(float(bal))
        if userid!='all':
            self.sendUPDATE(CHATID=userid,txt=TXT)
        else:
            self.sendUPDATE2all(txt=TXT)

    def search(self,IDw,POSw,keyword,findat=-1,findexact=False):
        results = get_index(creds=self.creds,getPOS=POSw,getID=IDw)
        search_area = results[0]["values"]
        while True:
            try:
                search_area[search_area.index([])]=['00/00/0000']
            except:
                break
        if keyword=='due':
            search_area = np.array(search_area).astype(dtype=np.float32)
            idx = np.flatnonzero(search_area.flatten()!=0)
        else:
            try:
                keynum = float(keyword)
                search_area = np.array(search_area).astype(dtype=np.float32)
                idx = np.flatnonzero(search_area.flatten()==keynum)
            except:
                print('Inside else except')
                if findexact:
                    print('findexact')
                    idx = np.flatnonzero(np.array(search_area)==keyword)
                    print('idx0:',idx)
                elif findexact==False and findat>=0:
                    idx = np.flatnonzero(np.core.defchararray.find(search_area,keyword)==findat)
                    print('idx1:',idx)
                else:
                    idx = np.flatnonzero(np.core.defchararray.find(search_area,keyword)!=-1)
                    print('idx2:',idx)
        #print(idx)
        return idx

    def addtxtl(self,ARG,userid):
        #fileObj = ARG[5]
        source,fabric,color,length,price = ARG[:5]
        price = str(eval(price))
        POSwr = "Textiles!A2:J"
        ppl = float(price)/float(length) # price per meter
        fhr = open('matcdn.dict','rb')
        matDict = pickle.load(fhr)
        fhr.close()
        try:
            cdn = matDict[fabric.lower()]
        except KeyError:
            counts=0
            cdn = matDict[list(matDict.keys())[0]]
            while cdn in list(matDict.values()):
                cdn = gencode(otype='material',keys=[fabric],bits=counts)
                counts+=1
        matDict[fabric.lower()]=cdn
        fhw = open('matcdn.dict','wb')
        pickle.dump(matDict,fhw)
        fhw.close()
        idx = self.search(IDw=SpreadSheet["materials"],POSw="Textiles!B1:B1000",keyword=cdn+'-',findat=0)+1
        if len(idx)>0:
            POSr = "Textiles!B%d"%(idx[-1])
            vals = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS=POSr)
            last_cdn = vals[-1][-1]
            new_cdn = cdn+'-'+str(eval(last_cdn[last_cdn.rfind('-')+1:])+1)
        else:
            new_cdn = cdn+'-1'
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        try:
            fileObj = ARG[5]
            matimgFile = './matimg/%s.jpg'%(new_cdn)
            fileObj.download(matimgFile)
            if color.lower() == 'x':
                caption = envrun('clipai','capimg.py',matimgFile)
                color = caption.stdout.read()
                TXT = color.decode('utf-8').strip()
                self.sendUPDATE(CHATID=userid,txt=TXT)
            envrun('imgsim','simimg_modsav.py',0)
        except:
            pass
        total = [timestamp,new_cdn,source,fabric,color,length,price,ppl,'',ppl]
        put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["materials"],putPOS=POSwr)
        TXT = "%s: material recently added."%(new_cdn)
        print(TXT)
        self.sendUPDATE(CHATID=userid,txt=TXT)

    def matfind(self,ARG,userid):
        fileObj = ARG[-1]
        fileTime = time.time()
        matimgFile = './matfind/%s.jpg'%(str(fileTime))
        fileObj.download(matimgFile)
        envrun('imgsim','simimg.py',matimgFile)
        fid = open('simimg.list','rb')
        simimg_list = pickle.load(fid)
        print(simimg_list)
        fid.close()
        for i in simimg_list:
            TXT = i[i.rfind('/')+1:i.rfind('.')]
            ffr = open(i,'rb')
            self.sendPHOTO(CHATID=userid,fileobj=ffr,CAPTION=TXT)
            ffr.close()
        os.remove(matimgFile)
        #self.sendUPDATE(CHATID=userid,txt=TXT)

    def details(self,cdn,userid):
        if '-P' in cdn:
            ID = SpreadSheet["designs"]
            idx = self.search(IDw=ID,POSw="Outfit!B1:B1000",keyword=cdn,findexact=True)+1
            prodet = get_spreadsheet(creds=self.creds,getID=ID,getPOS="Outfit!D%d:K%d"%(idx,idx))[0]
            prodeta = np.array(prodet)
            prodeta[prodeta=='']='--'
            prodet = list(prodeta)
            print(prodet)
            if len(prodet) < 7:
                length = prodet[0]
                prodcost = '--'
                upcat = '--'
            elif len(prodet) == 7:
                length,_,_,_,_,_,prodcost = prodet
                prodcost = '~'+str(round(float(prodcost)))+'INR'
                upcat = '--'
            else:
                length,_,_,_,_,_,prodcost,upcat = prodet
                prodcost = '~'+str(round(float(prodcost)))+'INR'
            if upcat=='--':
                upcats='product not listed in catalogue'
            else:
                upcats='product listed in catalogue'
            TXT = "*material: %s*\n\nlength: %s m\nprodcost: %s\n%s"%(cdn,length,prodcost,upcats)
        else:
            ID = SpreadSheet["materials"]
            idx = self.search(IDw=ID,POSw="Textiles!B1:B1000",keyword=cdn,findexact=True)+1
            matdet = get_spreadsheet(creds=self.creds,getID=ID,getPOS="Textiles!C%d:J%d"%(idx,idx))[0]
            print(matdet)
            source,matname,color,length,_,PPL,_,TPPL = matdet
            if TPPL=='':
                TPPL = PPL+'transport & GST'
            TXT = "*material: %s*\n\nsource: %s\nname: %s\ncolor: %s\nlength: %s m\nprice (1m): ~%s INR"%(cdn,source,matname,color,length,str(round(float(TPPL))))
        self.sendUPDATE(CHATID=userid,txt=TXT)

    def plan(self,matcdn,length,userid):
        #timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        idx = self.search(IDw=SpreadSheet["materials"],POSw="Textiles!B1:B1000",keyword=matcdn,findexact=True)+1
        matLen = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!F%d"%(idx))[0][0]
        basePrc = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!H%d"%(idx))[0][0]
        matPrc = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!J%d"%(idx))[0][0]
        rmnLen = float(matLen)-float(length)
        idd = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!B1:B1000",keyword=matcdn+'-',findat=0)+1
        if len(idd)>0:
            POSr = "Outfit!B%d"%(idd[-1])
            vals = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS=POSr)
            last_cdn = vals[-1][-1]
            new_cdn = matcdn+'-P'+str(eval(last_cdn[last_cdn.rfind('-P')+2:])+1)
        else:
            new_cdn = matcdn+'-P1'
        put_spreadsheet(creds=self.creds,putLIST=[rmnLen],putID=SpreadSheet["materials"],putPOS="Textiles!F%d"%(idx),write='update')
        put_spreadsheet(creds=self.creds,putLIST=[rmnLen*float(basePrc)],putID=SpreadSheet["materials"],putPOS="Textiles!G%d"%(idx),write='update')
        total = ['',new_cdn,float(matPrc)*float(length),float(length)]
        put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["designs"],putPOS="Outfit!A2:D")
        TXT = "%s: new product created."%(new_cdn)
        self.sendUPDATE(CHATID=userid,txt=TXT)

    def group(self,mats,rmrk):
        parent = mats[0]
        children = mats[1:]
        IDX = []
        for child in children:
            idx = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!B1:B1000",keyword=child,findat=0)+1
            print(idx)
            try:
                old_rmrk = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!L%d"%(idx))[0][0]
                new_rmrk = old_rmrk + ',' + rmrk
            except:
                old_rmrk = ''
                new_rmrk = rmrk
            total = [parent,new_rmrk]
            print(total)
            put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["designs"],putPOS="Outfit!L%d:M%d"%(idx,idx),write='update')

    def design(self,matcdn,workrmrk,price):
        color = ['paint','print','block','discharge','dye']
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        idd = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!B1:B1000",keyword=matcdn,findat=0)+1
        #matLen = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!F%d"%(idx))[0][0]
        #matPrc = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!K%d"%(idx))[0][0]
        #rmnLen = float(matLen)-float(length)
        #put_spreadsheet(creds=self.creds,putLIST=[rmnLen],putID=SpreadSheet["materials"],putPOS="Textiles!F%d"%(idx),write='update')
        #put_spreadsheet(creds=self.creds,putLIST=[rmnLen*float(matPrc)],putID=SpreadSheet["materials"],putPOS="Textiles!G%d"%(idx),write='update')
        #put_spreadsheet(creds=self.creds,putLIST=[rmnLen*float(matPrc)],putID=SpreadSheet["materials"],putPOS="Textiles!J%d"%(idx),write='update')
        total = [price,workrmrk]
        isStitch = True
        for clr in color:
            if clr in workrmrk:
                put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["designs"],putPOS="Outfit!E%d:F%d"%(idd,idd),write='update')
                isStitch = False
                break
            else:
                pass
        if isStitch:
            put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["designs"],putPOS="Outfit!G%d:H%d"%(idd,idd),write='update')
        put_spreadsheet(creds=self.creds,putLIST=[timestamp],putID=SpreadSheet["designs"],putPOS="Outfit!A%d"%(idd),write='update')

    def upcat(self,matcdn,prodcat,quantity,profit,userid):
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cdn = gencode(otype='product',keys=[matcdn,prodcat])
        idx = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!B1:B1000",keyword=matcdn,findat=0)+1
        print(idx+1)
        #status = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!K%d"%(idx))
        #if status == None:
        if len(idx)>0:
            status = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!K%d"%(idx))
            if status == None:
                prodCOST = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!J%d"%(idx))[0][0]
                idp = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=cdn,findat=0)+1
                print(idp,cdn)
                if len(idp)>0:
                    POSr = "Outfit!C%d"%(idp[-1])
                    vals = get_spreadsheet(creds=self.creds,getID=SpreadSheet["products"],getPOS=POSr)
                    print(vals)
                    last_cdn = vals[-1][-1]
                    new_cdn = cdn+'-'+str(eval(last_cdn[last_cdn.rfind('-')+1:])+1)
                else:
                    new_cdn = cdn+'-1'
                    #if vals == None:
                    #    new_cdn = cdn+'-1'
                    #else:
                    #    last_cdn = vals[-1][-1]
                    #    new_cdn = cdn+'-'+str(eval(last_cdn[last_cdn.rfind('-')+1:])+1)
                print(new_cdn)
                frh = open('prodtype.list','rb')
                prodList = pickle.load(frh)
                frh.close()
                prodArr = np.array(prodList)
                sid = np.flatnonzero(prodArr[:,1]==prodcat)[0]
                prodID = str(prodArr[sid,0])
                prodHRK = getProd_hrk(prodArr, prodID).split(' ')
                category = ' > '.join(prodHRK)
                price = float(prodCOST)/float(quantity)+float(profit)
                total = [timestamp, matcdn, new_cdn, category, '', '', quantity, price]
                put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["products"],putPOS="Outfit!A2:H")
                put_spreadsheet(creds=self.creds,putLIST=[1],putID=SpreadSheet["designs"],putPOS="Outfit!K%d"%(idx),write='update')
                TXT = '%s: product added in the catalogue.'%(new_cdn)
                self.sendUPDATE(CHATID=userid,txt=TXT)
            else:
                TXT = 'The material is already in product-catalogue!'
                self.sendUPDATE(CHATID=userid,txt=TXT)
        else:
            TXT = 'The material does not exist in design!'
            self.sendUPDATE(CHATID=userid,txt=TXT)

    def pfit(self,prodcdn,saleprc):
        pIDw = SpreadSheet["products"]
        dIDw = SpreadSheet["designs"]
        fIDw = SpreadSheet["finance"]
        idx = self.search(IDw=pIDw,POSw="Outfit!C1:C1000",keyword=prodcdn,findexact=True)+1
        descdn = get_spreadsheet(creds=self.creds,getID=pIDw,getPOS="Outfit!B%d"%(idx))[0][0]
        idd = self.search(IDw=dIDw,POSw="Outfit!B1:B1000",keyword=descdn,findexact=True)+1
        prdtyp = get_spreadsheet(creds=self.creds,getID=pIDw,getPOS="Outfit!D%d"%(idx))[0][0]
        prdcst = get_spreadsheet(creds=self.creds,getID=dIDw,getPOS="Outfit!J%d"%(idd))[0][0]
        _,sex,typ = prdtyp.split(' > ')
        profit = float(saleprc)-round(float(prdcst))
        total = [prodcdn,sex,typ,prdcst,saleprc,profit]
        put_spreadsheet(creds=self.creds,putLIST=total,putID=fIDw,putPOS="Stat!A2:F")

    def upname(self,mdlno,name):
        idx = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=mdlno)+1
        put_spreadsheet(creds=self.creds,putLIST=[name],putID=SpreadSheet["products"],putPOS="Outfit!E%d"%(idx),write='update')
        TXT = 'The product name is updated.'
        self.sendUPDATE2all(txt=TXT)

    def updes(self,mdlno,description):
        idx = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=mdlno)+1
        put_spreadsheet(creds=self.creds,putLIST=[description],putID=SpreadSheet["products"],putPOS="Outfit!F%d"%(idx),write='update')
        TXT = 'The product description is updated.'
        self.sendUPDATE2all(txt=TXT)

    def openAI_description(self,prodcd,prodcat,userid):
        self.openAI_write(prodcd,prodcat,typ='description',wordlim=50)

    def openAI_name(self,prodcd,prodcat,userid):
        self.openAI_write(prodcd,prodcat,typ='name',wordlim=2)

    def openAI_write(self,prodcd,prodcat,typ,wordlim):
        matcdn = prodcd[:prodcd.rfind('-')]
        if prodcat.lower() in ['saree','blouse']:
            idx = self.search(IDw=SpreadSheet["materials"],POSw="Textiles!B1:B1000",keyword=matcdn)+1
            mat = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!D%d"%(idx))[0][0]
            mat = mat.replace('-',' ')
            clr = get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!E%d"%(idx))[0][0]
            clr = clr.replace('-',' ')
            idd = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!B1:B1000",keyword=prodcd)+1
            #input_str = "Suggest a fifty-word description for a %s colored %s saree."%(clr,mat)
            input_str = "Write a %d word artistic %s for the following.\nOutfit: %s\nFabric: %s\nBase color: %s\n"%(wordlim,typ,prodcat,mat,clr)
            try:
                prnt = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!F%d"%(idd))[0][0]
                prnt = prnt.replace('-',' ')
                #input_str += "It has %s on it."%(prnt)
                input_str += "Art-work: %s\n"%(prnt)
            except:
                prnt = ''
            try:
                stap = get_spreadsheet(creds=self.creds,getID=SpreadSheet["designs"],getPOS="Outfit!H%d"%(idd))[0][0]
                stap = stap.replace('-',' ')
                #input_str += "It has %s work on it."
                input_str += "Stitch-work: %s\n"%(stap)
            except:
                stap = ''
            print(input_str)
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=input_str,
            temperature=0.85,
            max_tokens=1258,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
            TXT = response["choices"][0]["text"]
            self.sendUPDATE2all(txt=TXT)

    def rmvprod(self,child,parent):
        child = child.lower()
        parent = parent.lower()
        frh = open('prodtype.list','rb')
        prodList = pickle.load(frh)
        frh.close()
        frh = open('prodtype.dict','rb')
        prodDict = pickle.load(frh)
        frh.close()
        prodArr = np.array(prodList)
        cidL = np.flatnonzero(prodArr[:,1]==child)
        pidL = np.flatnonzero(prodArr[:,1]==parent)
        for pid in pidL:
            pidC = set(np.flatnonzero(prodArr[:,2]==str(pid)))
            cid = list(pidC.intersection(set(cidL)))
            if len(cid)>0:
                break
            else:
                pass
        if len(cid)>0:
            hrk = getProd_hrk(prodArr, cid[0])
            chrk = hrk.split(' ')
            print(chrk)
            prodDict = remove_nested_keys(prodDict, parent, child)
            del prodList[cid[0]]
            frh = open('prodtype.list','wb')
            pickle.dump(prodList,frh)
            frh.close()
            frh = open('prodtype.dict','wb')
            pickle.dump(prodDict,frh)
            frh.close()
            hrk_str = ' > '.join(chrk)
            TXT = 'This product is removed:\n'+hrk_str
            self.sendUPDATE2all(txt=TXT)
        else:
            TXT = 'This product does not exist.'
            self.sendUPDATE2all(txt=TXT)

    def addprod(self,child,parent):
        child = child.lower()
        parent = parent.lower()
        frh = open('prodtype.list','rb')
        prodList = pickle.load(frh)
        frh.close()
        prodArr = np.array(prodList)
        sidL = np.flatnonzero(prodArr[:,1]==child)
        print('sidL: ',sidL)
        if len(sidL)>0:
            sid = sidL[0]
            prod_hrk = getProd_hrk(prodArr, sid).split(' ')
            hrk_str = ' > '.join(prod_hrk)
            TXT = 'This product already exists under the category:\n'+hrk_str
            self.sendUPDATE2all(txt=TXT)
        else:
            print(prodArr[:,1]==parent)
            sidt = np.flatnonzero(prodArr[:,1]==parent)[0]
            print('tid,pid')
            tid = prodArr[:,0][sidt]    #type_id
            pid = prodList[-1][0]+1     #prod_id
            print('tid,pid')
            frh = open('prodtype.dict','rb')
            prodDict = pickle.load(frh)
            frh.close()
            print('tid,pid')
            prodList.append([pid,child,tid])
            prodArr = np.array(prodList)
            print(prodArr)
            prod_hrk = getProd_hrk(prodArr, pid).split(' ')
            subProd = prodDict
            for i in prod_hrk[:-1]:
                subProd = subProd[i]
            subProd[child] = {}
            fwh = open('prodtype.list','wb')
            pickle.dump(prodList,fwh)
            fwh.close()
            fwh = open('prodtype.dict','wb')
            pickle.dump(prodDict,fwh)
            fwh.close()
            hrk_str = ' > '.join(prod_hrk)
            TXT = 'The product is added under the category:\n'+hrk_str
            self.sendUPDATE2all(txt=TXT)

    def prodcost(self,date):
        idx = self.search(IDw=SpreadSheet["finance"],POSw="Expense!A2:A1000",keyword=date)+2
        RNG = []
        for ids in idx:
            RNG.append("Expense!B%d"%(ids))
        results = get_index(creds=self.creds,getPOS=RNG,getID=SpreadSheet["finance"])
        SUM = 0
        for val in results:
            SUM += float(val["values"][0][0])

        idd = self.search(IDw=SpreadSheet["designs"],POSw="Outfit!A2:A1000",keyword=date)+2     # designs made on the day
        idm1 = self.search(IDw=SpreadSheet["materials"],POSw="Textiles!A2:A1000",keyword=date)+2     # materials bought on the day
        idm2 = self.search(IDw=SpreadSheet["materials"],POSw="Textiles!G2:G1000",keyword=0)+2     # materials bought on the day
        idm = list(set(idm1)-set(idm2))
        PRCtoADD = SUM/float(len(idd)+len(idm))
        for ids in idd:
            put_spreadsheet(creds=self.creds,putLIST=[PRCtoADD],putID=SpreadSheet["designs"],putPOS="Outfit!I%d"%(ids),write='update')
            put_spreadsheet(creds=self.creds,putLIST=['=SUM(C%d,E%d,G%d,I%d)'%(ids,ids,ids,ids)],putID=SpreadSheet["designs"],putPOS="Outfit!J%d"%(ids),write='update')
        for ids in idm:
            matLen = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["materials"],getPOS="Textiles!F%d"%(ids))[0][0])
            put_spreadsheet(creds=self.creds,putLIST=[PRCtoADD/matLen],putID=SpreadSheet["materials"],putPOS="Textiles!I%d"%(ids),write='update')
            put_spreadsheet(creds=self.creds,putLIST=['=SUM(H%d,I%d)'%(ids,ids)],putID=SpreadSheet["materials"],putPOS="Textiles!J%d"%(ids),write='update')

    def deconv_dist(self,tdist='x,x,x',total=0):
        DIST = np.array(tdist.split(','))
        idnx = DIST!='x'
        rm = total-sum([eval(i) for i in DIST[idnx]])
        idex = DIST=='x'
        rmph = rm/sum(idex)
        DIST[idex]=str(rmph)
        return DIST

    def purchase(self,material,paid,due=0,userid='all'):
        IDw = SpreadSheet["finance"]
        POSw = "Purchase!A2:F"
        POSr = "Purchase!B2:B1000"
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        try:
            vals = get_spreadsheet(creds=self.creds,getID=IDw,getPOS=POSr)
            last_txn = vals[-1][-1]
        except:
            last_txn = 'none'
        date = datetime.datetime.now().strftime('%m-%d-%Y')
        if date in last_txn:
            txn = date+'-'+str(eval(last_txn[last_txn.rfind('-')+1:])+1)
            print(txn)
        else:
            txn = date+'-1'
        bill = '-'
        total=[timestamp, txn, material, paid, due, bill]
        put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw)
        balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
        balance -= float(paid)
        put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')

    def addbillIMG(self,ARG,userid):
        fileObj = ARG[-1]
        grandparent = DriveFolder[ARG[0]]
        txn = ARG[1]
        child = 'Page%s.jpg'%(ARG[2])
        #parent = DriveFolder[ARG[1]]
        #child = '%s.jpg'%(ARG[2])
        verbose = False
        try:
            vb = ARG[3]
            if vb == 'v':
                verbose = True
        except:
            pass
        ffr = open('billimgs.dict','rb')
        billimgs = pickle.load(ffr)
        ffr.close()
        try:
            parent = billimgs[txn]
        except:
            parent=create_folder(creds=self.creds,dirName=txn,parent_id=grandparent)
            billimgs[txn] = parent
            ffw = open('billimgs.dict','wb')
            pickle.dump(billimgs,ffw)
            ffw.close()
        childFile = './%s'%(child)
        fileObj.download(childFile)
        folder_link = "https://drive.google.com/drive/folders/%s"%(parent)
        upload_to_folder(creds=self.creds,fileName=child,filePath=childFile,parent_id=parent)
        idp = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!B1:B1000",keyword=ARG[1],findat=0)+1
        put_spreadsheet(creds=self.creds,putLIST=[folder_link],putID=SpreadSheet["finance"],putPOS="Purchase!F%d"%(idp),write='update')
        if verbose:
            LINK = "{}".format(folder_link)
            self.sendLINK(CHATID=userid,link=LINK,txt="Photos for %s is here!"%(ARG[1]))
        os.remove(child)

    def rmvbillIMG(self,ARG,userid):
        filetype = ARG[0]
        filename = ARG[1]
        ffr = open('billimgs.dict','rb')
        billdict = pickle.load(ffr)
        ffr.close()
        try:
            fileid = billdict[filename]
            delete_file(creds=self.creds,file_id=fileid)
            del billdict[filename]
            ffw = open('billimgs.dict','wb')
            pickle.dump(billdict,ffw)
            ffw.close()
            idp = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!B1:B1000",keyword=filename,findat=0)+1
            put_spreadsheet(creds=self.creds,putLIST=[''],putID=SpreadSheet["finance"],putPOS="Purchase!F%d"%(idp),write='update')
            TXT = 'bill removed successfully!'
        except:
            TXT = 'this bill does not exist!'
        self.sendUPDATE(CHATID=userid,txt=TXT)

    def addprodIMG(self,ARG,userid):
        fileObj = ARG[-1]
        grandparent = DriveFolder[ARG[0]]
        prodcd = ARG[1]
        child = 'Fig%s.jpg'%(ARG[2])
        verbose = False
        try:
            vb = ARG[3]
            if vb == 'v':
                verbose = True
        except:
            pass
        ffr = open('prodimgs.dict','rb')
        prodimgs = pickle.load(ffr)
        ffr.close()
        try:
            parent = prodimgs[prodcd]
        except:
            parent=create_folder(creds=self.creds,dirName=prodcd,parent_id=grandparent)
            prodimgs[prodcd] = parent
            ffw = open('prodimgs.dict','wb')
            pickle.dump(prodimgs,ffw)
            ffw.close()
        childFile = './%s'%(child)
        fileObj.download(childFile)
        folder_link = "https://drive.google.com/drive/folders/%s"%(parent)
        upload_to_folder(creds=self.creds,fileName=child,filePath=childFile,parent_id=parent)
        idp = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=prodcd,findat=0)+1
        put_spreadsheet(creds=self.creds,putLIST=[folder_link],putID=SpreadSheet["products"],putPOS="Outfit!I%d"%(idp),write='update')
        if verbose:
            LINK = "{}".format(folder_link)
            self.sendLINK(CHATID=userid,link=LINK,txt="Photos for %s is here!"%(prodcd))
        os.remove(childFile)
        #upload_to_folder(creds=self.creds,fileName=child,folder_id=parent)

    def rmvprodIMG(self,ARG,userid):
        filename = ARG[1]
        ffr = open('prodimgs.dict','rb')
        proddict = pickle.load(ffr)
        ffr.close()
        try:
            fileid = proddict[filename]
            print(fileid)
            delete_file(creds=self.creds,file_id=fileid)
            del proddict[filename]
            idp = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=filename,findat=0)+1
            put_spreadsheet(creds=self.creds,putLIST=[''],putID=SpreadSheet["products"],putPOS="Outfit!I%d"%(idp),write='update')
            print(proddict)
            ffw = open('prodimgs.dict','wb')
            pickle.dump(proddict,ffw)
            ffw.close()
            TXT = 'product images removed successfully!'
        except:
            TXT = 'this product images do not exist!'
        self.sendUPDATE(CHATID=userid,txt=TXT)

    def expense(self,ARG):
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        remark=ARG[0]
        amount=float(ARG[1])
        IDw = SpreadSheet["finance"]
        POSw = "Expense!A2:C"
        total=[timestamp, amount, remark]+['']
        put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw)
        #else:
        #    dist = ARG[2]
        #    DIST = list(self.deconv_dist(tdist=dist,total=amount))
        #    total=[timestamp, amount, remark]+DIST
        #    put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw)
        balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
        balance -= amount
        put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')
        #TXT = "A new transaction is done!\n\ntime: %s\ndebit: %s/-\nremarks: %s"%(timestamp,amount,remark)
        #self.sendUPDATE2all(txt=TXT)
        #self.balance(userid='all')

    def sold(self,ARG):
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        remark=ARG[0]
        amount=float(ARG[1])
        IDw = SpreadSheet["products"]
        total=[timestamp, amount, remark]+['']
        if 'other' in remark.lower():
            POSw = "Accessories!A2:C"
            put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw)
        else:
            idx = self.search(IDw=IDw,POSw="Outfit!C1:C1000",keyword=remark,findexact=True)+1
            print(idx)
            qty = int(get_spreadsheet(creds=self.creds,getID=IDw,getPOS="Outfit!G%d"%(idx))[0][0])
            try:
                amt = get_spreadsheet(creds=self.creds,getID=IDw,getPOS="Outfit!L%d"%(idx))[0][0]
            except:
                amt = 0
            print(qty,amt)
            if qty>0:
                amt = float(amt)+amount
                qty -= 1
                POSw = "Outfit!G%d"%(idx)
                put_spreadsheet(creds=self.creds,putLIST=[qty],putID=IDw,putPOS=POSw,write='update')
                POSw = "Outfit!L%d"%(idx)
                put_spreadsheet(creds=self.creds,putLIST=[amt],putID=IDw,putPOS=POSw,write='update')
        balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
        balance += amount
        put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')

    def list(self,key1,key2):
        if key1=='purchase':
            base = "Purchase!"
            RNG = []
            keyy = key2[0]
            nonzeroidx = False
            if keyy == "due":
                idx = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!E2:E1000",keyword=keyy)+2
                if len(idx)==0:
                    TXT = 'No due for purchases'
                else:
                    nonzeroidx = True
            else:
                date = keyy
                idx = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!A2:A1000",keyword=date)+2
                if len(idx)==0:
                    TXT = "No purchases on this date"
                else:
                    nonzeroidx = True
            if nonzeroidx:
                for ids in idx:
                    RNG.append(base+"A%d:F%d"%(ids,ids))
                results = get_index(creds=self.creds,getPOS=RNG,getID=SpreadSheet["finance"])
                TXT = "TXN\tmaterial\tpaid\tdue\tbill\n"
                for val in results:
                    TXT += "\t".join(val["values"][0][1:])+'\n'
            self.sendUPDATE2all(txt=TXT)
        else:
            pass

    def clrdue(self,ARG):
        txn = ARG[0]
        idx = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!B2:B1000",keyword=txn)[0]+2
        olddue = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Purchase!E%d"%(idx))[0][0])
        oldpaid = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Purchase!D%d"%(idx))[0][0])
        paid = float(ARG[1])
        nowpaid = oldpaid+paid
        nowdue = olddue-paid
        total = [nowpaid,nowdue]
        POSw = "Purchase!D%d:E%d"%(idx,idx)
        IDw = SpreadSheet["finance"]
        put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw,write='update')
        dist = ARG[2]
        DIST = np.asarray(self.deconv_dist(tdist=dist,total=paid),dtype=np.float32)
        olddist = np.asarray(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Purchase!G%d:I%d"%(idx,idx))[0],dtype=np.float32)
        newdist = [str(i) for i in olddist+DIST]
        print(newdist)
        POSw = "Purchase!G%d:I%d"%(idx,idx)
        put_spreadsheet(creds=self.creds,putLIST=newdist,putID=IDw,putPOS=POSw,write='update')

    def edit(self,key1,key2):
        if key1=='purchase':
            txn = key2[0]
            total = key2[1:4]
            paid = float(total[1])
            dist = key2[4]
            DIST = list(self.deconv_dist(tdist=dist,total=paid))
            idx = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!B2:B1000",keyword=txn)[0]+2
            POSw = "Purchase!C%d:E%d"%(idx,idx)
            IDw = SpreadSheet["finance"]
            put_spreadsheet(creds=self.creds,putLIST=total,putID=IDw,putPOS=POSw,write='update')
            POSw = "Purchase!G%d:I%d"%(idx,idx)
            put_spreadsheet(creds=self.creds,putLIST=DIST,putID=IDw,putPOS=POSw,write='update')
        else:
            pass

    def transactions(self,userid,args=1):
        values=get_spreadsheet(creds=self.creds)
        total_transactions=len(values)
        num=min(int(args),total_transactions-1)
        for i in range(total_transactions-num,total_transactions):
            val=values[i]
            TXT=self.maketext(val)
            self.sendUPDATE(CHATID=userid,txt=TXT)

    def genPAYlink(self,AMNT,MODELno,ICONfig,QRname):
        logo = Image.open(ICONfig)
        BOXsize = 10
        BORDER = 1
        QRcode = qrcode.QRCode(box_size=BOXsize,border=BORDER,error_correction=qrcode.constants.ERROR_CORRECT_H)
        link = "upi://pay?pa=7044319466-1@okbizaxis&pn=Antant for products %s&am=%d&tn=Antant products %s&cu=INR"%(MODELno,AMNT,MODELno)
        QRcode.add_data(link)
        QRcode.make(fit=True)
        QRimg = QRcode.make_image().convert('RGB')
        halfQRsize = (QRimg.size[0]-BOXsize*BORDER*2)/2
        halfLOGOsize = halfQRsize/3
        if halfQRsize%BOXsize == BOXsize/2:
            if halfLOGOsize%BOXsize == BOXsize/2:
                LOGOsize = 2*halfLOGOsize
            else:
                LOGOsize = 2*(halfLOGOsize-(halfLOGOsize%BOXsize-BOXsize/2))
        else:
            if halfLOGOsize%BOXsize == 0:
                LOGOsize = 2*halfLOGOsize
            else:
                LOGOsize = 2*(halfLOGOsize-(halfLOGOsize%BOXsize-BOXsize))
        LOGOsize = int(LOGOsize)
        logo = logo.resize((LOGOsize, LOGOsize), Image.ANTIALIAS)
        pos = ((QRimg.size[0] - logo.size[0]) // 2,(QRimg.size[1] - logo.size[1]) // 2)
        QRimg.paste(logo, pos)
        fileName = 'upiQR_%s.png'%(QRname)
        QRimg.save(fileName)
        return fileName

    def sendPAYlink(self,ARG,userid):
        arg = ARG[0]
        if type(arg)==str:
            modelNO = arg
        else:
            prodqrFile = 'Outfit4sell.jpg'
            arg.download(prodqrFile)
            img = cv2.imread(prodqrFile)
            qrcd = decode(img)[0]
            modelNO = qrcd.data.decode("utf-8")
            print(modelNO)
        idp = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=modelNO,findat=0)+1
        amnt = round(float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["products"],getPOS="Outfit!H%d"%(idp))[0][0]))
        fileName = self.genPAYlink(AMNT=amnt,MODELno=modelNO,ICONfig='ICON.png',QRname=modelNO)
        #logo = Image.open('ICON_pixel.png')
        #BOXsize = 10
        #BORDER = 1
        #QRcode = qrcode.QRCode(box_size=BOXsize,border=BORDER,error_correction=qrcode.constants.ERROR_CORRECT_H)
        #link = "upi://pay?pa=pratyashakundu9799-2@oksbi&pn=Antant for product %s&am=%d&tn=Antant product %s&cu=INR"%(modelNO,amnt,modelNO)
        #QRcode.add_data(link)
        #QRcode.make(fit=True)
        #QRimg = QRcode.make_image().convert('RGB')
        #halfQRsize = (QRimg.size[0]-BOXsize*BORDER*2)/2
        #halfLOGOsize = halfQRsize/3
        #if halfQRsize%BOXsize == BOXsize/2:
        #    if halfLOGOsize%BOXsize == BOXsize/2:
        #        LOGOsize = 2*halfLOGOsize
        #    else:
        #        LOGOsize = 2*(halfLOGOsize-(halfLOGOsize%BOXsize-BOXsize/2))
        #else:
        #    if halfLOGOsize%BOXsize == 0:
        #        LOGOsize = 2*halfLOGOsize
        #    else:
        #        LOGOsize = 2*(halfLOGOsize-(halfLOGOsize%BOXsize-BOXsize))
        #LOGOsize = int(LOGOsize)
        #logo = logo.resize((LOGOsize, LOGOsize), Image.ANTIALIAS)
        #pos = ((QRimg.size[0] - logo.size[0]) // 2,(QRimg.size[1] - logo.size[1]) // 2)
        #QRimg.paste(logo, pos)
        #fileName = 'upiQR_%s.png'%(modelNO)
        #QRimg.save(fileName)
        ffr = open(fileName,'rb')
        self.sendPHOTO(CHATID=userid,fileobj=ffr)
        ffr.close()
        os.remove(fileName)

    def genfINVOICE(self,ARG,userid):
        self.genINVOICE(ARG=ARG,userid=userid,false=True)

    def genINVOICE(self,ARG,userid,false=False):
        billtype = 'regular'
        phn = ARG[0]
        try:
            paid = float(ARG[-1])
            items = ARG[1:-1]
        except:
            discp = float(ARG[-1][:-1])
            paid = float(ARG[-2])
            items = ARG[1:-2]
            billtype = 'discount'
        print(paid,items)
        itqt = {}
        for item in items:
            posAstrx = item.rfind('*')
            if posAstrx != -1:
                qty = eval(item[posAstrx+1:])
                itm = item[:posAstrx]
            else:
                qty = 1
                itm = item
            itqt[itm] = qty
        total = 0
        itemROWin = ""
        itemROW = "    \centering %d & \centering %s & \centering %s & \centering %.2f & \centering %d & %.2f \\\\[2.5ex]\hline\n    & & & & &\\\\\n"
        for sl,i in enumerate(itqt):
            print(itqt)
            idp = self.search(IDw=SpreadSheet["products"],POSw="Outfit!C1:C1000",keyword=i,findat=0)+1
            stock = int(get_spreadsheet(creds=self.creds,getID=SpreadSheet["products"],getPOS="Outfit!G%d"%(idp))[0][0])
            if stock>0:
                print('i,idp: ',i)
                name = get_spreadsheet(creds=self.creds,getID=SpreadSheet["products"],getPOS="Outfit!E%d"%(idp))[0][0]
                code = i
                rate = round(float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["products"],getPOS="Outfit!H%d"%(idp))[0][0]))
                qnty = itqt[i]
                amnt = rate*qnty
                itemROWin += itemROW%(sl+1,name,code,rate,qnty,amnt)
                total += amnt
                stock -= qnty
            else:
                TXT = "Product:%s is out of stock!"%(i)
                self.sendUPDATE(CHATID=userid,txt=TXT)
            if false:
                pass
            else:
                put_spreadsheet(creds=self.creds,putLIST=[stock],putID=SpreadSheet["products"],putPOS="Outfit!G%d"%(idp),write='update')

        #disc = total*(discp/100)
        #total = total - disc
        #due = total - paid
        if billtype=='regular':
            ffr = open('invoice_template_regular.txt','r')
        else:
            disc = total*(discp/100)
            total = total - disc
            ffr = open('invoice_template_discount.txt','r')
        total = round(total)
        invtmp = ffr.read()
        ffr.close()
        due = total - paid
        flr = open('invcimgs.dict','rb')
        invcimgs = pickle.load(flr)
        flr.close()
        DATE = str(datetime.datetime.now().strftime('%m%d%Y'))
        try:
            todayINV = []
            for invc in invcimgs.keys():
                if DATE in invc:
                    todayINV.append(int(invc[invc.rfind('-')+1:]))
            INVno=DATE+'-'+str(max(todayINV)+1)
        except:
            INVno=DATE+'-1'
        print(INVno)
        #if DATE in lastINV:
        #    INVno=DATE+'-'+str(int(eval(lastINV[lastINV.rfind('-')+1:])+1))
        #else:
        #    INVno=DATE+'-1'
        #billNO = billist[-1]+1
        #billist.append(billNO)
        fileTemp = self.genPAYlink(AMNT=due,MODELno=INVno,ICONfig='ICON.png',QRname=INVno)
        ffr = open(fileTemp,'rb')
        self.sendPHOTO(CHATID=userid,fileobj=ffr)
        ffr.close()
        os.remove(fileTemp)
        fileName = self.genPAYlink(AMNT=due,MODELno=INVno,ICONfig='ICON_pixel.png',QRname=INVno)
        fcr = open('custdata.dict','rb')
        custdict = pickle.load(fcr)
        fcr.close()
        custInfo = custdict[phn] + ' | ' + phn
        if billtype=='regular':
            TUPLEin = (INVno,custInfo,itemROWin,'%.2f'%(total),'%.2f'%(paid),'%.2f'%(due),fileName)
        else:
            TUPLEin = (INVno,custInfo,itemROWin,str(discp)+'\%','%.2f'%(disc),'%.2f'%(total),'%.2f'%(paid),'%.2f'%(due),fileName)
        invtxt = invtmp%TUPLEin
        FLDR = 'INVOICE-%s'%(INVno)
        os.mkdir(FLDR)
        ffw = open('%s/bill.tex'%(FLDR),'w')
        print(invtxt,file=ffw)
        ffw.close()
        os.system('cp ICON.png %s/'%(FLDR))
        os.system('pdflatex -output-directory %s %s/bill.tex'%(FLDR,FLDR))
        FILEid = upload_to_folder(self.creds,'%s.pdf'%(INVno),'%s/bill.pdf'%(FLDR),parent_id=DriveFolder['invoice'],permit='writer')
        INVlink = "https://drive.google.com/file/d/%s/view?usp=share_link"%(FILEid)
        #PDF = open('%s/bill.pdf'%(FLDR),'rb')
        #self.sendPHOTO(CHATID=userid,fileobj=PDF)
        #PDF.close()
        IDw = SpreadSheet["finance"]
        POSw = "Sale!A2:I"
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        products = ','.join(list(itqt.keys()))
        putTOTAL=[timestamp,INVno,products,custdict[phn],phn,total,paid,due,INVlink]
        put_spreadsheet(creds=self.creds,putLIST=putTOTAL,putID=IDw,putPOS=POSw)
        invcimgs[INVno] = FILEid
        ffw = open('invcimgs.dict','wb')
        pickle.dump(invcimgs,ffw)
        ffw.close()
        LINK = "{}".format(INVlink)
        self.sendLINK(CHATID=userid,link=INVlink,txt="The invoice PDF is here!")
        if false:
            pass
        else:
            balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
            balance += float(paid)
            put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')
        if due==0:
            os.system('rm -rf %s'%(FLDR))
        elif due>0:
            if billtype=='regular':
                TUPLEind = (INVno,custInfo,itemROWin,'%s','%s','%s','%s')
            else:
                TUPLEind = (INVno,custInfo,itemROWin,str(discp)+'\%%','%.2f'%(disc),'%s','%s','%s','%s')
            invtxt = invtmp%TUPLEind
            ffw = open('%s/duebill.txt'%(FLDR),'w')
            print(invtxt,file=ffw)
            ffw.close()
            os.system('rm %s/bill.*'%(FLDR))

    def cbalance(self,month):
        idx_sale = self.search(IDw=SpreadSheet["finance"],POSw="Sale!A2:A1000",keyword=month)+2
        idx_purchase = self.search(IDw=SpreadSheet["finance"],POSw="Purchase!A2:A1000",keyword=month)+2
        idx_expense = self.search(IDw=SpreadSheet["finance"],POSw="Expense!A2:A1000",keyword=month)+2
        RNG = []
        for ids in idx_sale:
            RNG.append("Sale!G%d"%(ids))
        results = get_index(creds=self.creds,getPOS=RNG,getID=SpreadSheet["finance"])
        CREDIT = 0
        for val in results:
            CREDIT += float(val["values"][0][0])
        
        RNG = []
        for ids in idx_purchase:
            RNG.append("Purchase!D%d"%(ids))
        results = get_index(creds=self.creds,getPOS=RNG,getID=SpreadSheet["finance"])
        DEBIT = 0
        for val in results:
            DEBIT += float(val["values"][0][0])
        RNG = []
        for ids in idx_expense:
            RNG.append("Expense!B%d"%(ids))
        results = get_index(creds=self.creds,getPOS=RNG,getID=SpreadSheet["finance"])
        for val in results:
            DEBIT += float(val["values"][0][0])
        RETURN = CREDIT-DEBIT
        total = [month,DEBIT,CREDIT,RETURN]
        try:
            idx_invret = self.search(IDw=SpreadSheet["finance"],POSw="InvRet!A2:A1000",keyword=month,findat=0)+2
        except:
            idx_invret = []
        if len(idx_invret) > 0:
            idx = idx_invret[-1]
            put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["finance"],putPOS="InvRet!A%d:D%d"%(idx,idx),write='update')
        else:
            put_spreadsheet(creds=self.creds,putLIST=total,putID=SpreadSheet["finance"],putPOS="InvRet!A2:D")

    def updINVOICE(self,ARG,userid):
        INVno = ARG[0]
        amnt = float(ARG[1])
        idi = self.search(IDw=SpreadSheet["finance"],POSw="Sale!B1:B1000",keyword=INVno,findat=0)+1
        INVd = get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Sale!C%d:H%d"%(idi,idi))[0]    # Invoice details
        products = INVd[0]
        name = INVd[1]
        phn = INVd[2]
        total = float(INVd[3])
        paid = float(INVd[4])+amnt
        due = float(INVd[5])-amnt
        FLDR = 'INVOICE-%s'%(INVno)
        ffr = open('%s/duebill.txt'%(FLDR),'r')
        invtmp = ffr.read()
        ffr.close()
        fileName = self.genPAYlink(AMNT=due,MODELno=INVno,ICONfig='ICON_pixel.png',QRname=INVno)
        TUPLEind = ('%.2f'%(total),'%.2f'%(paid),'%.2f'%(due),fileName)
        print(invtmp,TUPLEind)
        invtxt = invtmp%(TUPLEind)
        print(invtxt)
        ffw = open('%s/bill.tex'%(FLDR),'w')
        print(invtxt,file=ffw)
        ffw.close()
        os.system('pdflatex -output-directory %s %s/bill.tex'%(FLDR,FLDR))
        self.rmvinvcIMG(['invoice',INVno])
        FILEid = upload_to_folder(self.creds,'%s.pdf'%(INVno),'%s/bill.pdf'%(FLDR),parent_id=DriveFolder['invoice'],permit='writer')
        INVlink = "https://drive.google.com/file/d/%s/view?usp=share_link"%(FILEid)
        flr = open('invcimgs.dict','rb')
        invcimgs = pickle.load(flr)
        flr.close()
        IDw = SpreadSheet["finance"]
        POSw = "Sale!A%d:I%d"%(idi,idi)
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        putTOTAL=[timestamp,INVno,products,name,phn,total,paid,due,INVlink]
        #putTOTAL=[total,paid,due,INVlink]
        put_spreadsheet(creds=self.creds,putLIST=putTOTAL,putID=IDw,putPOS=POSw,write='update')
        invcimgs[INVno] = FILEid
        ffw = open('invcimgs.dict','wb')
        pickle.dump(invcimgs,ffw)
        ffw.close()
        LINK = "{}".format(INVlink)
        self.sendLINK(CHATID=userid,link=INVlink,txt="The invoice PDF is here!")
        if due==0:
            os.system('rm -rf %s'%(FLDR))
        elif due>0:
            os.system('rm %s/bill.*'%(FLDR))
        balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
        balance += amnt
        put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')

    def rmvinvcIMG(self,ARG,userid=None):
        filetype = ARG[0]
        filename = ARG[1]
        ffr = open('invcimgs.dict','rb')
        invcdict = pickle.load(ffr)
        ffr.close()
        try:
            fileid = invcdict[filename]
            delete_file(creds=self.creds,file_id=fileid)
            del invcdict[filename]
            ffw = open('invcimgs.dict','wb')
            pickle.dump(invcdict,ffw)
            ffw.close()
            TXT = 'invoice removed successfully!'
        except:
            TXT = 'this invoice does not exist!'
        try:
            self.sendUPDATE(CHATID=userid,txt=TXT)
        except:
            pass

    def custINFO(self,ARG,userid):
        subcmd = ARG[0]
        ffr = open('custdata.dict','rb')
        custdata = pickle.load(ffr)
        ffr.close()
        if subcmd == 'add':
            phn = ARG[-1]
            name = ' '.join(ARG[1:-1])
            custdata[phn]=name
        if subcmd == 'del':
            del custdata[phn]
        ffw = open('custdata.dict','wb')
        pickle.dump(custdata,ffw)
        ffw.close()

    def credit(self,amount,remark):
        timestamp = datetime.datetime.now().strftime('%m/%Y')
        idx = 0
        try:
            idx = self.search(IDw="finance",POSw="Balance!D1:D1000",keyword=timestamp,findat=0)+3
        except:
            pass
        values=[timestamp, amount]
        if idx == 0:
            if remark.lower()=='pallab':
                values=[timestamp, amount]
                put_spreadsheet(creds=self.creds,putLIST=values,putID=SpreadSheet["finance"],putPOS="Balance!E3:F"%(idx))
            elif remark.lower()=='maatu':
                values=[timestamp, '', amount]
                put_spreadsheet(creds=self.creds,putLIST=values,putID=SpreadSheet["finance"],putPOS="Balance!E3:G"%(idx))
        else:
            pass
        #put_spreadsheet(values,creds=self.creds)
        balance = float(get_spreadsheet(creds=self.creds,getID=SpreadSheet["finance"],getPOS="Balance!A3")[-1][-1])
        balance += amount
        put_spreadsheet(creds=self.creds,putLIST=[balance],putID=SpreadSheet["finance"],putPOS="Balance!A3",write='update')

    def debit(self,amount,remark):
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        values=[timestamp, amount, remark, '', '']
        put_spreadsheet(values,creds=self.creds)
        TXT = "A new transaction is done!\n\ntime: %s\ndebit: %s/-\nremarks: %s"%(timestamp,amount,remark)
        self.sendUPDATE2all(txt=TXT)
        self.balance(userid='all')

    def maketext(self,vals):
        if len(vals)==3:
            text="time: %s\ndebit: %s/-\nremarks: %s"%(vals[0],vals[1],vals[2])
        else:
            if vals[1]=='' or vals[2]=='':
                text="time: %s\ncredit: %s/-\nremarks: %s"%(vals[0],vals[3],vals[4])
            else:
                text="time: %s\ndebit: %s/-\nremarks: %s\ncredit: %s/-\nremarks: %s"%(vals[0],vals[1],vals[2],vals[3],vals[4])
        return text

    def EXIT(self,userid):
        txt = "Thanks! I am signing off."
        self.sendUPDATE(userid,txt)
        time.sleep(2)
        self.cls(userid=userid)
        threading.Thread(target=self.shutdown).start()

    def shutdown(self):
        self.updater.stop()
        self.updater.is_idle = False

    def cls(self,userid):
        smsID = self.smsID[self.dict[userid]]
        for i in smsID:
            self.BOT.delete_message(chat_id=userid, message_id=i)
        self.smsID[self.dict[userid]] = []

    def sendUPDATE2all(self,txt):
        USERs = list(self.dict.keys())
        for CHATID in USERs:
            self.sendUPDATE(CHATID,txt)

    def sendUPDATE(self,CHATID,txt):
        self.BOT.sendChatAction(chat_id=CHATID, action="typing")
        msg = self.BOT.sendMessage(chat_id=CHATID, text=txt, parse_mode=ParseMode.MARKDOWN)
        self.smsID[self.dict[CHATID]].append(msg.message_id)

    def sendLINK(self,CHATID,link,txt='link'):
        self.BOT.sendChatAction(chat_id=CHATID, action="typing")
        HTML = "<a href='%s'>%s</a>"%(link,txt)
        msg = self.BOT.sendMessage(chat_id=CHATID, text=HTML ,parse_mode=ParseMode.HTML)
        self.smsID[self.dict[CHATID]].append(msg.message_id)

    def sendPHOTO(self,CHATID,fileobj):
        self.BOT.sendChatAction(chat_id=CHATID, action="typing")
        msg = self.BOT.send_photo(chat_id=CHATID, photo=fileobj)
        self.smsID[self.dict[CHATID]].append(msg.message_id)

def main():
    k=Termibot()
    k.cmdTRIGGER()

if __name__ == "__main__":
    main()
