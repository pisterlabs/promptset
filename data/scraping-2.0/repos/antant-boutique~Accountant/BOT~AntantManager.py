from __future__ import print_function

import os.path

adminID = '1651529355'  # Pallab Chat-ID

import time
import datetime
import pickle
from tqdm.contrib.telegram import tqdm, trange
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
import pandas as pd
from GDriveFunctions import *


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


def addtxtl(DATA):
    sources = DATA['source']
    materials = DATA['material']
    colors = DATA['color']
    sizes = np.array(DATA['quantity'],dtype=np.float)
    prices = np.array(DATA['price'],dtype=np.float)
    ppls = prices/sizes
    fhr = open('matcdn.dict','rb')
    matDict = pickle.load(fhr)
    fhr.close()
    TXT = "The following materials are recently added.\n\n"
    for i,fabric in enumerate(materials):
        fabric = fabric.strip()
        fabric = fabric.replace(' ','-')
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
        df = pd.read_csv('Antant_materials.csv')
        filtered_df = df[df['MatCode'].str.contains(cdn, case=True)]
        if len(filtered_df)==0:
            new_cdn = cdn+'-1'
        else:
            last_cdn = filtered_df.iloc[-1]['MatCode']
            new_cdn = cdn+'-'+str(eval(last_cdn[last_cdn.rfind('-')+1:])+1)
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        new_row = [timestamp,new_cdn,sources[i].strip(),fabric,colors[i].strip(),sizes[i],prices[i],ppls[i],0,ppls[i]]
        df = df.append(pd.Series(new_row, index=df.columns), ignore_index=True)
        sl_no = i+1
        TXT += f"{sl_no}. *{new_cdn}*: {colors[i]} {fabric.strip()}, {sizes[i]} mtrs.\n"
    df.to_csv('Antant_materials.csv', index=False)
    return TXT

def get_service_charge(X,attribute):
    try:
        Y = X[attribute][0]
        Y = 0 if Y=='' else float(Y) 
    except KeyError:
        Y = 0
    return Y

def get_checkbox_value(X,attribute):
    try:
        Y = X[attribute][0]
    except KeyError:
        Y = 'off'
    Y = True if Y=='on' else False
    return Y

def design(DATA):
    print(DATA)
    Poll = False
    materials = [mat.strip() for mat in DATA['materials[]']]
    measures = [0 if mes=='' else float(mes) for mes in DATA['measures[]']]
    handBlockCost = get_service_charge(DATA,'handBlockCost')
    handPaintCost = get_service_charge(DATA,'handPaintCost')
    handEmbrdCost = get_service_charge(DATA,'handEmbroideryCost')
    handApplqCost = get_service_charge(DATA,'handAppliqueCost')
    tailoringCost = get_service_charge(DATA,'tailoringCost')
    mergeMaterials = get_checkbox_value(DATA,'mergeMaterials')
    category = DATA['category'][0].strip()
    combineWith = DATA['combineWith'][0].strip()
    suggestPrice = get_checkbox_value(DATA,'suggestPrice')
    productPrice = get_service_charge(DATA,'productPrice')
    mergeLimit = 1 if mergeMaterials else len(materials)
    matdf = pd.read_csv('Antant_materials.csv')
    desdf = pd.read_csv('Antant_designs.csv')
    for i,matmes in enumerate(zip(materials,measures)):
        mat,mes = matmes
        frac = 0
        idx = False
        print(mat)
        if mat in desdf['MatCode'].to_list() and mes != 0:
            idx = desdf.index[desdf['MatCode']==mat][0]
            totmes = desdf.loc[idx,'Length (m)']
            updcat = desdf.loc[idx,'Added_to_catalogue']
            if updcat == 1:
                TXT = f"*{mat}* is already uploaded in product catalogue. We cannot design this anymore!"
                break
            if mes<=totmes:
                matcdn = mat[:mat.rfind('-P')]
                idl = desdf.index[desdf['MatCode'].str.contains(matcdn, case=True)].to_list()[-1]
                lastcdn = desdf.loc[idl,'MatCode']
                number = eval(lastcdn[lastcdn.rfind('-P')+2:])+1
                des_editColumns = desdf.columns[2:9].to_list()
                frac = mes/totmes
                remn = 1-frac
                newdf = desdf.loc[idx]
                if frac == 1:
                    newidx = idx
                    newcdn = mat
                else:
                    newcdn = f"{matcdn}-P{number}"
                    newdf.loc['MatCode'] = newcdn
                    newdf.loc[des_editColumns] *= round(frac,2)
                    newidx = len(desdf)
                    desdf.loc[idx,des_editColumns] *= round(remn,2)
                #desdf = pd.concat([desdf, newdf], ignore_index=True)
        elif mat in matdf['MatCode'].to_list() and mes != 0:
            idx = matdf.index[matdf['MatCode']==mat][0]
            totmes = matdf.loc[idx,'Length (m)']
            if mes<=totmes:
                matcdn = mat
                idlist = desdf.index[desdf['MatCode']==mat].to_list()
                if len(idlist)==0:
                    number = 1
                else:
                    idl = idlist[-1]
                    lastcdn = desdf.loc[idl,'MatCode']
                    number = eval(lastcdn[lastcdn.rfind('-P')+2:])+1
                mat_editColumns = matdf.columns[5:7].to_list()
                des_editColumns = desdf.columns.to_list()
                frac = mes/totmes
                remn = 1-frac
                newdf = desdf.loc[0]
                newcdn = f"{matcdn}-P{number}"
                TPPL = float(matdf.loc[idx,'TPPL (Total Price per Length)'])
                timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                print(TPPL,totmes,frac)
                newdat = [timestamp,newcdn,TPPL*totmes*frac,totmes*frac,0,0,0,0,0,0,np.nan]
                print(newdat)
                #newdf.loc['MatCode'] = newcdn
                newdf.loc[des_editColumns] = newdat
                newidx = len(desdf)
                matdf.loc[idx,mat_editColumns] *= round(remn,2)
                #desdf = pd.concat([desdf, newdf], ignore_index=True)
        else:
            if mes != 0:
                TXT = "This materia does not exist in the entire record."
                break
            else:
                TXT = "Please provide a non-zero value in the measures of the design form."
                break
        if i < mergeLimit:
            artColumns = desdf.columns[4:7].to_list()
            artCosts = [handBlockCost+handPaintCost, handEmbrdCost+handApplqCost, tailoringCost]
            baseCost = newdf.loc['Base']
            addExpns = 0 if np.isnan(newdf.loc['Added expenses']) else newdf.loc['Added expenses']
            prodCost = baseCost+np.sum(artCosts)+addExpns
            if i == 0:
                MatMaster = newcdn
                newdf.loc[artColumns] = artCosts
                newdf.loc['Production cost'] = prodCost
            if i > 0:
                newdf.loc['MatMaster'] = MatMaster
            desdf.loc[newidx] = newdf
        #if i < mergeLimit:
        #    artColumns = desdf.columns[4:7].to_list()
        #    artCosts = [handBlockCost+handPaintCost, handEmbrdCost+handApplqCost, tailoringCost]
        #    desdf.loc[idx,artColumns] = artCosts
        matdf.to_csv('Antant_materials.csv')
        desdf.to_csv('Antant_designs.csv')
        TXT = 'The materials are getting designed'
        if category:
            if suggestPrice or productPrice:
                TXT,Poll = Finish(newidx,category,combineWith,productPrice,suggestPrice)
            else:
                TXT = "Either provide a product price, or turn on the price suggestion."
        else:
            if suggestPrice or productPrice:
                TXT = "Please provide a product category."
            #Finish(newidx,category,combineWith,productPrice,suggestPrice)
    return TXT,Poll

def Finish(DesignIdx,Category,Combination,ProductPrice,PriceSuggestion):
    desdf = pd.read_csv('Antant_designs.csv')
    prddf = pd.read_csv('Antant_products.csv')
    desCode = desdf.loc[DesignIdx,'MatCode']
    prodCost = desdf.loc[DesignIdx,'Production cost']
    print(desCode,Category)
    cdn = gencode(otype='product',keys=[desCode,Category])
    idlist = prddf.index[prddf['Model No.']==cdn].to_list()
    if len(idlist)==0:
        number = 1
    else:
        idl = idlist[-1]
        lastcdn = prddf.loc[idl,'Model No.']
        number = eval(lastcdn[lastcdn.rfind('-')+1:])+1
    ModelNo = f"{cdn}-{number}"
    frh = open('prodtype.list','rb')
    prodList = pickle.load(frh)
    frh.close()
    prodArr = np.array(prodList)
    sid = np.flatnonzero(prodArr[:,1]==Category)[0]
    prodID = str(prodArr[sid,0])
    prodHRK = getProd_hrk(prodArr, prodID).split(' ')
    Category = ' > '.join(prodHRK)
    #price = float(prodCOST)/float(quantity)+float(profit)
    timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    quantity = 1
    if PriceSuggestion:
        price = ''
        #Poll = True if ProductPrice == 0 else ProductPrice
        Poll = [ModelNo] + SuggestProductPrice(prodCost,ProductPrice)
        print(Poll==True)
    else:
        price = ProductPrice
        Poll = False
    newdf = [timestamp, desCode, ModelNo, Category, '', '', quantity, price, '', Combination, '', '']
    print(newdf)
    newidx = len(prddf)
    prddf.loc[newidx] = newdf
    prddf.to_csv('Antant_products.csv')
    TXT = f"The product: *{ModelNo}* ({Category}) is created."
    return TXT,Poll

def round50(X):
    X = round(X)
    remd = X%50
    if remd<=25:
        Y = X-remd
    else:
        Y = X-remd+50
    return Y

def SuggestProductPrice(ProductionCost,ProductPrice):
    Options = []
    Prices = []
    if ProductPrice:
        Options.append(ProductPrice)
        Prices.append(ProductPrice)
    Profit = [40,50,55] # percent
    for p in Profit:
        PP = round50(ProductionCost*100/(100-p))
        optstr = f"{p}% profit - {PP}"
        Options.append(optstr)
        Prices.append(PP)
    return [Options,Prices]

def setProdPrice(ModelNo,ProductPrice):
    prddf = pd.read_csv('Antant_products.csv')
    idx = prddf.index[prddf['Model No.']==ModelNo][0]
    prddf.loc[idx,'Price'] = ProductPrice
    prddf.to_csv('Antant_products.csv')
    TXT = f"The product (*{ModelNo}*) price is set to: {ProductPrice} INR"
    return TXT

def get_attr_val(X,attribute):
    try:
        Y = X[attribute][0]
        Y = 0 if Y=='' else float(Y)
    except KeyError:
        Y = 0
    return Y

def genPAYlink(AMNT,INVCno,ICONfig,QRname):
    logo = Image.open(ICONfig)
    BOXsize = 10
    BORDER = 1
    QRcode = qrcode.QRCode(box_size=BOXsize,border=BORDER,error_correction=qrcode.constants.ERROR_CORRECT_H)
    link = f"upi://pay?pa=7044319466-1@okbizaxis&pn=Antant for invoice {INVCno}&am={AMNT}&tn=Antant Invoice: {INVCno}&cu=INR"
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

def bill(DATA,false=False):
    print(DATA)
    prddf = pd.read_csv('Antant_products.csv')
    custName = DATA['customerName'][0].strip()
    custPhone = DATA['customerContact'][0].strip()
    custAddr = DATA['customerAddress'][0].strip()
    ModelNo = [prod.strip() for prod in DATA['models[]']]
    Quantity = [0 if mes=='' else float(mes) for mes in DATA['quantities[]']]
    numItems = len(Quantity)
    discount = get_attr_val(DATA,'addDiscount')
    fullpaid = get_checkbox_value(DATA,'fullpaid')
    paid = get_attr_val(DATA,'paidAmount')
    upiQR = get_checkbox_value(DATA,'upiQR')
    itqt = {ModelNo[i]:Quantity[i] for i in range(numItems)}
    total = 0
    itemROWin = ""
    itemROW = "    \centering %d & \centering %s & \centering %s & \centering %.2f & \centering %d & %.2f \\\\[2.5ex]\hline\n    & & & & &\\\\\n"
    for sl,i in enumerate(itqt):
        print(itqt)
        idp = prddf.index[prddf['Model No.']==i][0]
        stock = prddf.loc[idp,'Quantity']
        if stock>0:
            print('i,idp: ',i)
            name = prddf.loc[idp,'Category'].split(' > ')[-1]
            code = i
            rate = round(prddf.loc[idp,'Price'])
            qnty = itqt[i]
            amnt = rate*qnty
            itemROWin += itemROW%(sl+1,name,code,rate,qnty,amnt)
            total += amnt
            stock -= qnty
        else:
            TXT = "Product:%s is out of stock!"%(i)
        if false:
            pass
        else:
            prddf.loc[idp,'Quantity'] = stock
    billtype = 'discount' if discount else 'regular'
    if paid and fullpaid and paid < total:
        discount = round((total-paid)/total*100,2)
        disc = total-paid
        billtype = 'discount'
        total = paid
    if billtype=='regular':
        ffr = open('invoice_template_regular.txt','r')
    else:
        if paid and fullpaid:
            pass
        else:
            disc = total*(discount/100)
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
    OUTs = []
    if upiQR:
        fileTemp = genPAYlink(AMNT=due,INVCno=INVno,ICONfig='ICON.png',QRname=INVno+'red')
        OUTs.append(fileTemp)
        #ffr = open(fileTemp,'rb')
        #awa
        #self.sendPHOTO(CHATID=userid,fileobj=ffr)
        #ffr.close()
    #os.remove(fileTemp)
    fileName = genPAYlink(AMNT=due,INVCno=INVno,ICONfig='ICON_pixel.png',QRname=INVno)
    fcr = open('custdata.dict','rb')
    custdict = pickle.load(fcr)
    fcr.close()
    custInfo = custdict[custPhone] + ' | ' + custPhone
    if billtype=='regular':
        TUPLEin = (INVno,custInfo,itemROWin,'%.2f'%(total),'%.2f'%(paid),'%.2f'%(due),fileName)
    else:
        TUPLEin = (INVno,custInfo,itemROWin,str(discount)+'\%','%.2f'%(disc),'%.2f'%(total),'%.2f'%(paid),'%.2f'%(due),fileName)
    invtxt = invtmp%TUPLEin
    FLDR = 'INVOICE-%s'%(INVno)
    os.mkdir(FLDR)
    ffw = open('%s/bill.tex'%(FLDR),'w')
    print(invtxt,file=ffw)
    ffw.close()
    os.system('cp ICON.png %s/'%(FLDR))
    os.system('pdflatex -output-directory %s %s/bill.tex'%(FLDR,FLDR))
    OUTs.append('%s/bill.pdf'%(FLDR))
    FILEid = upload_to_folder('%s.pdf'%(INVno),'%s/bill.pdf'%(FLDR),parent_id=DriveFolder['invoice'],permit='writer')
    INVlink = "https://drive.google.com/file/d/%s/view?usp=share_link"%(FILEid)
    ##IDw = SpreadSheet["finance"]
    ##POSw = "Sale!A2:I"
    saledf = pd.read_csv('./Antant_finance_Sale.csv')
    saleidx = len(saledf)
    timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    products = ','.join(list(itqt.keys()))
    putTOTAL=[timestamp,INVno,products,custName,custPhone,total,paid,due,INVlink]
    saledf.loc[saleidx] = putTOTAL
    ##put_spreadsheet(creds=self.creds,putLIST=putTOTAL,putID=IDw,putPOS=POSw)
    invcimgs[INVno] = FILEid
    ffw = open('invcimgs.dict','wb')
    pickle.dump(invcimgs,ffw)
    ffw.close()
    LINK = "{}".format(INVlink)
    ##self.sendLINK(CHATID=userid,link=INVlink,txt="The invoice PDF is here!")
    balancedf = pd.read_csv('Antant_finance_Balance.csv')
    if false:
        pass
    else:
        balance = float(balancedf.loc[1,'Current'])
        balance += float(paid)
        balancedf.loc[1,'Current'] = str(balance)
    if due==0:
        os.system('rm -rf %s'%(FLDR))
    elif due>0:
        if billtype=='regular':
            TUPLEind = (INVno,custInfo,itemROWin,'%s','%s','%s','%s')
        else:
            TUPLEind = (INVno,custInfo,itemROWin,str(discount)+'\%%','%.2f'%(disc),'%s','%s','%s','%s')
        invtxt = invtmp%TUPLEind
        ffw = open('%s/duebill.txt'%(FLDR),'w')
        print(invtxt,file=ffw)
        ffw.close()
        os.system('rm %s/bill.*'%(FLDR))
    TXT1 = f"Dear {custName},\n\nPlease find your invoice ({INVno})  at {INVlink}. We recommend you to download the invoice since it will be auto-deleted in a month.\n\nAntant Boutique :)"
    TXT2 = [f'{custName}',f'{custPhone}']
    OUTs.append(TXT1)
    OUTs.append(TXT2)
    print(OUTs)
    prddf.to_csv('Antant_products.csv')
    saledf.to_csv('Antant_finance_Sale.csv')
    balancedf.to_csv('Antant_finance_Balance.csv')
    return OUTs

