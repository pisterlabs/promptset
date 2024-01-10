import openai
import pandas as pd

deployment_name = "contract_search"
openai.api_type = "azure"
openai.api_key = "3a87ebf808cf4876b336ddbef5dd2528"
openai.api_base = "https://bpogenaiopenai.openai.azure.com/"
openai.api_version = "2023-05-15"

def questions(pdf1,i):
    pormptDict = {'Agreement Name':' What is the name of the agreement in this contract,I am seeking concise 1-2 keywords as the response',
                  'Contract Id':' What is the contract id mentioned in the agreement,I am seeking concise 1-2 keywords as the response',
                  'Payment Terms':'What is the numeric value of the payment terms agreed upon with the supplier in this agreement? Please provide a concise one-word numerical response',
                  'Service Price Per Unit':'Please provide the roles provided by the supplier along with their corresponding services prices and do not provide total prices.give only roles and prices. Ensure the values are consistent in every run.Please provide a response in the following format: [ROLE]. [PRICE]',
                  'Start date and end date of the Agreement':'When is the start date and end date of the agreement,I am seeking concise 1-2 keywords only the date response',
                  'Supplier name':'What is the name of the supplier in this contract,I am seeking concise 1-2 keywords as the response',
                  'Unit of Measure':'What is the unit of measure of the services'}
    try:
        question = pormptDict[i]
    except:
        question=i
    print("question:-",question)
    prompt = f"{question}\n{pdf1}\n\\n:"
    model = "text-davinci-003"
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=prompt,
        max_tokens=500,
        temperature=0,
        n = 1,
        stop=None
    )
    diffs = response.choices[0].text
    return diffs.strip()

def contractandpo(contract,po):
    final_po = pd.merge(contract,po,how='inner',on=['Contract_ID','Line_Item_Desc'])
    final_po = final_po[['PO_ID','Line_Item_Desc','Supplier_x','Payment_Terms_PO','Unit_Price_In_Orig_Curr','PO_Commit','Contract_ID','Payment_Terms_Contract','Contracted_Unit_Price']]
    final_po['Payment_Terms_Contract'] = final_po['Payment_Terms_Contract'].astype(int)
    final_po['Contracted_Unit_Price'] = final_po['Contracted_Unit_Price'].astype(int)
    final_po['Payment_Terms_PO'] = final_po['Payment_Terms_PO'].astype(int)
    final_po['Unit_Price_In_Orig_Curr'] = final_po['Unit_Price_In_Orig_Curr'].astype(int)
    final_po['Spend Leakage'] = ((final_po['Unit_Price_In_Orig_Curr'] - final_po['Contracted_Unit_Price'])/final_po['Contracted_Unit_Price'])*100
    final_po['Spend Leakage'] = final_po['Spend Leakage'].astype(int)
    final_po['Payter_Diff_PO_Contract'] = (final_po['Payment_Terms_Contract'] - final_po['Payment_Terms_PO'])
    openai.api_key = "3a87ebf808cf4876b336ddbef5dd2528"
    # prompt_s = ['provide the key differences in price and payterms from the contract and PO']
    # prompt_s = ['provide the difference of price and payterms in the table:-sum df :-- The difference of price and payterms in the table is 20.']
    # prompt_s = ['provide the difference of unit price and payterm from the data and also provide key points from the data']
    prompt_s =['provide me the total spend leakage and the average differences between the po and contract payterm']
   
    summary = ''
    summary+= (questions(final_po,prompt_s))
    return final_po,summary

def contractandinvoice(contract,invoice):
    final_df = pd.merge(contract,invoice,how='inner',on=['Contract_ID','Line_Item_Desc'])
    final_df = final_df[['Doc_Num','Line_Item_Desc','Inv_Gross_Value','Inv_Payterms','Inv_Unit_Price',
                         'Payment_Terms_Contract','Contracted_Unit_Price','Unit_Price_In_Orig_Curr','Contract_ID']]
    final_df['Payment_Terms_Contract'] = final_df['Payment_Terms_Contract'].astype(int)
    # final_df['Payment_Terms_PO'] = final_df['Payment_Terms_PO'].astype(int)
    final_df['Inv_Payterms'] = final_df['Inv_Payterms'].astype(int)
    final_df['Inv_Gross_Value'] = final_df['Inv_Gross_Value'].astype(int)
    final_df['Inv_Unit_Price'] = final_df['Inv_Unit_Price'].astype(int)
    final_df['Contracted_Unit_Price'] = final_df['Contracted_Unit_Price'].astype(int)
    final_df['Unit_Price_In_Orig_Curr'] = final_df['Unit_Price_In_Orig_Curr'].astype(int)
    final_df['Spend Leakage'] = ((final_df['Inv_Unit_Price'] - final_df['Contracted_Unit_Price'])/final_df['Contracted_Unit_Price'])*100
    final_df['Spend Leakage'] = final_df['Spend Leakage'].astype(int)
    final_df['Working_Capital_Inv_Contract'] = ((final_df['Payment_Terms_Contract'] - final_df['Inv_Payterms'])*final_df['Inv_Gross_Value'])/365
    final_df = final_df.rename(columns={'INVOICE': 'Doc_Numb', 'PO': 'PO_ID','Unit_Price_In_Orig_Curr':'PO_Unit_Price',
                                        'Product Description':'Line_Item_Desc'})
    final_df = final_df[['Doc_Num','Inv_Payterms','Inv_Unit_Price',
                         'PO_Unit_Price','Contract_ID','Payment_Terms_Contract',
                         'Contracted_Unit_Price','Spend Leakage','Working_Capital_Inv_Contract']]        
    openai.api_key = "3a87ebf808cf4876b336ddbef5dd2528"
    prompt_s = ['provide me the total spend leakage and the total cash flow ooportunity contract and invoice']
    summary = ''
    summary+= (questions(final_df,prompt_s))
    return final_df,summary

def threewaymatch(contract,po,invoice):
    final_df = pd.merge(pd.merge(contract,invoice,how='inner',on=['Contract_ID','Line_Item_Desc']),po,on=['Contract_ID','Line_Item_Desc'])
    final_df = final_df[['Doc_Num','Supplier_x','Contract_ID','Line_Item_Desc','Inv_Payterms','Inv_Unit_Price','Payment_Terms_Contract','Payment_Terms_PO','Contracted_Unit_Price','Inv_Gross_Value','Unit_Price_In_Orig_Curr_x']]
    final_df['Payment_Terms_Contract'] = final_df['Payment_Terms_Contract'].astype(int)
    # final_df['Payment_Terms_PO'] = final_df['Payment_Terms_PO'].astype(int)
    final_df['Inv_Payterms'] = final_df['Inv_Payterms'].astype(int)
    final_df['Inv_Gross_Value'] = final_df['Inv_Gross_Value'].astype(int)
    final_df['Inv_Unit_Price'] = final_df['Inv_Unit_Price'].astype(int)
    final_df['Contracted_Unit_Price'] = final_df['Contracted_Unit_Price'].astype(int)
    final_df['Unit_Price_In_Orig_Curr'] = final_df['Unit_Price_In_Orig_Curr_x'].astype(int)
    final_df['Spend Leakage'] = ((final_df['Inv_Unit_Price'] - final_df['Contracted_Unit_Price'])/final_df['Contracted_Unit_Price'])*100
    final_df['Spend Leakage'] = final_df['Spend Leakage'].astype(int)
    final_df['Working_Capital_Inv_Contract'] = ((final_df['Payment_Terms_Contract'] - final_df['Inv_Payterms'])*final_df['Inv_Gross_Value'])/365
    final_df = final_df.rename(columns={'INVOICE': 'Doc_Num','Unit_Price_In_Orig_Curr':'PO_Unit_Price','Product Description':'Line_Item_Desc'})
    final_df = final_df[['Doc_Num','Supplier_x','Line_Item_Desc','Inv_Payterms','Inv_Unit_Price','PO_Unit_Price','Contract_ID','Payment_Terms_Contract','Payment_Terms_PO','Contracted_Unit_Price','Inv_Gross_Value','Spend Leakage','Working_Capital_Inv_Contract']]        
    final_df = final_df.rename(columns={ 'Payment_Terms_Contract':'Contract_Payterms','Supplier_x':'Supplier','Unit_Price_In_Orig_Curr':'PO_Unit_Price','Product Description':'Line_Item_Desc','Payment_Terms_PO':'PO_Payterms'})
    final_s = final_df[['Line_Item_Desc','Supplier','Contracted_Unit_Price','PO_Unit_Price','Inv_Unit_Price','Contract_Payterms','Inv_Payterms','PO_Payterms']]
    openai.api_key = "3a87ebf808cf4876b336ddbef5dd2528"
    prompt_s = ['provide the differences in unit price across contract, PO and Invoice from the table in words and also provide summary of contract,po and Inv_Payterms']
    summary = ''
    summary+= (questions(final_s,prompt_s))
    return final_s,summary

# contractDummy =pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/dummy/CO_12L23.xlsx")
# po = pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/upload_3way/PO.xlsx")
# resultDF, reslutSumm=contractandpo(contract=contractDummy,po=po)
# print("result df :--", resultDF)
# print("sum df :--", reslutSumm)


# contractDummy =pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/dummy/C0_1L976.xlsx")
# invoice = pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/upload_3way/Invoice data_Updated.xlsx")
# resultDF, reslutSumm=contractandinvoice(contract=contractDummy,invoice=invoice)
# print("result df :--", resultDF)
# print("result df :--", reslutSumm)

# contractDummy =pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/dummy/C0_1L976.xlsx")
# invoice = pd.read_excel("C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract - v4.1/upload_3way/Invoice data.xlsx")
# resultDF, reslutSumm=threewaymatch(contract=contractDummy,po=po,invoice=invoice)
# print("result df :--", resultDF)
# print("result df :--", reslutSumm)
