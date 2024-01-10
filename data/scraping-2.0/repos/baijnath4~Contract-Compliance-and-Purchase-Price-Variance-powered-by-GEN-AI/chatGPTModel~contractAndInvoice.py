import openai
import pandas as pd
def contractandinvoice(contract,invoice):
    final_df = pd.merge(contract,invoice,how='inner',on=['Contract_ID','Line Item Description'])
    final_df = final_df[['INVOICE','PO','Contract_ID','Line Item Description','PAYMENTTERMS_PO','Invoice Payterms','Invoice unit price','Contracted Payterms','Contracted unit price','Invoice Gross Value','UNITPRICEINORIGINALCURRENCY']]
    final_df['Contracted Payterms'] = final_df['Contracted Payterms'].astype(int)
    final_df['PAYMENTTERMS_PO'] = final_df['PAYMENTTERMS_PO'].astype(int)
    final_df['Invoice Payterms'] = final_df['Invoice Payterms'].astype(int)
    final_df['Invoice Gross Value'] = final_df['Invoice Gross Value'].astype(int)
    final_df['Invoice unit price'] = final_df['Invoice unit price'].astype(int)
    final_df['Contracted unit price'] = final_df['Contracted unit price'].astype(int)
    final_df['UNITPRICEINORIGINALCURRENCY'] = final_df['UNITPRICEINORIGINALCURRENCY'].astype(int)
    final_df['Purchase Price variance(Invoice vs Contract)'] = ((final_df['Invoice unit price'] - final_df['Contracted unit price'])/final_df['Contracted unit price'])*100
    final_df['Purchase Price variance(Invoice vs Contract)'] = final_df['Purchase Price variance(Invoice vs Contract)'].astype(int)
    final_df['Working Capital(Invoice vs Contract)'] = ((final_df['Contracted Payterms'] - final_df['Invoice Payterms'])*final_df['Invoice Gross Value'])/365
    final_df = final_df.rename(columns={'INVOICE': 'Document Number', 'PO': 'PO_ID','UNITPRICEINORIGINALCURRENCY':'PO Unit Price','Product Description':'Line item description'})
    final_df = final_df[['Document Number','Line Item Description','Invoice Payterms','Invoice unit price','PO_ID','PAYMENTTERMS_PO','PO Unit Price','Contract_ID','Contracted Payterms','Contracted unit price','Invoice Gross Value','Purchase Price variance(Invoice vs Contract)','Working Capital(Invoice vs Contract)']]        
    openai.api_key = ""
    prompt_s = ['provide the summary of the below data:']
    summary = ''
    summary+= (questions(final_df,prompt_s))
    return final_df,summary