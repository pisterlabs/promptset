import datetime
import os
import json
from openaillm import openai_pricing 

lastRetrievedDateTimeFieldName = "lastRetrievedDateTime"
pricing_details_file_name = 'last_pricing_response.json'

def readPricingInfoFile():
    if os.path.exists(pricing_details_file_name):
        with open(pricing_details_file_name, 'r') as f:
            pricing_details = json.load(f)  
            return pricing_details
    return None

def writePricingInfoFile(pricing_details,current_datetime):
     # get string of current datetime in "%Y-%m-%dT%H:%M:%S.%f" format
     current_datetime_iso_string = current_datetime.isoformat()
     pricing_details[lastRetrievedDateTimeFieldName] = current_datetime_iso_string
     pricing_details_json = json.dumps(pricing_details)
     with open(pricing_details_file_name, "w") as f:
            f.write(pricing_details_json)
            
# def writePricingResponse(pricingPageRawHtml,current_datetime):
#     current_datetimeiso = current_datetime.isoformat()
#     with open(current_datetimeiso+"_pricing_response.html", "w") as f:
#             f.write(pricingPageRawHtml)

def main(llmType="openai"):
    if llmType == "openai":
        #check if file exists
        pricingInfoFromFile = readPricingInfoFile()
        #read last_pricing_response.json file
        current_datetime = datetime.datetime.now()
        if not pricingInfoFromFile == None:
            recordDate = getLastRetrievedDateTime(pricingInfoFromFile)
            # check if the datetime in the last_pricing_response.json is older than 1 day
            if (current_datetime - recordDate).days < 1:
                return pricingInfoFromFile
        pricing_details = openai_pricing.getPricingDetailsFromOpenAI(current_datetime)
        writePricingInfoFile(pricing_details,current_datetime)
        return pricing_details
    
    return "llmType not defined"


def getLastRetrievedDateTime(pricingInfoFromFile):
    recordDate = datetime.datetime.strptime(pricingInfoFromFile[lastRetrievedDateTimeFieldName], "%Y-%m-%dT%H:%M:%S.%f")
    return recordDate

# if __name__ == "__main__":
#     main()