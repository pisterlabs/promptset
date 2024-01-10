import PyPDF2
import openpyxl
import openai
import argparse

# FUNCTION FOR FINDING THE PAGE INTERVAL OF THE NECESSARY INFORMATION ABOUT THE SPECIAL FUND

def finding_pages(pdf_path, fund, next_fund):
    pages = [0, 0]
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    
        # in this loop I find the beginning page and the end page of the information about the fund. When I find the ending page I just break the loop
        for page_num in range(num_pages): 
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            lines = [p.strip() for p in page_text.split('\n') if p.strip()]

            # I am taking the index 1, because at index 0 there is previous page number
            if lines[1] == fund:
                pages[0] = page_num
        
            if lines[1] == next_fund:
                pages[1] = page_num
                break
    return pages

# getting the full text between page intervals and find out if it is Article 8 or Article 9
def getting_full_text_and_info(pdf_path, pages):
    
    #collect the texts into one big text
    full_text = ''
    source_and_category = ()

    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(pages[0], pages[1]):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            full_text += page_text

            for word in search_words:
                for line in page_text.split('\n'):
                    if word in line:
                        source_and_category = (word, line, page_num + 1)

    return full_text, source_and_category

pdf_path = 'william.pdf'

# getting the OpenAI API key from the comman line
parser = argparse.ArgumentParser(description='Your OpenAI API key')
parser.add_argument('api_key', type=str, help='Your API key')
args = parser.parse_args()


# these are two main categories which I should find for special fund
search_words = ['Article 8', 'Article 9']
source_and_category = ()


# I initialize the fund name and the sequenced fund name. By this I find tha page where information about that fund begins and ends.
fund = 'Global Leaders Fund'
next_fund = 'Global Leaders Sustainability Fund'


pages = finding_pages(pdf_path, fund, next_fund)

necesseray_funds_text, source_and_category = getting_full_text_and_info(pdf_path, pages)
    

# Load your API key from an environment variable or secret management service
openai.api_key = args.api_key

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": necesseray_funds_text+'extract the ESG strategy related key bullet points'}])

######## Below the structure of the response. I left it to be understandable when getting data from the Open API response
#########################################################################################################################

# chat_completion = {
#   "id": "chatcmpl-80BmGig5GuOTEX78VGQq3ONl3uznj",
#   "object": "chat.completion",
#   "created": 1695055964,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "- The Investment Manager seeks to integrate environmental, social, and governance (ESG) factors into its investment process.\n- ESG factors are considered alongside traditional financial factors in the Investment Manager's fundamental analysis.\n- The Investment Manager believes that ESG factors are linked to the sustainability of competitive strengths.\n- Material ESG issues are integrated into the investment process to assess company risks and opportunities.\n- The Fund excludes companies engaged in cluster munitions manufacturing or tobacco manufacturing.\n- Companies that violate global norms and conventions are also excluded from investment.\n- The Fund seeks to avoid companies that derive a significant portion of their revenues from thermal coal mining or thermal coal power generation.\n- The Investment Manager incorporates industry accepted screening tools from reliable vendors to determine investment decisions based on ESG principles."
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 1704,
#     "completion_tokens": 153,
#     "total_tokens": 1857
#   }
# }
#################################################################################################################################
#################################################################################################################################



# saving information on the excel file
workbook = openpyxl.Workbook()
sheet = workbook.active  # Get the active sheet (usually the first sheet)

# Add data to the Excel sheet
data = [
    [fund],
    ["Article 8", 'Article 8' == source_and_category[0]],
    ["Article 9", 'Article 9' == source_and_category[0]],
    ['Other sustainability data'],
]

if  source_and_category[0] == 'Article 8':
    data[1].append('Source page ' + str(source_and_category[2]) + ':' + source_and_category[1])
else:
    data[2].append('Source page ' + str(source_and_category[2]) + ':' + source_and_category[1])

for point in chat_completion["choices"][0]['message']['content'].split('\n'):
    data.append([point])


# Iterate through the data and add it to the sheet
for row in data:
    sheet.append(row)

# Save the workbook to a file
workbook.save("my_sample_reply_1.xlsx")

# Close the workbook (optional)
workbook.close()




