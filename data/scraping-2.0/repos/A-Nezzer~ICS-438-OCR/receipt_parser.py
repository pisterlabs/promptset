from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI

class ReceiptParser:
    def __init__(self, openai_api_key):
        self.model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=openai_api_key)
        self._template()
    
    def receipt_json(self, receipt_text):
        data = {"receipt_text": receipt_text}
        chain = self.fewshot_template | self.model
        return chain.invoke(data).content
    
    def _template(self):
        prompt_examples = [
            {
                "text": "129\nFor question comments or concerns\nCall McDonald's Hotline\n800-683-5587\nNow Delivering with\nDoor Dash\nSurvey Code:\n14616-01290-70423-16249-00085-\nMcDonald's Restaurant #14616\n3549 RUSSETT GREEN E (WM#1985)\nMD\nANNE\n<UNKNOWN> 20724\nTEL# 301-7767980\nThank You Valued Customer\nKS# 1\n07/04/2023 04:24 PM\nSidel\nOrder 29\n1 Happy Meal Ch Burger\n4.39\n1 Cheeseburger\nNO Pickle\n1 Extra Kids Fry\n1 Apple Juice\n1 S Apl Jc Surcharge\n1 ELEMENTAL\n1 S Grimace Bday Shake\n3.69\n1 S Shake Surcharge\nSubtotal\n8.08\nTax\n0.48\nTake-Out Total\n8.56\nCashless\n8.56\nChange\n0.00\nMER# 464239\nCARD ISSUER\nACCOUNT#\nVisa SALE\n<UNKNOWN>\nTRANSACTION AMOUNT\n8.56\nCONTACTLESS\nAUTHORIZATION CODE - 03172D\nSEQ# 035443\nAID: A0000000031010\nNow Hiring\nText MD349 To 38000\nSign up for MyMcDonald's rewards\nto earn points on future visits\n",
                "json": "{{\"merchant\":\"McDonald's Restaurant #14616\",\"address\":\"3549 RUSSETT GREEN E (WM#1985)\",\"city\":\"ANNE\",\"state\":\"MD\",\"phoneNumber\":\"TEL# 301-7767980\",\"tax\":0.48,\"total\":8.56,\"receiptDate\":\"07\/04\/2023\",\"receiptTime\":\"04:24 PM\",\"ITEMS\":[{{\"description\":\"Happy Meal Ch Burger\",\"quantity\":1,\"unitPrice\":4.39,\"totalPrice\":4.39,\"discountAmount\":0}},{{\"description\":\"Cheeseburger\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}},{{\"description\":\"Extra Kids Fry\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}},{{\"description\":\"Apple Juice\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}},{{\"description\":\"S Apl Jc Surcharge\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}},{{\"description\":\"ELEMENTAL\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}},{{\"description\":\"S Grimace Bday Shake\",\"quantity\":1,\"unitPrice\":3.69,\"totalPrice\":3.69,\"discountAmount\":0}},{{\"description\":\"S Shake Surcharge\",\"quantity\":1,\"unitPrice\":0,\"totalPrice\":0,\"discountAmount\":0}}]}}"
            },
            {
                "text": "Longs <UNKNOWN> <UNKNOWN>\n4211 WAIALAE AVE\nHONOLULU, HI 96816\n<UNKNOWN> 0781\nREG#11 TRN#3591 CSHR#0000096 STR#9220\nExtraCare Card <UNKNOWN>\n********7312\n1 CFRIO SF PEG BAG 3Z\n4.59B\n1 HLMRK EVERYDY CARD ASST\n4.99T\n1 HLMRK EVERYDY CARD ASST\n4.99T\n1 TROLLI SOUR CRAWLR 5Z\n2.79B\n4 ITEMS\nSUBTOTAL\n17.36\nHI 4.712% TAX\n<UNKNOWN>\nTOTAL\n18.18\nCHARGE\n18.18\n************9212\nRF\nVISA DEBIT\nAPPROVED# 031652\nREF# 115918\nTRAN TYPE: SALE\nAID: A0000000031010\nTC: 41D3F99C585064A1\nTERMINAL# 04025348\nNO SIGNATURE REQUIRED\nCVM: 1F0000\nTVR(95): 0000000000\nTSI(9B): 0000\nCHANGE\n.00\n3509 2203 1593 5911 17\nReturns with receipt, subject to\nCVS Return Policy, thru 08/07/2023\nRefund amount is based on price\nafter all coupons and discounts.\nJUNE 8, 2023\n6:17 PM\n<UNKNOWN>\n<UNKNOWN>\n<UNKNOWN>\nTHANK YOU. SHOP 24 HOURS AT <UNKNOWN> COM\nExtraCare Card balances as of 06/02\nYear to Date Savings\n138.21\nFill 10 prescriptions Get $5EB\nPharmacy and Health ExtraBucks\nQuantity Toward this Reward\n92\nQuantity Needed to Earn Reward\n8\nPharmacy & Health Rewards Enrollment Status\nActive Members\n4\nAccess all coupons & rewards, and\ntrack your 2% earnings in the CVS\nPharmacy <UNKNOWN>\n",
                "json": "{{\"merchant\":\"Longs\",\"address\":\"4211 WAIALAE AVE\",\"city\":\"HONOLULU\",\"state\":\"HI\",\"phoneNumber\":\"<UNKNOWN> 0781\",\"tax\":0.82,\"total\":18.18,\"receiptDate\":\"JUNE 8, 2023\",\"receiptTime\":\"6:17 PM\",\"ITEMS\":[{{\"description\":\"CFRIO SF PEG BAG 3Z\",\"quantity\":1,\"unitPrice\":4.59,\"totalPrice\":4.59,\"discountAmount\":0}},{{\"description\":\"HLMRK EVERYDY CARD ASST\",\"quantity\":2,\"unitPrice\":4.99,\"totalPrice\":9.98,\"discountAmount\":0}},{{\"description\":\"TROLLI SOUR CRAWLR 5Z\",\"quantity\":1,\"unitPrice\":2.79,\"totalPrice\":2.79,\"discountAmount\":0}}]}}"
            },
            {
                "text": "sam's club\nSelf Checkout\nCLUB MANAGER <UNKNOWN> CISNEROS\n( 808 ) 945 <UNKNOWN> 9841\n05/13/23 19:17 0827 04755 092\n9092\nAkib\nE\n423450 EZPEELSHRINF\n16.98 <UNKNOWN>\nE 980048788 DC 24PK CANF\n11.58 <UNKNOWN>\nE 980049707 HI DEPOSIT F\n1.20 H\nE 980049708 HI HANDLINGF\n0.24 H\nE 980109376 MIXED NUTS F\n9.98 T\nE 1980332988 DORITOS COOF\n5.96 <UNKNOWN>\nE 980263212 <UNKNOWN> 150F\n5.96 T\nE\n369320 BASHATI <UNKNOWN>\n19.98 <UNKNOWN>\nE U INST SV\nDORITOS COO\n1.00-N\nSUBTOTAL\n70.88\nTAX 1\n3.32\nTOTAL\n74.20\nDISCV TEND\n74.20\nDiscover Credit ***\n****\n1153 I 2\nAPPROVAL # 01430R\nAID A0000001523010\nAAC <UNKNOWN>\nTERMINAL # SC010566\n*NO SIGNATURE REQUIRED\nCHANGE DUE\n0.00\nAdditional Savings This Trip:\nSam's Instant Savings: $1.00\nNew! Free shipping for Plus members.\nLearn more: <UNKNOWN>\nVisit samsclub.com to see your savings\n#\nITEMS SOLD 8\nTC# 5250 4429 6871 2814 40\n*** MEMBER COPY ***\n",
                "json": "{{\"merchant\":\"Sam's Club\",\"address\":\"Self Checkout\",\"city\":\"<UNKNOWN>\",\"state\":\"<UNKNOWN>\",\"phoneNumber\":\"(808) 945 <UNKNOWN> 9841\",\"tax\":3.32,\"total\":74.2,\"receiptDate\":\"05\/13\/23\",\"receiptTime\":\"19:17\",\"ITEMS\":[{{\"description\":\"EZPEELSHRINF\",\"quantity\":1,\"unitPrice\":16.98,\"totalPrice\":16.98,\"discountAmount\":0}},{{\"description\":\"DC 24PK CANF\",\"quantity\":1,\"unitPrice\":11.58,\"totalPrice\":11.58,\"discountAmount\":0}},{{\"description\":\"HI DEPOSIT F\",\"quantity\":1,\"unitPrice\":1.2,\"totalPrice\":1.2,\"discountAmount\":0}},{{\"description\":\"HI HANDLINGF\",\"quantity\":1,\"unitPrice\":0.24,\"totalPrice\":0.24,\"discountAmount\":0}},{{\"description\":\"MIXED NUTS F\",\"quantity\":1,\"unitPrice\":9.98,\"totalPrice\":9.98,\"discountAmount\":0}},{{\"description\":\"DORITOS COOF\",\"quantity\":1,\"unitPrice\":5.96,\"totalPrice\":5.96,\"discountAmount\":0}},{{\"description\":\"<UNKNOWN> 150F\",\"quantity\":1,\"unitPrice\":5.96,\"totalPrice\":5.96,\"discountAmount\":0}},{{\"description\":\"BASHATI <UNKNOWN>\",\"quantity\":1,\"unitPrice\":19.98,\"totalPrice\":19.98,\"discountAmount\":0}}]}}"
            }
        ]
        example_prompt_str = """Receipt Text: {text}\n JSON:\n {open_curly} "ReceiptInfo": {json} {close_curly} """
        example_prompt = PromptTemplate(
              input_variables=["text", "json"],
              partial_variables={"open_curly": "{{", "close_curly": "}}"},
              template = example_prompt_str
            )
        self.fewshot_template = FewShotPromptTemplate(
            prefix="read through the data above and return a json object with merchant info and a list of items, with the information and items shown formatted as shown below. Do not add any additional information and make sure all items are correctly represented. Ensure the JSON response is valid JSON.\n\nExamples:\n",
            input_variables=["receipt_text"], 
            examples=prompt_examples,
            example_prompt=example_prompt,
            example_separator="\n\n",
            suffix="Receipt Text: {receipt_text}\n JSON:\n"
        )
        