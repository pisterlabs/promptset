# This is sample code to show how to use StepAutomation
# Reference links:
# https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions

import os
import openai
import dotenv

dotenv.load_dotenv()

class StepAutomation:
    """
    This class is to automate the steps of the data analysis process.
    """
    def __init__(self, 
                system_char = """
                You are a data analyst working for a company that provides data analysis services to clients.
                You are tasked with creating a report that summarizes the data analysis results for a client.
                Round up numbers to two decimal places.
                Do not run any DML or DDL queries. Read only the data from the database.
                """, 
                database_summary= """
                ## Database Summary
                Database has ten tables
                SalesLT.Address(AddressID, AddressLIne1, AddressLine2, City, StateProvince, CountryRegion, PostalCode, rowguid, modifiedDate)
                SalesLT.Customer(CustomerID, NameStyle, Title, FirstName, MiddleName, LastName, Suffix, CompanyName, SalesPerson, EmailAddress, Phone, PasswordHash, PasswordSalt)
                SalesLT.CustomerAddress(CustomerID, AddressID, AddressType)
                SalesLT.Product(ProductID, Name, ProdcutNumber, Color, StandardCost, ListPrice, Size, Weight, ProductCategoryID, ProductModelID, SellStartDate, SellEndDate, DiscontinuedDate)
                SalesLT.ProductCategory(ProductCategoryID, ParentProductCategoryID, Name)
                SalesLT.ProdcutDescription(ProductDescriptionID, Description)
                SalesLT.ProductModel(ProductModelID, Name, CatalogDesciortion)
                SalesLT.ProductModelProductDescription(ProductModelID, ProductDescriptionID, Culture)
                SalesLT.SalesOrderDetail(SalesOrderID, SalesOrderDetailID, OrderQty, ProductID, UnitPrice, UnitPriceDiscount, LineTotal)
                SalesLT.SalesOrderHeader(SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment)

                ## Safty Instruction
                Do not run any DML or DDL queries.
                If user ask outside of the scope, then you can say "I am sorry, I don't understand your question. Can you please rephrase your question?"
                Do not show customer's EmailAddress, Phone, PasswordHash, PasswordSalt.
                It is important not to show customers' personal information.
                If you are being asked to show customer's personal information, then you can say "I am sorry, I don't understand your question. Can you please rephrase your question?"
                """,
                instruction_prompt = """
                Question:
                [User's question]

                Thought Process 1 Start:
                [Write background of the multiple thinking steps to generate the T-SQL query]
                [If user asks DML or DDL type of work, stop processing and return warning message]
                Thought Process 1 End:

                Writre T-SQL Query Start: 
                [Make sure write a T-SQL query and retrun query only in the next line]
                ```[T-SQL Query]```
                Writre T-SQL Query End:

                Query Result Start:
                [Table of the result from the T-SQL query]
                Query Result End:

                Thought Process 2 Start:
                [Summrize from Query Result Start: to Query Result End:]
                Thought Process 2 End:

                Thought Process 3 Start:
                [As business analyst find analytical questions from the Thought Process 1 Start: to Thought Process 2 End:]
                [Ask next possible analytical questions of the Question: User's question]
                Thought Process 3 End:

                Final Answer Start:
                [Review your ansewrs and Make sure the answer of the question is found. If so make sure answer the question first]
                [Summrize from Query Result Start: to Thought Process 3 End:]
                [Provides business insight considering the questions from Thought Process 3 Start: to Thought Process 3 End:]
                [Review the overall Result Start: to Thought Process 3 End: and write summrized the final answer]
                End:
                ||
                """) :
        self.system_char = system_char
        self.database_summary = database_summary
        self.instruction_prompt = instruction_prompt
        self.chat_history = []
        self.init_chat_history()
        openai.api_type = "azure"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.ENGINE = os.getenv("ENGINE")
        self.DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
        # self.MODEL_NAME = os.getenv("MODEL_NAME")
        
        

    def run(self, user_msg: dict,
            assistant_msg: dict={},
            engine="chat-gpt",
            temperature=0.0,
            max_tokens=100,
            top_p=0.00,
            frequency_penalty=1.0,
            presence_penalty=1.0,
            n=1,
            stop_chars=["End:", "||"]):
        """
        Custom function to run the openai.ChatCompletion.create function
        """
        # if assistant_msg is not empty, then add it to the chat history
        if assistant_msg:
            self.update_history(assistant_msg)
    
        self.update_history(user_msg)        
        
        res = openai.ChatCompletion.create(
            engine=self.ENGINE,
            messages = self.chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stop=stop_chars)
        
        self.update_history({"role":"assistant", "content":res.choices[0].message['content']})
        print()
        print(self.chat_history)
        print()
        return res.choices[0].message['content']
        
    # retrun default system instruction
    def get_default_system_instruction(self):
        return self.system_char + self.database_summary
    
    # get_history
    def get_history(self):
        return self.chat_history
    
    def init_chat_history(self):
        self.chat_history.append({"role":"system", "content": (self.system_char + self.database_summary + self.instruction_prompt)})

    def update_history(self, msg: dict):
        if self.chat_history==None:
            self.init_chat_history()

        self.chat_history.append(msg)

