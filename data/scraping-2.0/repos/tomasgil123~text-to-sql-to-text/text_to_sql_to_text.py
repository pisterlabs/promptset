from langchain import SQLDatabaseChain

class TextToSQLToTextChain():
    def __init__(self, llm, db):

        self.db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, return_direct=False)
    
    def run(self, query, number_of_tries=0):
        try:
            response = self.db_chain.run(query)
            return response
        except Exception as err:
            # if error is of type ProgrammingError, we create a new query with the error
            # we do this up to three times
            if type(err).__name__ == "ProgrammingError":
                number_of_tries += 1
                
                if number_of_tries < 3:
                    query_with_error = f"""
                        I asked this question: {query} and got this error: '{err}'. Redo the sql query in order for it to work
                    """
                    self.run(query=query_with_error, number_of_tries=number_of_tries)
                else:
                    return err
            else:
                return err