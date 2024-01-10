# NL2SQL Functions
import pandas as pd
from sqlalchemy import exc, create_engine, text as sql_text
import openai
from scipy import spatial  # for calculating vector similarities for search
from ast import literal_eval # Function used to cast embedding vectors stored as strings to arrays

##############################################################################
# Prepares Basic Template
def Prepare_Message_Template(Template_Filename = None, Verbose=False, Debug=False):
        # Import Mesage Template file
        try:
            with open(Template_Filename, 'r') as file:
                Template = file.read().replace('\n', '')
        except:
            print(f'Prompt file {Template_Filename} load failed ')
            return  []
        
        Messages = [{"role": "system", "content": Template}]
        if Debug:
            print(f'Prepare Message Template: \n {Messages} \n end \n')
        return Messages
    
##############################################################################
# Run SQL Query against MYSQL DB

def Run_Query(Credentials=None, Query=None, Verbose=False):
    """
    """

    #Unpack DB Credentials 
    MYSQL_User = Credentials['User']
    MYSQL_PWD = Credentials['PWD']
    try:
        Conn = create_engine(f'mysql+mysqldb://{MYSQL_User}:{MYSQL_PWD}@localhost/fakebank')
    except exc.SQLAlchemyError:
        print('failed to connect to MYSQL DB')
        return -1
    try:
        df = pd.read_sql_query(con=Conn.connect(),sql=sql_text(Query))
        return df
    except exc.SQLAlchemyError:
        if Verbose:
            print('read_sql_query error - MySQL')
            print(f'Query Error sql_text {sql_text(Query)}' )
            print(f'returned message {df}')
        status = -5
        # return an empty dataframe
        return pd.DataFrame()
##############################################################################
#

# Export_df
def Export_df(df, Filename, WD=None, Delimator = ",", Verbose=False):   
        # Determine File type (txt with deliminator, xlsx, csvs, etc ...)
        # strip the suffix, e.g. .csv, .xlsx, assume filename is given as prefix-i.suffix

        Suffix = Filename[-4:].replace('.','')
        if Suffix in ({'csv', 'txt', 'xlsx'}):
            Format = Suffix
        else:
            print(f'Output_results_df: Unsupported Filetype {Suffix}')
            return 0
        if WD is not None:
            Filename = f"{WD}/{Filename}" 

        n = len(df)
        # Convert embedding vector to n columns in dataframe before saving
        if Format == 'xlsx':
            try:
                df.to_excel(Filename,sheet_name='DF',index=False, header=True)
                msg = f'{n} rows were exported to {Filename}'
                return msg
            except:
                msg = f'Failed to write to {Filename}'
                return msg
        elif Format == 'csv':
            try:
                Delimator = ","
                df.to_csv(Filename,header=True, index=False, sep = Delimator) 
                msg = f'{n} rows were exported to {Filename}'
                return msg
            except:
                msg = f'Failed to write to {Filename}'
                return msg
        elif Format == 'txt':
            try:
                Delimator = "\t"
                df.to_csv(Filename,header=True, index=False, sep = Delimator) 
                msg = f'{n} rows were exported to {Filename}'
                return msg
            except:
                msg = f'Failed to write to {Filename}'
                return msg
        else:
            print(f"Output_results_df: File format {Format} is not supported")
