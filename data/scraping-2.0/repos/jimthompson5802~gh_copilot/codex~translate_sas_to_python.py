# Access OpenAI Codex API

import json
import openai

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to translate SAS code to Python
def translate_sas_to_python(sas_code):
    prompt = f'convert this SAS program to Python using the sklearn package:\n\n{sas_code}\n\nPython code:'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",   #'text-davinci-003',
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.0,
        n=1,
        stop=None,
    )

    python_code = response.choices[0].message.content.strip()
    return python_code

# Example SAS code to translate
sas_code = '''
/* Read in the CSV file */
proc import datafile='path_to_csv_file.csv'
     out=mydata
     dbms=csv
     replace;
     getnames=yes;
run;

/* Generate the linear regression model */
proc reg data=mydata;
     model y = x1 x2 x3; /* Replace x1, x2, x3 with your predictor variables */
run;

/* Print model diagnostics */
proc reg data=mydata;
     model y = x1 x2 x3;
     output out=model_diagnostics p pchi r rstudent;
run;

/* Print the model diagnostics */
proc print data=model_diagnostics;
run;
'''

###
# Example SAS code to translate
###

python_code = translate_sas_to_python(sas_code)
print("\n### generated simple SAS Code ###")
print(python_code)


### generated simple SAS Code ###
# import pandas as pd
# from sklearn.linear_model import LinearRegression
#
# # Read in the CSV file
# mydata = pd.read_csv('path_to_csv_file.csv')
#
# # Generate the linear regression model
# X = mydata[['x1', 'x2', 'x3']]
# y = mydata['y']
# model = LinearRegression().fit(X, y)
#
# # Print model diagnostics
# model_diagnostics = pd.DataFrame({'p': model.pvalues_, 'pchi': model.pvalues_, 'r': model.score(X, y), 'rstudent': model.score(X, y)})
# print(model_diagnostics)


####
# Example SAS macro code to translate
###

sas_macro_code = '''
%macro generate_regression_models(input_files);
  %local i;

  %do i = 1 %to %sysfunc(countw(&input_files));
    %let input_file = %scan(&input_files, &i);

    data mydata;
      infile "&input_file" dlm=',' firstobs=2;
      input x y;
    run;

    proc reg data=mydata outest=outest&i;
      model y = x;
    run;

    %put Linear regression model for &input_file has been generated;
  %end;

  %put All regression models have been generated successfully;
%mend;

%generate_regression_models(input_files = "path/to/file1.csv path/to/file2.csv path/to/file3.csv");
'''

# Translate SAS code to Python
python_macro_code = translate_sas_to_python(sas_macro_code)
print("\n### generated SAS Macro Code ###")
print(python_macro_code)

### generated SAS Macro Code ###
# from sklearn.linear_model import LinearRegression
# import pandas as pd
#
# input_files = ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"]
#
# for i in range(len(input_files)):
#   input_file = input_files[i]
#   df = pd.read_csv(input_file, delimiter=',', skiprows=1)
#   X = df['x'].values.reshape(-1,1)
#   y = df['y'].values.reshape(-1,1)
#   model = LinearRegression().fit(X, y)
#   print("Linear regression model for {} has been generated".format(input_file))
#
# print("All regression models have been generated successfully")