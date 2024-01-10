# Noah Hicks
# A program to keep all of my functions for N/N, C/N, and C/C relationships using Pandas and Seaborn on CSV files.
#   Also includes data wrangling functions.

import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import chi2_contingency
import openai as ai

# OpenAI API key (keep this private) and model
ai.api_key = 'Put your API key here' # Ideally, use an environment variable to keep this private
model = "gpt-3.5-turbo" 
max_tokens = 200

# OpenAI Functions
def generate_text_with_chatgpt(results):
    try:
        results_str = str(results)
        messages = [
            {"role": "system", "content": f"You are a helpful statistical assistant called 'CSV-ME', you are direct. You are very concise with your answers. You have {max_tokens} tokens to use. If the results you receive are long, keep your response short."},
            {"role": "user", "content": "Given the following results: " + results_str + ", provide some context and explanation of what the results mean."},
        ]
        response = ai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        text = ("CSV-ME: " + response['choices'][0]['message']['content'].strip())
        return text
    except ai.error.OpenAIError as e:
        print(f"An error occurred while generating the text with {model}: {e}")
        return


# Data Wrangling Functions________________________________________________________________________________________
def DFInfo (df):
    try:
        print('---------------------\nShape of Dataframe: ' + str(df.shape)) # Will show the number of rows and columns in the data set

        print(df.info()) # Will show the data types of each column

        print(df.describe())  # Will show summary statistics for numeric columns

        print(str(generate_text_with_chatgpt("Say hi."))) # Uses an AI to generate a description of the dataframe

        print('---------------------\nList of Null Values:\n' + str(df.isnull().sum())) # Will show the number of null values in each column

        return
    except Exception as e:
        print(f"An error occurred while getting the dataframe info: {e}")

        return

def deleteColumnNulls (df, column):
    try:
        print('Before Deletion: ' + str(df.shape)) # Will show the number of rows and columns in the data set before the nulls are deleted

        df = df.dropna(subset=[column]) # Deletes all rows with null values in the specified column

        print(df.isnull().sum()) # Will show the number of null values in each column

        print('After Deletion: ' + str(df.shape)) # Will show the number of rows and columns in the data set

        return df # Returns the new dataframe
    except:
        print('An error occurred while deleting the null values')

        return

def deleteColumn (df, column):
    try:
        print('Before Deletion: ' + str(df.shape)) # Will show the number of rows and columns in the data set before the column is deleted

        df = df.drop([column], axis=1) # Deletes the specified column

        print('After Deletion: ' + str(df.shape)) # Will show the number of rows and columns in the data set

        return df # Returns the new dataframe
    except:
        print('An error occurred while deleting the column')

        return
    
def columnDataStats (df, column):
    # Uses df.describe() to get the data for each column. Also shows meadian, mode, standard deviation, skewness and kurtosis.
    try:
        print('---------------------\n' + column + ' Data Stats:\n' + str(df[column].describe())) # Will show the number of rows and columns in the data set
        print('Median: ' + str(df[column].median()))    # Median
        print('Mode: ' + str(df[column].mode()))        # Mode  
        print('Standard Deviation: ' + str(df[column].std())) # Standard Deviation
        print('Skewness: ' + str(df[column].skew()))    # Skewness
        print('Kurtosis: ' + str(df[column].kurt()))    # Kurtosis

        # AI Generated Description
        print(str(generate_text_with_chatgpt(f'{column} Data Stats: {str(df[column].describe())} Please provide some context and explanation of what the results mean.'))) # Uses an AI to generate a description of the column

        return
    except Exception as e:
        print(f"An error occurred while getting the column data stats: {e}")

        return
    
def changeColumnName (df, oldName, newName):
    try:
        print('Before Change:\n' + str(df.columns)) # Will show the column names before the change

        df = df.rename(columns={oldName: newName}) # Changes the name of the specified column

        print('After Change:\n' + str(df.columns)) # Will show the column names after the change

        return df # Returns the new dataframe
    except:
        print('An error occurred while changing the column name')

        return

def changeColumnType (df, column, newType):
    try:
        print('Before Change:\n' + str(df.dtypes)) # Will show the column types before the change

        df[column] = df[column].astype(newType) # Changes the type of the specified column

        print('After Change:\n' + str(df.dtypes)) # Will show the column types after the change

        return df # Returns the new dataframe
    except:
        print('An error occurred while changing the column type')

        return
    
def selectDataType ():
    try:
        print('---------------------\nData Types:\n1. int64\n2. float64\n3. object\n4. bool\n5. datetime64\n6. timedelta[ns]\n7. category')
        dataType = int(input('Select the number of the data type you want to use: '))

        if dataType == 1:
            return 'int64'
        elif dataType == 2:
            return 'float64'
        elif dataType == 3:
            return 'object'
        elif dataType == 4:
            return 'bool'
        elif dataType == 5:
            return 'datetime64'
        elif dataType == 6:
            return 'timedelta[ns]'
        elif dataType == 7:
            return 'category'
        else:
            print('Invalid Input')
            return
    except:
        print('An error occurred while selecting the data type')

        return

def columnData (df, column):
    try:
        print('---------------------\n' + column + ' Data:\n' + str(df[column].value_counts())) # Will show the number of rows and columns in the data set

        return
    except:
        print('An error occurred while getting the column data')

        return
    
def histPlot (df, column):
    try:
        sns.histplot(data=df, x=column, kde=True) # Creates a histogram plot of the specified column

        plt.title('Histogram of ' + column)

        plt.show()

        return
    except:
        print('An error occurred while creating the histogram plot')

        return

def histPlotHue (df, column, hue):
    try:
        sns.histplot(data=df, x=column, hue=hue, kde=True) # Creates a histogram plot of the specified column with a hue

        plt.title('Histogram of ' + column + ' with a Hue')

        plt.show()

        return
    except:
        print('An error occurred while creating the histogram plot with a hue')

        return
    
# Function to merge two dataframes. Needs to be fixed.
def mergeDataframes ():
    from CSVME import getColumnNames, getUserColumn
    try:
        print('------------CSV Management------------')
        print('Select the first CSV file you want to use: ')
        DF1 = selectCSVFile()
        print('Select the second CSV file you want to use: ')
        DF2 = selectCSVFile()
        print('Before Merge:\n' + str(DF1.shape)) # Will show the number of rows and columns in the data set before the merge

        column = getUserColumn(DF1, 'Select the column you want to merge on: ')

        DF1 = pd.merge(DF1, DF2, on=column, how='left') # Merges the two dataframes on the specified column

        print('After Merge:\n' + str(DF1.shape)) # Will show the number of rows and columns in the data set after the merge

        return DF1 # Returns the new dataframe  
    except:
        print('An error occurred while merging the dataframes.\nRemember to have both column names the same.')

        return

# N/N Functions__________________________________________________________________________________________________________
# Numeric to Numeric Bi-Relationship
def NNBiRelationship (df, iVar, dVar):
    try:

        # This is for Pearson's Correlation Coefficient

        print('---------------------\n' + iVar + ' vs ' + dVar + ':\n')

        # R and P-Value
        r, p = stats.pearsonr(df[iVar], df[dVar])
        print('r: ' + str(round(r, 3)))
        print('p-value:' + str(round(p, 3)))

        # R Squared Value
        r, p = stats.pearsonr(df[iVar], df[dVar])
        r2 = (r ** 2).round(3)
        r2
        print("r-square:" + str(r2))

        # Linear Regression Equasion
        model = np.polyfit(df[iVar], df[dVar], 1)
        print('y = ' + str(round(model[0], 3)) + 'x +' + str(round(model[1], 3)))

        # Display Means
        print('Mean of ' + iVar + ': ' + str(round(df[iVar].mean(), 3)))
        print('Mean of ' + dVar + ': ' + str(round(df[dVar].mean(), 3)))
    except ValueError:
        print("An error occurred while calculating the bi-relationship. Make sure to use only numeric columns.")

# Scatter Plot, for N/N relationships
def ScatterPlot (df, iVar, dVar, title):
    try:
        # Scatter Plot with Linear Regression Line
        sns.lmplot(x = iVar, y = dVar, data = df)
        # title = 'Scatter Plot of ' + iVar + ' vs ' + dVar
        plt.title(title)
        plt.show()
    except ValueError:
        print("An error occurred while creating the scatter plot. Make sure to use only numeric columns.")

# Numeric to Numeric Correlation Matrix (all Variables compared to one)
def NNCorrMatrix (df, iVar):
    try:
        correlation_matrix = df.corr()
        sorted_correlation = correlation_matrix[iVar].sort_values(ascending=False)
        print(sorted_correlation)
    except ValueError:
        print("An error occurred while calculating the correlation matrix. Make sure to only have numeric columns.")

# Numeric to Numeric 4d Scatter Plot
def NN4d (df, x, y, z, color, title):
    try:
        # Create the 4D scatter plot
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title)

        # Show the plot
        fig.show()
    except ValueError:
        print("An error occurred while creating the 4D scatter plot. Make sure to use only numeric columns.")

# C/N Functions________________________________________________________________________________________
# Category to Numeric Bi-Relationship

# T Test, between two groups and one numberic variable shared between them. Shows means as well.
def TTest (df, CatColumn, Cat1, Cat2, NumColumn):
    try:
        Cat1Filter = df[df[CatColumn] == Cat1]
        Cat2Filter = df[df[CatColumn] == Cat2]

        t, p = stats.ttest_ind(Cat1Filter[NumColumn], Cat2Filter[NumColumn])

        print("---------------------\nT-Test Results:\n")
        print("t: " + str(round(t, 3))) # T value showing the difference between the two groups
        print("p: " + str(round(p, 3))) # P value showing the probability that the difference is due to chance

        print(f"{Cat1} Mean: " + str(round(Cat1Filter[NumColumn].mean(), 3)))
        print(f"{Cat2} Mean: " + str(round(Cat2Filter[NumColumn].mean(), 3)))
    except ValueError:
        print("An error occurred while calculating the t-test. Make sure to use correct data columns.")

# ANOVA Test, between three or more groups and one numberic variable shared between them
def ANOVATest (df, CatColumn, NumColumn):
    try:
        groups = df[CatColumn].unique()  #Filter to all the unique regions (northwest, southeast, northwest, southwest)

        group_labels = [] #Create an empty list that will be a two-dimensional list of lists to store the label values associated with each category

        for g in groups: # Loop through each unique region
            group_labels.append(df[df[CatColumn] == g][NumColumn]) # Add to the group_labels list the charges for that region

        f, p =  stats.f_oneway(*group_labels) # Perform a one way anova on all the regions. *group_labels is a shortcut way of listing out each of the regions.
        # F is the effect size and p is the p-value

        print('ANOVA Test Results:\nF: ' + str(round(f, 4)))
        print('p: ' + str(round(p, 4)))

        print(TukeyTest(df, CatColumn, NumColumn))

        return
    except ValueError:
        print("An error occurred while calculating the ANOVA test. Make sure to use correct data columns.")

# Tukey Test, between three or more groups and one numberic variable shared between them
def TukeyTest (df, CatColumn, NumColumn):
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey = pairwise_tukeyhsd(endog=df[NumColumn], groups=df[CatColumn], alpha=0.05)

        print(tukey)

        return
    except ValueError:
        print("An error occurred while calculating the Tukey test. Make sure to use correct data columns.")

# Bar Plot, for N/C relationships
def BarPlot (df, CatColumn, NumColumn, Title):
    try:
        sns.barplot(x=NumColumn, y=CatColumn, data=df)

        plt.title(Title)

        plt.show()

        return
    except ValueError:
        print("An error occurred while creating the bar plot. Make sure to use correct data columns.")

# Bar Plot, for N/C relationships with a hue
def BarPlotHue (df, CatColumn, NumColumn, HueColumn, Title):
    try:
        sns.barplot(x=NumColumn, y=CatColumn, hue=HueColumn, data=df)

        plt.title(Title)

        plt.show()

        return
    except ValueError:
        print("An error occurred while creating the bar plot with hue. Make sure to use correct data columns.")

# C/C Functions________________________________________________________________________________________
# Category to Category Bi-Relationship
def ChiSquare (df, Cat1, Cat2):
    try:
        crosstab = pd.crosstab(index = df[Cat1], columns = df[Cat2], margins = True)

        x, p, dof, expected_values = chi2_contingency(crosstab)

        print('---------------------\nChi-Square Results:\n')

        print('Chi-Square: ' + str(round(x, 3)))
        print('P-value: ' + str(round(p, 3)))
        print('Degrees of Freedom: ' + str(round(dof, 3)))

        print('Expected Values:\n' + str(expected_values))

        return
    
    except ValueError:
        print("An error occurred while calculating the chi-square test. Make sure to use correct data columns.")

# Category to Category Crosstab Expected
def CrosstabExpected (df, Cat1, Cat2, title):
    try:
        crosstab = pd.crosstab(index = df[Cat1], columns = df[Cat2], margins = True)

        x, p, dof, expected_values = chi2_contingency(crosstab)

        ex_df = pd.DataFrame(np.rint(expected_values).astype('int64'), columns=crosstab.columns, index = crosstab.index )

        sns.heatmap(ex_df, annot=True,  fmt='d', cmap='coolwarm')

        plt.title(title)

        plt.show()

        return
    except ValueError:
        print("An error occurred while creating the crosstab. Make sure to use correct data columns.")

        return

# Category to Category Crosstab
def CrosstabObserved (df, Cat1, Cat2, title):
    try:
        crosstab = pd.crosstab(index = df[Cat1], columns = df[Cat2], margins = True)

        plt.title(title)

        sns.heatmap(crosstab, annot=True, fmt='d', cmap='coolwarm')

        plt.show()

        return
    except ValueError:
        print("An error occurred while creating the crosstab. Make sure to use correct data columns.")

        return
    
def CrosstabObservedPercent (df, Cat1, Cat2, title):
    try:
        crosstab = pd.crosstab(index = df[Cat1], columns = df[Cat2], margins = True, normalize='all')

        plt.title(title)

        sns.heatmap(crosstab, annot=True, fmt='.2%', cmap='coolwarm', cbar=False)

        plt.show()

        return
    except ValueError:
        print("An error occurred while creating the crosstab. Make sure to use correct data columns.")

        return


# CSV Manipulation Functions________________________________________________________________________________________
def showCSVFiles ():
    import os
    try:
        print('------------CSV Management------------')
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for i, file in enumerate(files):
            if file.endswith('.csv'):
                print(f"{i+1}. {file}")

        return files
    except Exception as e:
        print(f"An error occurred while getting the CSV files: {e}")
        return None
    
def selectCSVFile ():
    # Selects the CSV file to use and returns the dataframe
    try:
        print('------------CSV Management------------')
        print('Select the CSV file you want to use:')
        showCSVFiles()
        file = int(input('Enter the number of the CSV file you want to use: '))
        files = showCSVFiles()
        df = pd.read_csv(files[file - 1])
        print("DataFrame linked successfully!")
        print('------------Data Frame Reset------------')
        return df
    except Exception as e:
        print(f"An error occurred while linking the data frame: {e}")
        return None
    
def exportCSV (df, name):
    try:
        df.to_csv(name + '.csv', index=False) # Exports the dataframe to a CSV file

        print('CSV File Exported')

        return
    except:
        print('An error occurred while exporting the CSV file')

        return
    
def importCSV (link):
    try:
        df = pd.read_csv(link)
        print("DataFrame linked successfully!")
        print('------------Data Frame Reset------------')
        return df
    except Exception as e:
        print(f"An error occurred while linking the data frame: {e}")
        print('Remember to use the full file path (Copy Path if local, with no quotes)')
        return None