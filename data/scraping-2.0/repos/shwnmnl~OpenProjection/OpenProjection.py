import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Markdown
import openai
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

### Load data into dataframe
df = pd.read_excel('https://jewlscholar.mtsu.edu/bitstreams/7a60d5db-47ed-430c-809b-3c0cefeaa9d7/download')
embeddings_df = pd.read_csv('https://raw.githubusercontent.com/shwnmnl/OpenProjection/main/embeddings.csv', index_col=0)

### Categorical targets
schiz = df[df['Psychiatric Diagnosis (If Any)'] == 'schizophrenia']
schiz_embeddings = embeddings_df.loc[schiz.index]
schiz_embeddings['diagnosis'] = 0

psychoneuro = df[(df['Psychiatric Diagnosis (If Any)'] == 'Psychoneurosis: Somatization') | (df['Psychiatric Diagnosis (If Any)'] == 'psychoneurosis')]
psychoneuro_embeddings = embeddings_df.loc[psychoneuro.index]
psychoneuro_embeddings['diagnosis'] = 1

### Combine into a single dataframe
combined_embeddings = pd.concat([schiz_embeddings, psychoneuro_embeddings])

### Shuffle data for good measure while keeping indexes and we're good to go!
combined_embeddings = combined_embeddings.sample(frac=1, random_state=42)

### Continous targets
targets = pd.read_csv('https://raw.githubusercontent.com/shwnmnl/OpenProjection/main/targets.csv', index_col=0)

### Get targets for combined_embeddings indexes
final_targets = targets.loc[combined_embeddings.index]

### Classification Pipeline
class ClassificationPipeline:
    def __init__(self, data, target_column, test_size=0.2, random_state=42):
        self.data = data
        self.target = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.classifier = None

    def split_data(self):
        X = self.data
        y = self.target
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_classifier(self, model):
        X_train, X_test, y_train, y_test = self.split_data()
        self.classifier = model
        self.classifier.fit(X_train, y_train)

    def evaluate_classifier(self):
        y_pred = self.classifier.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        classification_report = metrics.classification_report(y_test, y_pred)
        
        return accuracy, confusion_matrix, classification_report

pipeline = ClassificationPipeline(combined_embeddings.iloc[:,:-1], combined_embeddings.iloc[:,-1])
pipeline.train_classifier(LogisticRegression(random_state=42))
accuracy, confusion_matrix, classification_report = pipeline.evaluate_classifier()
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix)
print("Classification Report:")
print(classification_report)

### Regression Pipeline
class RegressionPipeline:
    def __init__(self, data, target_column, test_size=0.2, random_state=42):
        self.data = data
        self.target = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.regressor = None

    def split_data(self):
        X = self.data
        y = self.target
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_regressor(self, model):
        X_train, X_test, y_train, y_test = self.split_data()
        self.regressor = model
        self.regressor.fit(X_train, y_train)

    def evaluate_regressor(self):
        y_pred = self.regressor.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, r2
    
pipeline = RegressionPipeline(combined_embeddings.iloc[:,:-1], final_targets.Cognition)
pipeline.train_regressor(Ridge())
mse, r2 = pipeline.evaluate_regressor()
print("Mean Squared Error:", mse)
print("Coefficient of Determination:", r2)

### Run regression pipeline for each target
def run_regression_pipeline(data, targets):
    """
    Runs the regression pipeline for each target and stores results in a dataframe.
    
    Args:
        data (pd.DataFrame): The dataframe containing the embeddings.
        targets (pd.DataFrame): The dataframe containing the targets.
        
    Returns:
        pd.DataFrame: The dataframe containing the results.
    """
    results = pd.DataFrame()
    
    for target in targets.columns[:4]:
        pipeline = RegressionPipeline(data, targets[target])
        pipeline.train_regressor(Ridge())
        mse, r2 = pipeline.evaluate_regressor()
        results[target] = [mse, r2]
        
    results.index = ['MSE', 'R2']
    
    return results

results = run_regression_pipeline(combined_embeddings.iloc[:,:-1], final_targets)
results = results.T
results.columns = ['MSE', 'R2']
results.sort_values(by='R2', ascending=False)