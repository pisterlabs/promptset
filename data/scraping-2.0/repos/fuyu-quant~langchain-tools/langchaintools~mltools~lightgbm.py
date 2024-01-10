import pickle
import pandas as pd
import numpy as np

import lightgbm as lgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

from langchain.tools import BaseTool


class LgbmTrainTool(BaseTool):
    name = "lgbm_train_tool"
    description = """It is useful for learning LightGBM. The input is the column name of the target variable."""

    def _run(self, query: str) -> str:
      """Use the tool."""
      
      df = pd.read_csv('/home/langchain-tools/data/train.csv')
      x = df.drop([query], axis = 1)
      y = df[query]

      x_train,x_valid,y_train,y_valid = train_test_split(x, y ,test_size = 0.2, random_state=3655)

      # categorical features
      categorical_features = []
      for i in df.columns:
          if df[i].dtypes == 'category':
              categorical_features.append(i)


      lgb_train = lgbm.Dataset(x_train,y_train,categorical_feature=categorical_features,free_raw_data=False)
      lgb_eval = lgbm.Dataset(x_valid,y_valid,reference=lgb_train,categorical_feature=categorical_features,free_raw_data=False)


      # number of classes of the objective variable
      num_class = len(df[query].unique())
      if num_class == 2:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'binary', 'metric': 'auc', 'verbose': -1}
      elif num_class <= 50:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'multiclass', 'metric': 'multi_logloss','num_class': num_class, 'verbose': -1}
      else:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'regression','metric': 'rmse', 'verbose': -1}


      lgbm_model = lgbm.train(
          params,
          lgb_train,
          valid_sets=[lgb_train,lgb_eval],
          verbose_eval=10,
          #num_boost_round=1000,
          )
      

      file = '/home/langchain-tools/data/trained_model.pkl'
      pickle.dump(lgbm_model, open(file, 'wb'))

      split_df = pd.DataFrame(lgbm_model.feature_importance(importance_type = 'split'),index = x.columns, columns=['split_importance']).sort_values('split_importance', ascending=False)
      gain_df = pd.DataFrame(lgbm_model.feature_importance(importance_type = 'gain'),index = x.columns,columns=['gain_importance'])
      importance = split_df.join(gain_df)
      importance.to_csv('/home/langchain-tools/data/importance.csv')

      result = "LightGBM learning is complete."
      return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class LgbmInferenceTool(BaseTool):
    name = "lgbm_inference_tool"
    description = """It is useful for inference using LightGBM. The input is the column name of the objective variable."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        df = pd.read_csv('/home/langchain-tools/data/test.csv')


        file = '/home/langchain-tools/data/trained_model.pkl'
        lgbm_model = pickle.load(open(file, 'rb'))

        df = pd.read_csv('/home/langchain-tools/data/test.csv')
        X = df.drop([query], axis = 1)
        y = df[query]

        num_class = len(df[query].unique())
        # binary classification
        if num_class == 2:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            roc_auc = roc_auc_score(y, y_pred)
            y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
            accuracy = accuracy_score(y, y_pred_binary)
            df_report = pd.DataFrame([roc_auc, accuracy], index=['roc_auc', 'accuracy'], columns=['score'])
            df_report.to_csv('/home/langchain-tools/data/report.csv')

        
        # multiclass classification
        elif num_class <= 50:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            y_pred_class = [np.argmax(pred) for pred in y_pred]
            accuracy = accuracy_score(y, y_pred_class)
            #conf_mat = confusion_matrix(y, y_pred_class)
            df_report = pd.DataFrame([accuracy], index=['accuracy'], columns=['score'])
            df_report.to_csv('/home/langchain-tools/data/report.csv')


        
        # regression
        else:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            df_report = pd.DataFrame([mse, r2], index=['mse', 'r2'], columns=['score'])
            df_report.to_csv('/home/langchain-tools/data/report.csv')




        result = "LightGBM inference is complete." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
