
import openai
import csv
import datetime
import time
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code_ensemble

def get_prompt(task='classification'):
    if task == 'classification':
        additional_information = "Do not use EnsembleSelectionClassifier instead use EnsembleVoteClassifier, consider that EnsembleVoteClassifier accept only the next parameters: clfs, voting, weights, verbose, use_clones, fit_base_estimators, consider the restriction that 'voting' must always be 'soft' (voting='soft')."
    else:
        additional_information = ""

    return f"""
The dataframe split in ‘X_train’, ‘y_train’ and a list called ‘list_pipelines’ are loaded in memory.

This code was written by an expert data scientist working to create a suitable “Multi-Layer Stack Ensembling”, the task is {task}. It is a snippet of code that import the packages necessary to create a such Multi-layer stack ensembling model using a list of pipelines called ‘list_pipelines’, such list contain ‘sklearn’ pipeline objects. 

Code formatting the Multi-Layer Stack Ensembling:
```python
(Some packages imported and code necessary to create a Multi-Layer Stack Ensembling Model, which must be called ‘model’. {additional_information}
This model will be created by reusing all of its base layer model types “list_pipelines” as stackers. Those stacker models take as input not only the predictions of the models at the previous layer, but also the original data features themselves (input vectors are data features concatenated with lowerlayer model predictions).
The second and final stacking layer applies ensemble selection.
In addition, from 'model' call its respective 'fit' function to feed the model with 'X_train' and 'y_train')
```end

This codeblock ends with "```end" and starts with "```python"
Codeblock:

"""


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from(task='classification'):
    return get_prompt(task=task)

def generate_code_embedding(
        list_pipelines,
        X,
        y,
        model="gpt-3.5-turbo",
        display_method="markdown",
        task='classification',
        just_print_prompt=False,
        iterations_max=2,
        identifier='',
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print
    prompt = build_prompt_from(task=task)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        # A small sample if the dataset is too large
        value_to_consider_for_fast_training = 5000
        if task == "classification":
            if len(X) >= value_to_consider_for_fast_training:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    train_size=value_to_consider_for_fast_training,
                                                                    stratify=y, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, stratify=y, random_state=0)
        else:
            if len(X) >= value_to_consider_for_fast_training:
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    train_size=value_to_consider_for_fast_training,
                                                                    random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=0)

        try:
            pipe = run_llm_code_ensemble(
                code,
                X_train,
                y_train,
                list_pipelines
            )
            # Works for both, regression and classification, I guess
            performance = pipe.score(X_test, y_test)
        except Exception as e:
            pipe = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None

        return None, performance, pipe

    messages = [
        {
            "role": "system",
            "content": f"You are an expert datascientist assistant creating a Multi-Layer Stack Ensembling for a dataset X_train, y_train, you need to use the pipelines storaged in 'list_pipelines’ . You answer only by generating code. Answer as concisely as possible. The task is {task}",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset with specific description, task:*\n {task}")

    e = 1
    iteration_counts = 0
    pipe = None # If return None, means the code could not be executed and we need to generated the code manually
    while e is not None:
        iteration_counts += 1
        if iteration_counts > iterations_max:
            break
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            time.sleep(60) # Wait 1 minute before next request
            continue
        e, performance, pipe = execute_and_evaluate_code_block(code)
        if isinstance(performance, float):
            print('The performance of the LLM ensemble is:', performance)
            valid_model = True
            pipeline_sentence = f"The code was executed and generated a ´model´ with score {performance}"
        else:
            valid_model = False
            pipeline_sentence = "The last code did not generate a valid ´model´, it was discarded."

        display_method(
            "\n"
            + f"*Error? : {str(e)}*\n"
            + f"*Valid model: {str(valid_model)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )
        if e is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, code, str(performance)])
        if e is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, code, e])

            general_considerations = ''
            if 'required positional argument' in str(e):
                general_considerations = "Consider passing the list_pipelines to the ensemble"
            if task == 'classification':
                if 'required positional argument' in str(e):
                    general_considerations = "Consider that EnsembleVoteClassifier accept only the next parameters: (clfs, voting, weights, verbose, use_clones, fit_base_estimators), consider the restriction that 'voting' must always be 'soft' (voting='soft')."
            else:
                if 'required positional argument' in str(e):
                    general_considerations = "Consider that StackingRegressor from sklearn accept only the next parameters: (estimators, final_estimator, cv, n_jobs, passthrough, verbose)"

            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed, error type {type(e)}, error: {str(e)}.\n 
                    Code: ```python{code}```\n. 
                    {general_considerations} \n
                    Do it again and fix error, breathe and think deeply.\n
                                ```python
                                """,
                },
            ]
            continue

    return pipe

# https://developer.ibm.com/articles/stack-machine-learning-models-get-better-results/
# https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28

#### this is working mlxtend v1 but I don't see the improvement
def generate_ensemble_manually(
        X, y,
        get_pipelines,
        task='classification',
):
    print('Doing stacking manually')
    if task == "classification":
        import pandas as pd
        from sklearn.svm import SVC
        from mlxtend.classifier import StackingCVClassifier

        # Define the meta regressor
        svc_rbf = SVC(kernel='rbf', probability=True)

        # Base models without preprocessing steps
        estimators = [list(pipe.named_steps.values())[-1] for pipe in get_pipelines]

        # Create the stacked model
        stacker = StackingCVClassifier(classifiers=estimators,
                                       meta_classifier=svc_rbf,
                                       cv=5,
                                       use_features_in_secondary=True,
                                       store_train_meta_features=True,
                                       )

        preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
        numeric_X = preprocessing_steps[0].fit_transform(X)
        numeric_X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        stacker.fit(numeric_X, y)
    else:
        import pandas as pd
        from sklearn.svm import SVR
        from mlxtend.regressor import StackingCVRegressor

        # Define the meta regressor
        svr_rbf = SVR(kernel='rbf')

        # Base models without preprocessing steps
        estimators = [list(pipe.named_steps.values())[-1] for pipe in get_pipelines]

        # Create the stacked model
        stacker = StackingCVRegressor(regressors=estimators,
                                      meta_regressor=svr_rbf,
                                      cv=5,
                                      use_features_in_secondary=True,
                                      store_train_meta_features=True,
                                      )

        preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
        numeric_X = preprocessing_steps[0].fit_transform(X)
        numeric_X = pd.DataFrame(numeric_X, columns=[f"{i}" for i in range(numeric_X.shape[1])])
        stacker.fit(numeric_X, y)
    return stacker


# # #### mlxtend v2
# def generate_ensemble_manually(
#         X, y,
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     if task == "classification":
#         import numpy as np
#         from sklearn.svm import SVC
#         from sklearn.model_selection import train_test_split
#         from mlxtend.classifier import StackingCVClassifier
#
#         predictions_base_models = []
#         new_base_models = []
#         for base_model in get_pipelines:
#             base_model.fit(X, y)
#             new_base_models.append(base_model)
#             this_predict = base_model.predict(X)
#             predictions_base_models.append(this_predict)
#
#         # Train the meta-model
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#         transformed_X = preprocessing_steps[0].fit_transform(X)
#         # Combine the predictions of the base models into a single feature matrix
#         X_test_meta = np.hstack((transformed_X, np.column_stack(tuple(predictions_base_models))))
#
#         # Define the meta regressor
#         svc_rbf = SVC(kernel='rbf', probability=True)
#
#         # Base models without preprocessing steps
#         estimators = [list(pipe.named_steps.values())[-1] for pipe in get_pipelines]
#
#         # Create the stacked model
#         stacker = StackingCVClassifier(classifiers=estimators,
#                                        meta_classifier=svc_rbf,
#                                        cv=5,
#                                        use_features_in_secondary=True,
#                                        store_train_meta_features=True,
#                                        )
#
#         stacker.fit(X_test_meta, y)
#
#     else:
#         import numpy as np
#         from sklearn.svm import SVR
#         from sklearn.model_selection import train_test_split
#         from mlxtend.regressor import StackingCVRegressor
#
#         predictions_base_models = []
#         new_base_models = []
#         for base_model in get_pipelines:
#             base_model.fit(X, y)
#             new_base_models.append(base_model)
#             this_predict = base_model.predict(X)
#             predictions_base_models.append(this_predict)
#
#         # Train the meta-model
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#         transformed_X = preprocessing_steps[0].fit_transform(X)
#         # Combine the predictions of the base models into a single feature matrix
#         X_test_meta = np.hstack((transformed_X, np.column_stack(tuple(predictions_base_models))))
#
#         # Define the meta regressor
#         svr_rbf = SVR(kernel='rbf')
#
#         # Base models without preprocessing steps
#         estimators = [list(pipe.named_steps.values())[-1] for pipe in get_pipelines]
#
#         # Create the stacked model
#         stacker = StackingCVRegressor(regressors=estimators,
#                                       meta_regressor=svr_rbf,
#                                       cv=5,
#                                       use_features_in_secondary=True,
#                                       store_train_meta_features=True,
#                                       )
#         stacker.fit(X_test_meta, y)
#
#     return stacker, new_base_models

###### IT IS WORKING BUT WITHOUT IMPROVEMENT
# def generate_ensemble_manually(
#         X, y,
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     if task == "classification":
#         import numpy as np
#         from sklearn.svm import SVC
#         from sklearn.model_selection import train_test_split
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
#         predictions_base_models = []
#         new_base_modesl = []
#         for base_model in get_pipelines:
#             base_model.fit(X_train, y_train)
#             new_base_modesl.append(base_model)
#             this_predict = base_model.predict(X_test)
#             predictions_base_models.append(this_predict)
#
#         # Train the meta-model
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#         transformed_X = preprocessing_steps[0].fit_transform(X_test)
#         # Combine the predictions of the base models into a single feature matrix
#         X_test_meta = np.hstack((transformed_X, np.column_stack(tuple(predictions_base_models))))
#
#         # Train the meta-model on the combined feature matrix and the target values
#         meta_model = SVC(kernel='rbf', probability=True)
#         meta_model.fit(X_test_meta, y_test)
#     else:
#         import numpy as np
#         from sklearn.svm import SVR
#         from sklearn.model_selection import train_test_split
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
#         predictions_base_models = []
#         new_base_modesl = []
#         for base_model in get_pipelines:
#             base_model.fit(X_train, y_train)
#             new_base_modesl.append(base_model)
#             this_predict = base_model.predict(X_test)
#             predictions_base_models.append(this_predict)
#
#         # Train the meta-model
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#         transformed_X = preprocessing_steps[0].fit_transform(X_test)
#         # Combine the predictions of the base models into a single feature matrix
#         X_test_meta = np.hstack((transformed_X, np.column_stack(tuple(predictions_base_models))))
#
#         # Define the meta regressor
#         meta_model = SVR(kernel='rbf')
#         meta_model.fit(X_test_meta, y_test)
#
#     return meta_model, new_base_modesl

# def generate_ensemble_manually(
#         X, y,
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     if task == "classification":
#         from mlxtend.classifier import StackingClassifier
#         from sklearn.pipeline import make_pipeline
#         from sklearn.linear_model import LogisticRegression
#
#         # Create a list of base models
#         base_models = [make_pipeline(model) for model in get_pipelines]
#
#         # Create the meta-model
#         meta_model = LogisticRegression()
#
#         # Create the stacked ensemble
#         stacked_ensemble = StackingClassifier(
#             classifiers=base_models,
#             meta_classifier=meta_model,
#             use_probas=True,
#             average_probas=False
#         )
#
#         # Train the stacked ensemble on the training data
#         stacked_ensemble.fit(X, y)
#
#     else:
#         from mlxtend.regressor import StackingRegressor
#         from sklearn.pipeline import make_pipeline
#         from sklearn.linear_model import LinearRegression
#         from sklearn.pipeline import make_pipeline
#         from sklearn.compose import ColumnTransformer
#         from sklearn.impute import SimpleImputer
#         from sklearn.preprocessing import StandardScaler, OrdinalEncoder
#
#         # Identify categorical and numeric column names
#         categorical_cols = X.select_dtypes(include=['object']).columns
#         numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
#
#         # Create the column transformer
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', SimpleImputer(strategy='mean'), numeric_cols),
#                 ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols),
#                 ('cat_encoder', OrdinalEncoder(), categorical_cols)
#             ])
#
#         # Create a list of base models
#         base_models = [make_pipeline(preprocessor, model) for model in get_pipelines]
#
#         # Create the meta-model
#         meta_model = LinearRegression()
#
#         # Create the stacked ensemble
#         stacked_ensemble = StackingRegressor(
#             regressors=base_models,
#             meta_regressor=meta_model,
#             use_features_in_secondary=True
#         )
#
#         # Train the stacked ensemble on the training data
#         stacked_ensemble.fit(X, y)
#
#     return stacked_ensemble

# def generate_ensemble_manually(
#         X, y,
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     import numpy as np
#     from sklearn.ensemble import StackingClassifier, StackingRegressor
#     from sklearn.svm import SVC, SVR
#     from sklearn.pipeline import make_pipeline
#     if task == "classification":
#         # Create the first layer of stackers
#         stackers = []
#         for pipeline in get_pipelines:
#             stacker = pipeline
#             stackers.append(stacker)
#
#         # Create the second layer of stackers
#         def predict_proba_with_features(model, X):
#             # Get the predicted probabilities from the base model
#             proba = model.predict_proba(X)
#
#             # Add the original features to the predicted probabilities
#             features = X.values
#             return np.hstack([proba, features])
#
#         # Create the stacked model with preprocessing steps
#         stacker = StackingClassifier(estimators=[('stacker_' + str(i), stacker) for i, stacker in enumerate(stackers)],
#                                      final_estimator=SVC(kernel='rbf', probability=True),
#                                      cv=5,
#                                      passthrough=True,
#                                      )
#
#         # Create a new pipeline that includes the stacked model
#         pipe = make_pipeline(stacker)
#
#         # Train the stacked model on the training data
#         pipe.fit(X, y)
#
#         # Add the original features to the predicted probabilities
#         meta_features = np.apply_along_axis(predict_proba_with_features, 1,
#                                             [est[1] for est in stacker.named_estimators_], X)
#
#         # Train the meta-model on the predicted probabilities and original features
#         pipe.named_steps['stackingclassifier'].final_estimator_.fit(meta_features, y)
#
#
#     else:
#         # Create the first layer of stackers
#         stackers = []
#         for pipeline in get_pipelines:
#             stacker = pipeline
#             stackers.append(stacker)
#
#         # Create the second layer of stackers
#         def predict_with_features_regr(model, X):
#             # Get the predicted values from the base model
#             y_pred = model.predict(X)
#
#             # Add the original features to the predicted values
#             features = X.values
#             return np.hstack([y_pred.reshape(-1, 1), features])
#
#         # Create the stacked model with preprocessing steps
#         stacker = StackingRegressor(estimators=[('stacker_' + str(i), stacker) for i, stacker in enumerate(stackers)],
#                                     final_estimator=SVR(kernel='rbf'),
#                                     cv=5,
#                                     passthrough=True,
#                                     )
#
#         # Create a new pipeline that includes the stacked model
#         pipe = make_pipeline(stacker)
#
#         # Train the stacked model on the training data
#         pipe.fit(X, y)
#
#         # Add the original features to the predicted values
#         meta_features = np.apply_along_axis(predict_with_features_regr, 1,
#                                             [est[1] for est in stacker.named_estimators_], X)
#
#         # Train the meta-model on the predicted values and original features
#         pipe.named_steps['stackingregressor'].final_estimator_.fit(meta_features, y)
#
#     return pipe

# def generate_ensemble_manually(
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     from sklearn.pipeline import make_pipeline
#     if task == "classification":
#         from sklearn.svm import SVC
#         from mlxtend.classifier import StackingCVClassifier
#
#         # Extract the preprocessing steps from the first pipeline
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#
#         # Create the first layer of stackers
#         svc_rbf = SVC(kernel='rbf', probability=True)
#
#         # Create the stacked model with preprocessing steps
#         stacker = StackingCVClassifier(classifiers=get_pipelines,
#                                        meta_classifier=svc_rbf,
#                                        cv=5,
#                                        use_features_in_secondary=True,
#                                        store_train_meta_features=True,
#                                        )
#
#         # Create a new pipeline that includes the preprocessing steps and the stacked model
#         pipe = make_pipeline(*preprocessing_steps, stacker)
#
#
#     else:
#         from sklearn.ensemble import VotingRegressor
#         from sklearn.svm import SVR
#         from mlxtend.regressor import StackingCVRegressor
#
#         # Extract the preprocessing steps from the first pipeline
#         preprocessing_steps = list(get_pipelines[0].named_steps.values())[:-1]
#
#         # Define the meta regressor
#         svr_rbf = SVR(kernel='rbf')
#
#         # Create the stacked model with preprocessing steps
#         stacker = StackingCVRegressor(regressors=get_pipelines,
#                                       meta_regressor=svr_rbf,
#                                       cv=5,
#                                       use_features_in_secondary=True,
#                                       store_train_meta_features=True,
#                                       )
#
#         # Create a new pipeline that includes the preprocessing steps and the stacked model
#         pipe = make_pipeline(*preprocessing_steps, stacker)
#
#     return pipe

# def generate_ensemble_manually(
#         X, y,
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     if task == "classification":
#         from sklearn.ensemble import VotingClassifier
#         from sklearn.svm import SVC
#         from mlxtend.classifier import StackingCVClassifier
#
#         # Create the first layer of stackers
#         svc_rbf = SVC(kernel='rbf', probability=True)
#         stackers = []
#         for pipeline in get_pipelines:
#             stacker = StackingCVClassifier(classifiers=[pipeline],
#                                            meta_classifier=svc_rbf,
#                                            cv=5,
#                                            use_features_in_secondary=True,
#                                            store_train_meta_features=True,
#                                            )
#             stackers.append(stacker)
#
#         # Create the second layer of stackers
#         estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers)]
#         pipe = VotingClassifier(estimators=estimators, voting='soft')
#         pipe.fit(X, y)
#
#     else:
#         from sklearn.ensemble import VotingRegressor
#         from sklearn.svm import SVR
#         from mlxtend.regressor import StackingCVRegressor
#
#         # Define the meta regressor
#         svr_rbf = SVR(kernel='rbf')
#
#         # Create the first layer of stackers
#         stackers = []
#         for pipeline in get_pipelines:
#             stacker = StackingCVRegressor(regressors=[pipeline],
#                                           meta_regressor=svr_rbf,
#                                           cv=5,
#                                           use_features_in_secondary=True,
#                                           store_train_meta_features=True,
#                                           )
#             stackers.append(stacker)
#
#         # Create the second layer of stackers
#         estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers)]
#         pipe = VotingRegressor(estimators=estimators)
#         pipe.fit(X, y)
#     return pipe


# def generate_ensemble_manually(
#         get_pipelines,
#         task='classification',
# ):
#     print('Doing stacking manually')
#     if task == "classification":
#         from sklearn.ensemble import VotingClassifier
#         from sklearn.svm import SVC
#         from mlxtend.classifier import StackingCVClassifier
#
#         # Create the first layer of stackers
#         svc_rbf = SVC(kernel='rbf', probability=True)
#         stackers1 = []
#         for pipeline in get_pipelines:
#             stacker = StackingCVClassifier(classifiers=[pipeline],
#                                            meta_classifier=svc_rbf)
#             stackers1.append(stacker)
#
#         # Create the second layer of stackers
#         stackers2 = []
#         for stacker1 in stackers1:
#             stacker = StackingCVClassifier(classifiers=[stacker1],
#                                            meta_classifier=svc_rbf)
#             stackers2.append(stacker)
#
#         # Create the third layer of stackers
#         estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers2)]
#         pipe = VotingClassifier(estimators=estimators, voting='soft')
#
#     else:
#         from sklearn.ensemble import VotingRegressor
#         from sklearn.svm import SVR
#         from mlxtend.regressor import StackingCVRegressor
#
#         # Define the meta regressor
#         svr_rbf = SVR(kernel='rbf')
#
#         # Create the first layer of stackers
#         stackers1 = []
#         for pipeline in get_pipelines:
#             stacker = StackingCVRegressor(regressors=[pipeline],
#                                           meta_regressor=svr_rbf)
#             stackers1.append(stacker)
#
#         # Create the second layer of stackers
#         stackers2 = []
#         for stacker1 in stackers1:
#             stacker = StackingCVRegressor(regressors=[stacker1],
#                                            meta_regressor=svr_rbf)
#             stackers2.append(stacker)
#
#         # Create the third layer of stackers
#         estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers2)]
#         pipe = VotingRegressor(estimators=estimators)
#
#     return pipe

# Original:


# print('Ensemble with LLM failed, doing it manually')
# if task == "classification":
#     from sklearn.ensemble import VotingClassifier
#     from sklearn.svm import SVC
#     from mlxtend.classifier import StackingClassifier
#
#     # Create the first layer of stackers
#     svc_rbf = SVC(kernel='rbf', probability=True)
#     stackers = []
#     for pipeline in get_pipelines:
#         stacker = StackingClassifier(classifiers=[pipeline],
#                                      meta_classifier=svc_rbf)
#         stackers.append(stacker)
#
#     # Create the second layer of stackers
#     estimators = [('stacker' + str(i), stacker) for i, stacker in enumerate(stackers)]
#     pipe = VotingClassifier(estimators=estimators, voting='soft')
#
# else:
#     from sklearn.ensemble import VotingRegressor
#     # Create the ensemble
#     estimators = [('pipeline' + str(i), pipeline) for i, pipeline in enumerate(get_pipelines)]
#     pipe = VotingRegressor(estimators=estimators)
