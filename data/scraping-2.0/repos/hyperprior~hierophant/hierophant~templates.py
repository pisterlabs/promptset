
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

base_prompt = SystemMessagePromptTemplate.from_template(
    "You are creating the explanation for a machine learning model whose architecture is {model}. Please explain why the model made the predictions it did knowing that your target audience is {target_audience}. Depending on your target audience, you should include numbers to support your findings. All numbers should be converted to scientific notation with 3 significant digits"
)



shap_score_prompt = HumanMessagePromptTemplate.from_template("""## SHAP
        
SHAP (SHapley Additive exPlanations) values offer a measure of the contribution of each feature towards the prediction for a specific instance in contrast to a baseline value. They are based on Shapley values, a concept from cooperative game theory that assigns a payout (in this case, the prediction for an instance) to each player (in this case, each feature) based on their contribution to the total payout.

In more concrete terms, for a given instance, a SHAP value for a feature is calculated as the average difference in the model's output when that feature is included versus when it is excluded, considering all possible subsets of features. Positive SHAP values indicate that the presence of a feature increases the model's output, while negative SHAP values suggest that the presence of the feature decreases the model's output.

### Results
{shap_values}
""")



feature_importance_prompt = HumanMessagePromptTemplate.from_template("""## Feature Importance
            
Normalized feature importance is a way to measure the relative importance of each feature by taking into account the absolute contribution of each feature across all instances and classes. In the context of SHAP values, we first calculate the feature importance by finding the average absolute SHAP value for each feature across all instances and classes. We then normalize these importance values by dividing each one by the sum of all feature importances, ensuring that the total sums to 1. This gives us a measure of each feature's contribution relative to the total contribution of all features. This method assumes that the importance of a feature is proportional to the absolute magnitude of its SHAP values, irrespective of the direction (positive or negative) of its influence on the prediction.
            
### Results

{feature_importances}
""")


class_importance_prompt = HumanMessagePromptTemplate.from_template("""## Class Importances

Class importance gives an indication of which classes are most often influenced by the features in a multi-class prediction problem. It is especially useful when you want to understand how each class is affected by the different features.

To calculate class importance, we use the SHAP values which measure the contribution of each feature to the prediction of each class for each instance. Specifically, we compute the average absolute SHAP value for each class across all instances and features. This is done by taking the absolute SHAP values (to consider the magnitude of influence regardless of direction), summing them across all instances and features for each class, and then dividing by the total number of instances and features. The result is a measure of the average influence of features on each class. The higher this value, the more a class is influenced by the features on average.

### Results

{class_importances}
""")


instance_importance_prompt = HumanMessagePromptTemplate.from_template("""## Instance Importances

Instance importance is a measure of how much each individual instance (or data point) is influenced by the features in your model. It is calculated by taking the sum of the absolute SHAP values for each instance across all features and classes.

This gives you an idea of how strongly the model's prediction is influenced by the features for each individual instance. Instances with higher importance values have predictions that are more strongly influenced by their features. This can be particularly useful if you want to understand which instances are driving your model's performance, or if you want to investigate instances where the model is particularly confident or uncertain.

### Results

{instance_importances}
""")


feature_class_interaction_prompt = """## Feature-Class Interaction

The feature-class interactions can be calculated by averaging the absolute SHAP values for each feature-class pair over all instances. This gives a measure of how much each feature contributes to the prediction of each class, on average. This can be useful for understanding how different features influence different classes, which can be particularly important in multi-class problems.

{feature_class_interactions}
"""



lime_instances = HumanMessagePromptTemplate.from_template("""## LIME Instance Explainer
LIME, which stands for Local Interpretable Model-agnostic Explanations, is a method for explaining the predictions of any machine learning model. It was introduced in a paper by Ribeiro, Singh, and Guestrin in 2016.

Here's how LIME works:

Local surrogate model: LIME creates a local surrogate model around the specific instance you want to interpret. This model is simpler than the original model (often a linear model), and thus easier to interpret.

Perturbation: LIME perturbs the instance, creating a lot of slightly modified versions of it. It then uses the original model to predict the outcomes of these modified instances.

Weighting: LIME weights these new instances according to their proximity to the original instance. Those that are closer to the original instance get more weight.

Fit surrogate model: LIME fits the local surrogate model to the outcomes of the perturbed instances, taking into account the weights. This model is then used to interpret the prediction for the original instance.

By doing this, LIME can explain complex models locally (i.e., for specific instances) using simpler, interpretable models. The explanations provided by LIME are in the form of feature contributions, which tell you how much each feature contributed to the prediction for a specific instance. These contributions are represented as weights or coefficients in the local surrogate model.

It's important to note that LIME is model-agnostic, meaning it can be used with any type of machine learning model. It's also flexible and can be used for different types of data, including tabular data, text, and images.

### Output Format

The output from LIME is a list of tuples, where each tuple contains a feature and its corresponding weight in the explanation.

Each tuple corresponds to a feature and its impact on the prediction. The first element of the tuple is a statement about the feature's value, and the second element is the weight of that feature in the model's prediction.

{lime_instances}
""")



query = HumanMessagePromptTemplate.from_template(
    "Based on this analysis please answer this question: {query}"
)