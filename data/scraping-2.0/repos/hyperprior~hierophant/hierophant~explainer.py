import os
from types import SimpleNamespace

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import torch
import numpy as np

from functools import cached_property
from sklearn.inspection import permutation_importance
import shap
import os
from lime import lime_tabular

from hierophant import templates


class Explainer:
    def __init__(
        self,
        model,
        features,
        feature_names,
        output=None,
        class_names=None,
        target_audience="data scientist trying to debug the model and better understand it",
        predictions=None,
        llm=ChatOpenAI(openai_api_key=os.getenv(
            "OPENAI_API_KEY"), temperature=0),
    ):
        self.model = model
        self.features = features
        self.feature_names = feature_names
        self.num_features = len(self.feature_names)
        self.output = output
        self.class_names = class_names
        self.predictions = predictions
        self.target_audience = target_audience
        self.llm = llm
        self.prompts = [templates.base_prompt]
        self.shap_values = None
        self.lime_instances = None

    def shap(self, sample=None, check_additivity=False):
        # TODO keep check_additivity=False?
        if not sample:
            sample = self.features
        explainer = shap.Explainer(self.model.predict, sample)
        self.shap_values = explainer(self.features).values

    def add_shap(self, sample=None):
        self.prompts.append(templates.shap_score_prompt)

    @cached_property
    def feature_importances(self):
        if not self.shap_values:
            return None
        importances = self.shap_values
        feature_importances = np.mean(np.abs(self.shap_values), axis=(0, 2))
        feature_importances = feature_importances / np.sum(feature_importances)

        if self.feature_names:
            feature_importances = dict(
                zip(self.feature_names, feature_importances))

        return feature_importances

    def add_feature_importances(self):
        self.prompts.append(templates.feature_importance_prompt)

    @cached_property
    def class_importances(self):
        if not self.shap_values:
            return None
        class_importances = np.mean(np.abs(self.shap_values), axis=(0, 1))

        class_importances = class_importances / np.sum(class_importances)

        if self.class_names:
            class_importances = dict(zip(self.class_names, class_importances))

        return class_importances

    def add_class_importances(self):
        self.prompts.append(templates.class_importance_prompt)

    @cached_property
    def instance_importances(self):
        if not self.shap_values:
            return None
        instance_importances = np.sum(np.abs(self.shap_values), axis=(1, 2))
        return instance_importances / np.sum(instance_importances)

    def add_instance_importances(self):
        self.prompts.append(templates.instance_importance_prompt)

    @cached_property
    def feature_class_interactions(self, normalize=True):
        if not self.shap_values:
            return None
        feature_class_importances = np.mean(np.abs(self.shap_values), axis=0)
        if normalize:
            feature_class_importances = feature_class_importances / \
                np.sum(feature_class_importances, axis=0)
        feature_class_dict = {}
        for i, feature in enumerate(self.feature_names):
            feature_class_dict[feature] = {}
            for j, class_name in enumerate(self.class_names):
                feature_class_dict[feature][class_name] = feature_class_importances[i, j]

        return feature_class_dict

    def add_feature_class_interactions(self):
        self.prompts.append(templates.feature_class_interaction_prompt)

    @cached_property
    def permutation_importance(self):
        return permutation_importance(self.model, self.features, self.output)

    def add_permutation_importance(self):
        perm_dict = dict(
            zip(
                self.feature_names,
                list(
                    zip(
                        self.permutation_importance.importances_mean,
                        self.permutation_importance.importances_std,
                    )
                ),
            )
        )
        prompt = "## Permutation Importance Results:\n" + " ".join(
            [
                f"Feature `{k}` has a mean permutation importance of {v[0]} and a standard deviation of {v[1]}."
                for k, v in perm_dict.items()
            ]
        )

        # self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))

    @property
    def lime_explainer(self):
        return lime_tabular.LimeTabularExplainer(self.features,
                                                 feature_names=self.feature_names,
                                                 class_names=self.class_names,
                                                 discretize_continuous=True)

    def lime_explanations(self, X):
        all_explanations = []

        for instance in X:
            exp = self.lime_explainer.explain_instance(
                np.array(instance), self.model.predict_proba, num_features=self.num_features)
            all_explanations.append(exp.as_list())
        return all_explanations

    def add_lime(self, X):
        self.lime_instances = self.lime_explanations(X)
        self.prompts.append(templates.lime_instances)

    def explain(self, query: str = "Please give a detailed summary of your findings."):
        if isinstance(self.model, torch.nn.Module):
            X_var = Variable(torch.FloatTensor(self.features))
            y_pred = self.model(X_var)
            y_pred = F.softmax(self.predictions, dim=1).data.numpy()
        else:
            y_pred = self.model.predict_proba(self.features)

        return self._generate_explanations(
            self.features, self.output, self.predictions, query
        )

    def _generate_explanations(self, X, y, y_pred, query):
        self.prompts.append(
            templates.query

        )
        chat_prompt = ChatPromptTemplate.from_messages(self.prompts)

        chat_prompt.format_messages(
            model=self.model,
            target_audience=self.target_audience,
            shap_values=self.shap_values,
            feature_importances=self.feature_importances,
            class_importances=self.class_importances,
            instance_importances=self.instance_importances,
            feature_class_interactions=self.feature_class_interactions,
            lime_instances=self.lime_instances,
            query=query
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=chat_prompt,
        )

        print(
            chain.run(
                {
                    "target_audience": self.target_audience,
                    "shap_values": self.shap_values,
                    "query": query,
                    "model": "model",
                    'class_importances': self.class_importances,
                    'instance_importances': self.instance_importances,
                    'lime_instances': self.lime_instances,
                    'feature_class_interactions': self.feature_class_interactions,
                    'feature_importances': self.feature_importances,
                }
            )
        )

    def shap_waterfall(self, shap_value, max_display=14):
        return shap.plots.waterfall(shap_value, max_display=max_display)

    @property
    def plots(self):
        return SimpleNamespace(
            shap_waterfall=self.shap_waterfall
        )
