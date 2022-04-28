from functools import reduce
from typing import List

import feyn
import numpy as np
import pandas as pd
from feyn import Model, metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline


def random_forest_benchmark(data, target, param_grid=None, n_folds=5, num_experiments=1, feat_selection=None,
                            n_jobs=-1, inner_cv=None, outer_cv=None):
    if param_grid:
        param_grid = param_grid
    else:
        param_grid = {'rf__bootstrap': [True, False],
                      'rf__ccp_alpha': [0.0],
                      'rf__class_weight': ['balanced'],
                      'rf__max_depth': [3, 4, 5],
                      'rf__max_features': ['auto', 'sqrt'],
                      'rf__max_leaf_nodes': [None],
                      'rf__max_samples': [None],
                      'rf__min_impurity_decrease': [0.0],
                      'rf__min_samples_leaf': [1, 2, 4],
                      'rf__min_samples_split': [5, 10, 15],
                      'rf__min_weight_fraction_leaf': [0.0],
                      'rf__n_estimators': [50, 75, 100],
                      'rf__random_state': [42]
                      }

    data = pd.get_dummies(data)

    list_scores = list()

    for i in range(num_experiments):
        if not inner_cv:
            inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        if not outer_cv:
            outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)

        if feat_selection == 'lasso':
            rfc = Pipeline([
                ('feature_selection', SelectFromModel(LogisticRegressionCV(penalty="l1", solver='liblinear'))),
                ('rf', RandomForestClassifier())
            ])
        elif feat_selection == 'mi':
            rfc = Pipeline([
                ('feature_selection', SelectKBest(mutual_info_classif, k=10)),
                ('rf', RandomForestClassifier())
            ])
        elif feat_selection == 'f_score':
            rfc = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=10)),
                ('rf', RandomForestClassifier())
            ])
        else:
            rfc = Pipeline([
                ('rf', RandomForestClassifier())
            ])

        clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=n_jobs)
        nested_scores = cross_val_score(clf, X=data.drop(columns=target),
                                        y=data[target], cv=outer_cv, scoring='roc_auc')

        nested_score_mean = nested_scores.mean()
        list_scores.append(nested_score_mean)

    if num_experiments == 1:
        ans = nested_scores
    else:
        ans = list_scores

    return ans


def gradient_boosting_benchmark(data, target, param_grid=None, n_folds=5, num_experiments=1, feat_selection=None,
                                n_jobs=-1, inner_cv=None, outer_cv=None):
    if param_grid:
        param_grid = param_grid
    else:
        param_grid = {'learning_rate': [0.05, 0.1, 0.2],
                      'n_estimators': [50, 100, 200],
                      'max_depth': [2, 3, 4],
                      'max_features': [None, 'sqrt', 'log2'],
                      'subsample': [0.7, 1]
                      }

    data = pd.get_dummies(data)

    list_scores = list()

    for i in range(num_experiments):
        if not inner_cv:
            inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        if not outer_cv:
            outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)

        if feat_selection == 'lasso':
            gbc = Pipeline([
                ('feature_selection', SelectFromModel(LogisticRegressionCV(penalty="l1",solver='liblinear'))),
                ('rf', GradientBoostingClassifier())
            ])
        elif feat_selection == 'mi':
            gbc = Pipeline([
                ('feature_selection', SelectKBest(mutual_info_classif, k=10)),
                ('rf', GradientBoostingClassifier())
            ])
        elif feat_selection == 'f_score':
            gbc = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=10)),
                ('rf', GradientBoostingClassifier())
            ])
        else:
            gbc = Pipeline([
                ('rf', GradientBoostingClassifier())
            ])

        clf = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=n_jobs)
        nested_scores = cross_val_score(clf, X=data.drop(columns=target),
                                        y=data[target], cv=outer_cv, scoring='roc_auc')

        nested_score_mean = nested_scores.mean()
        list_scores.append(nested_score_mean)

    if num_experiments == 1:
        ans = nested_scores
    else:
        ans = list_scores

    return ans


def lasso_benchmark(data, target, param_grid=None, n_folds=5, num_experiments=5, n_jobs=-1, inner_cv=None,
                    outer_cv=None):
    if param_grid:
        param_grid = param_grid
    else:
        param_grid = {'C': [1000, 300, 100, 30, 10, 3, 1, .3, .1, .03, .01, .003, .001, .0003, .0001]}

    data = pd.get_dummies(data)

    list_scores = list()

    for i in range(num_experiments):
        if not inner_cv:
            inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        if not outer_cv:
            outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)

        lr = LogisticRegression(penalty='l1', solver='liblinear')

        clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=n_jobs)
        nested_scores = cross_val_score(clf, X=data.drop(columns=target),
                                        y=data[target], cv=outer_cv, scoring='roc_auc')

        nested_score_mean = nested_scores.mean()
        list_scores.append(nested_score_mean)

    if num_experiments == 1:
        ans = nested_scores
    else:
        ans = list_scores

    return ans


def elasticnet_benchmark(data, target, param_grid=None, n_folds=5, num_experiments=5, n_jobs=-1, inner_cv=None,
                         outer_cv=None):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
    from sklearn.linear_model import LogisticRegression

    if param_grid:
        param_grid = param_grid
    else:
        param_grid = {'C': [1000, 300, 100, 30, 10, 3, 1, .3, .1, .03, .01, .003, .001, .0003, .0001],
                      'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

    data = pd.get_dummies(data)

    list_scores = list()

    for i in range(num_experiments):
        if not inner_cv:
            inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        if not outer_cv:
            outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)

        lr = LogisticRegression(penalty='elasticnet', solver='saga')

        clf = GridSearchCV(estimator=lr, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=n_jobs)
        nested_scores = cross_val_score(clf, X=data.drop(columns=target),
                                        y=data[target], cv=outer_cv, scoring='roc_auc')

        nested_score_mean = nested_scores.mean()
        list_scores.append(nested_score_mean)

    if num_experiments == 1:
        ans = nested_scores
    else:
        ans = list_scores

    return ans


def crossvalidation_as_framework(df, target, n_folds=5, random_state=42, use_sample_weights=True, **kwargs):
    kfold_test = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)

    results = ModelResults()

    for i, (train, val) in enumerate(kfold_test.split(df, df[target])):
        train, val = df.iloc[train], df.iloc[val]

        if use_sample_weights:
            sample_weights = np.where(train[target] == 1,
                                      np.sum(train[target] == 0) / sum(train[target]), 1)
        else:
            sample_weights = None

        ql = feyn.connect_qlattice()
        ql.reset(42)
        models = ql.auto_run(train, target, sample_weights=sample_weights, **kwargs)

        for j in models:
            results.update(train, val, i, j)

    return results.df


class ModelResults:
    def __init__(self, kind="classification"):
        self.kind = kind

        if self.kind == "classification":
            self.df = pd.DataFrame(columns=['model_structure', 'fold', 'aic', 'bic', 'roc_auc_train',
                                            'accuracy_train', 'roc_auc_val', 'accuracy_val', 'pr_auc', 'f1'])
        elif self.kind == "regression":
            self.df = pd.DataFrame(columns=['model_structure', 'fold', 'aic', 'bic', 'rmse', 'mae', 'r2_score'])
        else:
            raise ValueError("kind must be classification or regression")

    def update(self, train, val, fold, model):
        if model:
            if self.kind == "classification":
                model_precision, model_recall, _ = precision_recall_curve(val[model.output], model.predict(val))
                if model:
                    self.df = self.df.append(pd.DataFrame(data={
                        'model_structure': [str(model.sympify(include_weights=False))],
                        'query_string': [model.to_query_string()],
                        'fold': [fold],
                        'aic': [model.aic],
                        'bic': [model.bic],
                        'roc_auc_train': [model.roc_auc_score(train)],
                        'accuracy_train': [model.accuracy_score(train)],
                        'roc_auc_val': [model.roc_auc_score(val)],
                        'accuracy_val': [model.accuracy_score(val)],
                        'pr_auc': [auc(model_recall, model_precision)],
                        'f1': [f1_score(val[model.output], model.predict(val) > model.accuracy_threshold(train)[0])]
                    }))

            elif self.kind == "regression":
                if model:
                    preds = model.predict(val)
                    self.df = self.df.append(pd.DataFrame(data={
                        'model_structure': [str(model.sympify(include_weights=False))],
                        'query_string': [model.to_query_string()],
                        'fold': [fold],
                        'aic': [model.aic],
                        'bic': [model.bic],
                        'rmse': [feyn.metrics.rmse(val[model.output], preds)],
                        'mae': [feyn.metrics.mae(val[model.output], preds)],
                        'r2_score': [feyn.metrics.r2_score(val[model.output], preds)]
                    }))


def _get_table_styler(result: pd.DataFrame, extra_columns: List[str], features_used: List[str]):
    # Reorganize data frame column order
    result_sum = result.sum()
    sorted_columns = extra_columns + sorted(
        features_used, key=lambda feature: result_sum[feature], reverse=True
    )
    result.index.name = "Model#"
    result = result[sorted_columns]

    ### Set styling

    # Rotate table header
    result_styler = result.style.set_table_styles(
        [
            dict(selector="th", props=[("width", "30px")]),
            dict(
                selector="th.col_heading",
                props=[("writing-mode", "vertical-rl"),
                       ('transform', 'rotateZ(0deg)'),
                       ('vertical-align', 'top'), ]
            ),
        ]
    )

    # Add table grid
    result_styler = result_styler.apply(
        lambda x: [
            "background: black"
            if isinstance(is_feature_present, bool) and is_feature_present
            else (
                "background: white; color: white; border: solid 1px black"
                if isinstance(is_feature_present, bool) and not is_feature_present
                else ""
            )
            for is_feature_present in x
        ],
        axis=1,
    )

    return result_styler


def model_features_chart(data: pd.DataFrame, models: Model, metric: str):
    features_used = list(set(reduce(lambda acc, m: acc + m.features, models, [])))

    # Construct feature part of table data frame
    data_dict = {
        feature: [feature in m.features for m in models]
        for feature in features_used
    }
    result = pd.DataFrame(data=data_dict, columns=features_used)

    # Add metric column
    metric_function = getattr(metrics, metric)
    result[metric] = [metric_function(data[m.target], m.predict(data)) for m in models]

    # Sort rows by performance metric
    result = result  # .sort_values(by=metric, ascending=False)

    # Add similarity column with the top model as reference model
    similarities = []
    for m in models:
        agree, disagree = prediction_overlap(data, models[0], m)
        agree_norm = agree / (agree + disagree)
        similarities.append(agree_norm)
    # result["agreement"] = similarities

    result_styler = _get_table_styler(result, [metric], features_used)

    return result_styler


def prediction_overlap(truth_df, model_1, model_2):
    pred_1 = model_1.predict(truth_df).round()
    pred_2 = model_2.predict(truth_df).round()

    agree = np.sum(pred_1 == pred_2)
    disagree = len(truth_df) - agree

    return agree, disagree


def modsum(models, train, test):
    model_list = []
    auc_list_train = []
    auc_list_test = []
    bic_list = []
    feat_list = []
    function_list = []
    loss_list = []
    i = 0
    for x in models:
        model_list.append(str(i))
        auc_list_train.append(str(x.roc_auc_score(train).round(2)))
        auc_list_test.append(str(x.roc_auc_score(test).round(2)))
        bic_list.append(str(x.bic.round(2)))
        feat_list.append(len(x.features))
        function_list.append(x.sympify(symbolic_lr=False, symbolic_cat=True, include_weights=False))
        loss_list.append(x.loss_value)
        i += 1
    df = pd.DataFrame(
        list(zip(model_list, auc_list_train, auc_list_test, bic_list, feat_list, function_list, loss_list)),
        columns=['Model', 'AUC Train', 'AUC Test', 'BIC', 'N. Features', 'Functional form', 'Loss'])

    return (df)
