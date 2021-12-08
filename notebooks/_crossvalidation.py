import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

import feyn


def get_crossvalidated_models(train_val, target, n_epochs=20, kind="classification", fast_cv=False, stypes=None,
                              sample_weights=None, **kwargs):
    models = candidate_search(train_val, target, n_epochs=n_epochs, kind=kind, fast_cv=fast_cv, stypes=stypes,
                              sample_weights=sample_weights, **kwargs)
    results = cross_validate(models, train_val, fast_cv=fast_cv)

    return results


def candidate_search(train_val, target, n_epochs=20, kind='classification', fast_cv=False, stypes=None,
                     sample_weights=None, **kwargs):
    models = []

    if not stypes:
        stypes = {}
        for f in train_val.columns:
            if train_val[f].dtype == 'object':
                stypes[f] = 'c'

    # Candidate Search
    if not sample_weights:
        sample_weights = np.where(train_val[target] == 1, np.sum(train_val[target] == 0) / sum(train_val[target]), 1)

    ql = feyn.connect_qlattice()  # Connecting
    ql.reset(42)  # Resetting
    all_data_models = ql.auto_run(data=train_val, output_name=target, kind=kind, stypes=stypes, n_epochs=n_epochs,
                                  sample_weights=sample_weights, **kwargs)
    models += all_data_models[:5]

    # Sampling variation
    if not fast_cv:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train, val) in enumerate(kfold.split(train_val, train_val[target])):
            ql = feyn.connect_qlattice()  # Connecting
            ql.reset(42)  # Resetting

            train, val = train_val.iloc[train], train_val.iloc[val]
            sw = np.where(train[target] == 1, np.sum(train[target] == 0) / sum(train[target]), 1)

            fold_models = ql.auto_run(data=train, output_name=target, kind=kind, stypes=stypes, n_epochs=n_epochs,
                                      query_string=query_string, sample_weights=sw)
            models += fold_models[:5]

    res = []

    for m in models:
        found = any([feyn._selection._canonical_compare(other, m) for other in res])
        math_invalid = np.isnan(m.predict(train_val)).any()

        if not found and not math_invalid:
            res.append(m)

    return res


def cross_validate(models, train_val, fast_cv=False):
    target = models[0].target

    results = ModelResults()

    # "Bootstrapping"
    for j in models:
        if fast_cv:
            fold_runs = 1
        else:
            fold_runs = 8  # 8 random 5-fold cross validations --> 40 folds in total

        # New shuffled KFold
        for x in range(fold_runs):
            kfold_repeated = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for i, (train, val) in enumerate(kfold_repeated.split(train_val, train_val[target])):
                train, val = train_val.iloc[train], train_val.iloc[val]

                ql = feyn.connect_qlattice()  # Connecting
                ql.reset(42)  # Resetting
                j.params = feyn._model.initialize_parameters(j.fnames)
                for _ in range(20):
                    j.fit(train)

                results.update(train, val, i, j)

        for _ in range(50):
            j.fit(train_val)

    df_results = results.df.drop(columns=["fold"])
    df_results_grouped = pd.concat(
        [df_results.groupby('model_structure').mean().add_suffix('_mean'),
         df_results.groupby('model_structure').std().add_suffix('_std'),
         df_results.groupby('model_structure').apply(
             lambda x: x.quantile(q=0.05).add_suffix('_05percentile')),
         df_results.groupby('model_structure').apply(
             lambda x: x.quantile(q=0.95).add_suffix('_05percentile'))], axis=1
    ).reset_index()

    return df_results_grouped


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
                        'fold': [fold],
                        'aic': [model.aic],
                        'bic': [model.bic],
                        'rmse': [feyn.metrics.rmse(val[model.output], preds)],
                        'mae': [feyn.metrics.mae(val[model.output], preds)],
                        'r2_score': [feyn.metrics.r2_score(val[model.output], preds)]
                    }))
