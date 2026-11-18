"""
Módulo de Treinamento e Validação de Modelos.
Responsável pelo particionamento estratificado dos dados, encapsulamento em
pipelines e ajuste de hiperparâmetros via validação cruzada.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Realiza a divisão estratificada inicial em conjuntos de treino (80%) e teste (20%).
    O conjunto de teste permanece isolado até a Fase 4 (Avaliação Final).
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def get_candidate_models(random_state=42):
    """
    Define os 4 algoritmos selecionados e suas grades enxutas de hiperparâmetros,
    cobrindo modelos lineares regularizados (L1/L2) e comitês de árvores (Bagging/Boosting).
    """
    models = {
        "Regressão Logística (LASSO L1)": {
            "estimator": LogisticRegression(
                l1_ratio=1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            "param_grid": {
                "classifier__C": [0.05, 0.1, 0.5, 1.0],
            },
        },
        "Regressão Logística (Ridge L2)": {
            "estimator": LogisticRegression(
                l1_ratio=0,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "Random Forest (Bagging)": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [6, 10, None],
                "classifier__min_samples_leaf": [5, 10],
            },
        },
        "Gradient Boosting": {
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__min_samples_leaf": [20, 40],
            },
        },
    }
    return models


def train_and_validate(X_train, y_train, preprocessor, cv_splits=5, random_state=42):
    """
    Encapsula pré-processador e classificador em Pipelines, executa GridSearchCV
    com StratifiedKFold no conjunto de treino e retorna os melhores modelos ajustados
    junto com a tabela comparativa de desempenho na validação cruzada.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    candidates = get_candidate_models(random_state=random_state)

    best_estimators = {}
    cv_results = []

    for name, config in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", config["estimator"]),
        ])

        search = GridSearchCV(
            estimator=pipe,
            param_grid=config["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_idx = search.best_index_
        mean_auc = search.cv_results_["mean_test_score"][best_idx]
        std_auc = search.cv_results_["std_test_score"][best_idx]

        # Limpa o prefixo 'classifier__' para exibição limpa dos hiperparâmetros
        clean_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        best_estimators[name] = search.best_estimator_
        cv_results.append({
            "Modelo": name,
            "ROC-AUC (CV Média)": mean_auc,
            "ROC-AUC (CV DP)": std_auc,
            "Melhores Hiperparâmetros": str(clean_params),
        })

    results_df = pd.DataFrame(cv_results).sort_values(
        by="ROC-AUC (CV Média)", ascending=False
    )

    return best_estimators, results_df