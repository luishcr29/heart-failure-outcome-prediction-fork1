from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.utils import save_fig

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def evaluate_models_on_test(best_models, X_test, y_test):
    """
    Avalia todos os pipelines treinados no conjunto de teste isolado
    e retorna uma tabela comparativa ordenada por ROC-AUC.
    """
    rows = []
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Modelo": name,
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Sensibilidade (Recall)": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })

    results_df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False)
    return results_df


def plot_roc_curves(best_models, X_test, y_test, filename="roc_curves_test"):
    """
    Plota as curvas ROC sobrepostas de todos os modelos no conjunto de teste
    e salva a figura em PDF vetorial.
    """

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, model in best_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Aleatório (AUC = 0.5000)")
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
    ax.set_title("Curvas ROC no Conjunto de Teste (n = 402)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    save_fig(fig, filename)


def plot_confusion_matrix(model, model_name, X_test, y_test, filename="confusion_matrix_best"):
    """
    Plota a matriz de confusão do modelo selecionado no conjunto de teste.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Estável (0)", "Óbito/Readmissão (1)"],
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Matriz de Confusão — {model_name}")
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Real")
    ax.grid(False)

    plt.tight_layout()
    plt.show()
    save_fig(fig, filename)


def _clean_feature_names(raw_names):
    """Remove os prefixos do ColumnTransformer (num__, cat__, bin__)."""
    return [name.split("__", 1)[-1] for name in raw_names]


def plot_clinical_interpretability(
    rf_pipeline, lasso_pipeline, top_n=15, filename="feature_importance.pdf"
):
    """
    Extrai e plota lado a lado:
    1) Top N variáveis por importância de Gini no Random Forest.
    2) Top N variáveis por magnitude de coeficiente não-nulo no LASSO (com direção de risco).
    """
    # 1. Importância no Random Forest (Gini)
    rf_names = _clean_feature_names(rf_pipeline.named_steps["preprocessor"].get_feature_names_out())
    rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_
    rf_df = (
        pd.DataFrame({"Atributo": rf_names, "Importância": rf_importances})
        .sort_values(by="Importância", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    # 2. Coeficientes não-nulos no LASSO (L1)
    lasso_names = _clean_feature_names(lasso_pipeline.named_steps["preprocessor"].get_feature_names_out())
    lasso_coefs = lasso_pipeline.named_steps["classifier"].coef_.ravel()
    lasso_df = pd.DataFrame({"Atributo": lasso_names, "Coeficiente": lasso_coefs})
    n_selected = (lasso_df["Coeficiente"] != 0).sum()
    lasso_top = (
        lasso_df[lasso_df["Coeficiente"] != 0]
        .assign(Abs=lambda d: d["Coeficiente"].abs())
        .sort_values(by="Abs", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    print(f"LASSO (L1) manteve {n_selected} de {len(lasso_names)} atributos com coeficiente não-nulo.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Painel Esquerdo: Random Forest
    axes[0].barh(rf_df["Atributo"], rf_df["Importância"], color="#2b6cb0")
    axes[0].set_title(f"Top {top_n} Atributos — Random Forest (Gini)")
    axes[0].set_xlabel("Importância Relativa")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    axes[0].grid(axis="x", alpha=0.3)

    # Painel Direito: LASSO
    colors = ["#c53030" if c > 0 else "#2f855a" for c in lasso_top["Coeficiente"]]
    axes[1].barh(lasso_top["Atributo"], lasso_top["Coeficiente"], color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title(f"Top {top_n} Coeficientes — LASSO L1 (Vermelho = ↑ Risco)")
    axes[1].set_xlabel("Coeficiente Log-Odds Padronizado")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()
    save_fig(fig, filename)

    return rf_df.iloc[::-1], lasso_top.drop(columns=["Abs"]).iloc[::-1]