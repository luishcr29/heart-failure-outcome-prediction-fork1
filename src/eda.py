"""
Módulo de Análise Exploratória de Dados (EDA).
Responsável por gerar visualizações e resumos estatísticos.
"""

import textwrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from IPython.display import FileLink, Markdown, display
from src.utils import SERIES_COLOR, NEUTRAL_COLOR, NEUTRAL_LIGHT_COLOR, MUTED_COLOR, RESULTS_DIR, CSV_DIR, save_fig

# Importações para geração do PDF com as tabelas
# Requer: pip install reportlab
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

def variable_kind(series):
    """Determina o tipo da variável clínica."""
    n_unique = series.nunique()
    if n_unique == 2:
        return "binária"
    if pd.api.types.is_numeric_dtype(series):
        return "numérica" if n_unique > 10 else "discreta"
    return "categórica"

def observed_values(series):
    """Formata os valores observados para o relatório."""
    values = series.dropna()
    if pd.api.types.is_numeric_dtype(series) and values.nunique() > 10:
        return f"{values.min():g} – {values.median():g} – {values.max():g}"
    return ", ".join(str(v) for v in sorted(values.unique(), key=str))

def generate_variable_overview(patients, data_dictionary, variable_groups):
    """
    Gera e guarda o resumo geral das variáveis (CSV) e exibe no notebook
    apenas a Tabela 1 (Resumo por Grupo Clínico).
    """
    column_group = {column: group for group, columns in variable_groups.items() for column in columns}
    descriptions = data_dictionary.set_index("Variables")["Description"]

    variable_overview = pd.DataFrame(
        {
            "grupo": [column_group.get(c, "Desconhecido") for c in patients.columns],
            "variável": patients.columns,
            "descrição": descriptions.reindex(patients.columns).values,
            "tipo": [variable_kind(patients[c]) for c in patients.columns],
            "faltantes (%)": patients.isna().mean().mul(100).round(1).values,
            "valores observados": [observed_values(patients[c]) for c in patients.columns],
        }
    )

    variable_overview["grupo"] = pd.Categorical(
        variable_overview["grupo"], categories=list(variable_groups), ordered=True
    )
    variable_overview = variable_overview.sort_values("grupo", kind="stable").reset_index(drop=True)
    # variable_overview.to_csv(RESULTS_DIR / "variable_overview.csv", index=False)
    variable_overview.to_csv(CSV_DIR / "variable_overview.csv", index=False)
        
    summary = variable_overview.groupby("grupo", observed=True).agg(
        n_variáveis=("variável", "size"),
        faltantes_médio=("faltantes (%)", "mean"),
        faltantes_máx=("faltantes (%)", "max"),
    ).round(1)

    print("Resumo por Grupo Clínico:")
    display(summary)
    
    csv_path = CSV_DIR / "variable_overview.csv"
    display(FileLink(csv_path, result_html_prefix="📄 Arquivo CSV contendo os dados brutos salvo! Abrir CSV: "))
        
    return variable_overview


# def export_variable_overview_pdf(variable_overview, path):
def export_variable_overview_pdf(variable_overview):
    """
    Exporta uma tabela formatada por grupo clínico para um PDF (A4 paisagem).
    """
    pdf_name = "variable_overview.pdf"
    path = RESULTS_DIR / pdf_name
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=styles["Heading2"], spaceAfter=6)
    cell = ParagraphStyle("c", parent=styles["BodyText"], fontSize=7.5, leading=9.5)
    cell_wrap = ParagraphStyle("cw", parent=cell, wordWrap="CJK")  # quebra textos longos sem espaços
    head = ParagraphStyle("h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def p(value, style=cell):
        text = "—" if pd.isna(value) else str(value)
        return Paragraph(escape(text), style)

    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="Resumo geral das variáveis",
    )

    columns = ["variável", "descrição", "tipo", "faltantes (%)", "valores observados"]
    col_widths = [3.5 * cm, 8.0 * cm, 2.5 * cm, 2.2 * cm, 10.5 * cm]  # soma = 26,7 cm

    story = []
    groups = list(variable_overview.groupby("grupo", observed=True))
    for i, (group, table) in enumerate(groups):
        story.append(Paragraph(escape(str(group)), title_style))

        data = [[Paragraph(c, head) for c in columns]]
        for _, row in table.iterrows():
            data.append([
                p(row["variável"]), p(row["descrição"]), p(row["tipo"]),
                p(row["faltantes (%)"]), p(row["valores observados"], cell_wrap),
            ])

        t = Table(data, colWidths=col_widths, repeatRows=1)  # cabeçalho repetido a cada página
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4B7C")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F4F9")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C2D3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        if i < len(groups) - 1:
            story.append(PageBreak())  # cada grupo começa em uma nova página

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.8 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    
    display(FileLink(path, result_html_prefix="📄 Arquivo PDF contendo as tabelas salvo! Abrir PDF: "))

def generate_drug_summary(drugs, patients):
    """Gera um resumo estatístico do uso de medicamentos."""
    drug_summary = (
        drugs.groupby("Drug_name")
        .agg(registros=("inpatient.number", "size"), internações=("inpatient.number", "nunique"))
        .assign(**{"internações (%)": lambda t: t["internações"].div(len(patients)).mul(100).round(1)})
        .sort_values("internações", ascending=False)
    )
    print(f"Internações com ao menos um medicamento: {drugs['inpatient.number'].nunique()} de {len(patients)}")
    print(f"Internações presentes em drugs e ausentes de patients: {len(set(drugs['inpatient.number']) - set(patients['inpatient.number']))}")
    
    # display(drug_summary)
    return drug_summary

def plot_missing_rate_by_variable(patients):
    """Calcula estatísticas de dados faltantes e plota o gráfico de barras por variável."""
    missing_rate = patients.isna().mean().mul(100)
    
    missing_bins = pd.cut(missing_rate, bins=[-0.1, 0, 20, 50, 100], labels=["0%", "(0, 20]%", "(20, 50]%", "(50, 100]%"])
    print("Nº de variáveis por faixa de faltantes:")
    print(missing_bins.value_counts().sort_index().to_string())
    print(f"\nrespiratory.support. — faltantes: {missing_rate['respiratory.support.']:.1f}%")
    print(f"Internações sem nenhum faltante: {patients.notna().all(axis=1).sum()}")
    
    missing_only = missing_rate[missing_rate > 0].sort_values()
    
    fig, ax = plt.subplots(figsize=(7, 0.16 * len(missing_only) + 1))
    ax.barh(missing_only.index, missing_only.values, color=SERIES_COLOR, height=0.7)
    for threshold in (20, 50):
        ax.axvline(threshold, color=MUTED_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Valores faltantes (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=6)
    ax.margins(y=0.005)
    ax.grid(axis="y", visible=False)
    plt.show()
    save_fig(fig, "missing_rate_by_variable")

def plot_missing_matrix(patients, variable_groups):
    """
    Gera a matriz visual de dados faltantes (Missing Matrix).
    """
    column_group = {column: group for group, columns in variable_groups.items() for column in columns}
    missing_rate = patients.isna().mean().mul(100)
    
    missing_columns = [c for c in patients.columns if missing_rate[c] > 0]
    missing_columns = sorted(missing_columns, key=lambda c: list(variable_groups).index(column_group[c]))
    
    missing_matrix = patients[missing_columns].isna()
    missing_matrix = missing_matrix.loc[missing_matrix.sum(axis=1).sort_values(kind="stable").index]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(missing_matrix.values, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([NEUTRAL_LIGHT_COLOR, SERIES_COLOR]))
    
    group_of_column = pd.Series([column_group[c] for c in missing_columns])
    boundaries = np.flatnonzero(group_of_column.ne(group_of_column.shift())).tolist() + [len(missing_columns)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start > 0:
            ax.axvline(start - 0.5, color="white", linewidth=1.5)
    ax.set_xticks([(start + end - 1) / 2 for start, end in zip(boundaries[:-1], boundaries[1:])])
    ax.set_xticklabels(group_of_column.iloc[np.asarray(boundaries[:-1], dtype=int)], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel("Internações (mais completa no topo)")
    ax.grid(False)
    plt.show()
    save_fig(fig, "missing_matrix")

def plot_outcome_prevalence(patients, target):
    """Gera a tabela e o gráfico de prevalência dos diferentes desfechos e do alvo."""
    BINARY_OUTCOMES = {
        "death.within.28.days": "Óbito em 28 dias",
        "death.within.3.months": "Óbito em 3 meses",
        "death.within.6.months": "Óbito em 6 meses",
        "re.admission.within.28.days": "Readmissão em 28 dias",
        "re.admission.within.3.months": "Readmissão em 3 meses",
        "re.admission.within.6.months": "Readmissão em 6 meses",
        "return.to.emergency.department.within.6.months": "Retorno ao PS em 6 meses",
    }
    
    outcome_summary = pd.DataFrame(
        {
            "positivos": patients[list(BINARY_OUTCOMES)].sum().astype(int),
            "prevalência (%)": patients[list(BINARY_OUTCOMES)].mean().mul(100).round(1),
        }
    ).rename(index=BINARY_OUTCOMES)
    
    outcome_summary.loc["Óbito ou readmissão em 6 meses (alvo)"] = [target.sum(), round(target.mean() * 100, 1)]
    
    display(outcome_summary)
    
    fig, ax = plt.subplots(figsize=(6, 3.2))
    plot_data = outcome_summary.iloc[::-1]
    colors = [SERIES_COLOR if "alvo" in label else NEUTRAL_COLOR for label in plot_data.index]
    ax.barh(plot_data.index, plot_data["prevalência (%)"], color=colors, height=0.6)
    for position, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(row["prevalência (%)"] + 0.8, position, f"{row['positivos']:.0f} ({row['prevalência (%)']:.1f}%)",
                va="center", fontsize=8, color=MUTED_COLOR)
    ax.set_xlabel("Prevalência (%)")
    ax.set_xlim(0, plot_data["prevalência (%)"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    plt.show()
    save_fig(fig, "outcome_prevalence")


def print_outcome_details(patients, target):
    """Exibe o detalhamento das interseções e consistências dos desfechos."""
    READMISSION_COLUMNS = ["re.admission.within.28.days", "re.admission.within.3.months", "re.admission.within.6.months"]
    readmitted = patients[READMISSION_COLUMNS].eq(1).any(axis=1)
    died = patients["death.within.6.months"].eq(1)
    
    for event in ("death", "re.admission"):
        d28, m3, m6 = (patients[f"{event}.within.{window}"] for window in ("28.days", "3.months", "6.months"))
        print(f"{event}: 28d sem 3m = {(d28 > m3).sum()} | 3m sem 6m = {(m3 > m6).sum()}")

    print(f"\nSó readmissão: {(readmitted & ~died).sum()}")
    print(f"Só óbito: {(died & ~readmitted).sum()}")
    print(f"Readmissão e óbito: {(readmitted & died).sum()}")
    print(f"Óbitos durante a internação: {patients['outcome.during.hospitalization'].eq('Dead').sum()}")
    print(f"\nTotal no alvo: {target.sum()} de {len(target)} ({target.mean():.1%})")

def generate_table1_stratified(patients, target, key_features):
    """
    Gera a Tabela 1 consolidada (estatísticas descritivas estratificadas pelo alvo).
    Unifica variáveis contínuas e categóricas em um único DataFrame para visualização limpa.
    """
    df = patients[key_features].copy()
    df['Desfecho'] = target.map({0: 'Estável (0)', 1: 'Óbito/Readmissão (1)'})
    
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns.drop('Desfecho', errors='ignore')
    
    rows = []
    
    # 1. Variáveis Contínuas/Discretas
    rows.append({'Característica': '[ Contínuas: Média ± DP ]', 'Estável (0)': '', 'Óbito/Readmissão (1)': ''})
    for col in num_cols:
        stats = df.groupby('Desfecho')[col].agg(['mean', 'std'])
        val_0 = f"{stats.loc['Estável (0)', 'mean']:.2f} ± {stats.loc['Estável (0)', 'std']:.2f}"
        val_1 = f"{stats.loc['Óbito/Readmissão (1)', 'mean']:.2f} ± {stats.loc['Óbito/Readmissão (1)', 'std']:.2f}"
        rows.append({'Característica': col, 'Estável (0)': val_0, 'Óbito/Readmissão (1)': val_1})
        
    # 2. Variáveis Categóricas
    rows.append({'Característica': '[ Categóricas: Distribuição % ]', 'Estável (0)': '', 'Óbito/Readmissão (1)': ''})
    for col in cat_cols:
        cross = pd.crosstab(df[col], df['Desfecho'], normalize='columns') * 100
        for cat in cross.index:
            val_0 = f"{cross.loc[cat, 'Estável (0)']:.1f}%"
            val_1 = f"{cross.loc[cat, 'Óbito/Readmissão (1)']:.1f}%"
            rows.append({'Característica': f"{col} = {cat}", 'Estável (0)': val_0, 'Óbito/Readmissão (1)': val_1})
            
    table1 = pd.DataFrame(rows)
    
    # Exibe a tabela sem o índice numérico lateral para ficar mais limpa
    display(table1.style.hide(axis="index"))

def plot_clinical_bivariate(patients, target):
    """
    Gera boxplots comparativos para identificar o comportamento de variáveis 
    críticas e outliers biológicos entre as duas classes do alvo.
    """
    df = patients.copy()
    df["Desfecho"] = target.map({0: "Estável", 1: "Óbito/Readmissão"})

    # Seleção de variáveis clínicas com forte impacto em Insuficiência Cardíaca
    vars_to_plot = [
        "LVEF",
        "CCI.score",
        "systolic.blood.pressure",
        "brain.natriuretic.peptide",
    ]
    valid_vars = [var for var in vars_to_plot if var in df.columns]

    if not valid_vars:
        raise ValueError("Nenhuma das variáveis clínicas selecionadas foi encontrada em patients.")

    fig, axes = plt.subplots(
        1, len(valid_vars), figsize=(16, 4), squeeze=False
    )
    axes = axes.ravel()

    for ax, var in zip(axes, valid_vars):
        sns.boxplot(
            x="Desfecho",
            y=var,
            hue="Desfecho",
            data=df,
            ax=ax,
            palette={"Estável": NEUTRAL_COLOR, "Óbito/Readmissão": SERIES_COLOR},
            showfliers=True,
            legend=False,
        )
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    save_fig(fig, "bivariate_clinical_analysis")
    
def analyze_collinearity(df, threshold=0.85):
    """
    Calcula a correlação de Pearson para variáveis contínuas e identifica 
    pares altamente correlacionados (colinearidade).
    Apresenta os resultados ordenados em uma tabela Pandas formatada.
    """
    # Seleciona apenas as colunas numéricas
    num_df = df.select_dtypes(include=[np.number])
    
    # Calcula a matriz de correlação absoluta
    corr_matrix = num_df.corr().abs()
    
    # Seleciona o triângulo superior da matriz para evitar duplicatas (A-B e B-A)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Busca os pares acima do limiar
    collinear_pairs = []
    for column in upper.columns:
        highly_correlated_with = upper[column][upper[column] > threshold].index.tolist()
        for correlated_var in highly_correlated_with:
            collinear_pairs.append({
                'Variável 1': correlated_var,
                'Variável 2': column,
                'Correlação (Pearson)': upper.loc[correlated_var, column]
            })
            
    if not collinear_pairs:
        print(f"Nenhum par com correlação > {threshold} encontrado.")
        return pd.DataFrame()
        
    # Converte para DataFrame e ordena da maior para a menor correlação
    df_collinear = pd.DataFrame(collinear_pairs)
    df_collinear = df_collinear.sort_values(by='Correlação (Pearson)', ascending=False)
    
    # Exibe a tabela ocultando o índice lateral e formatando as casas decimais
    print(f"--- Pares Altamente Correlacionados (Pearson > {threshold}) ---")
    display(df_collinear.style.hide(axis="index").format({'Correlação (Pearson)': "{:.3f}"}))
    
    return df_collinear