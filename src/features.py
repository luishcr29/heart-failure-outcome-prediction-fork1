"""
Módulo de engenharia e seleção de atributos (Feature Engineering).
Responsável por remover variáveis com vazamento de informação, 
transformar a base de medicamentos, fundir os dados e criar indicadores de ausência.
"""

import pandas as pd

# Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


def remove_leakage(patients_df, variable_groups):
    """
    Remove as colunas que causam vazamento de informação (data leakage),
    como identificadores, desfechos e informações obtidas apenas após a alta.
    
    Parâmetros:
        patients_df (pd.DataFrame): DataFrame original de pacientes.
        variable_groups (dict): Dicionário com os grupos de variáveis gerado no preprocessing.
        
    Retorna:
        pd.DataFrame: DataFrame sem as colunas de vazamento.
    """
    leakage_columns = (
        variable_groups["Desfechos e seguimento"]
        + variable_groups["Informações da alta"]
        + variable_groups["Identificação"]
    )
    
    # Remove as colunas que existem no DataFrame para evitar erros
    cols_to_drop = [col for col in leakage_columns if col in patients_df.columns]
    
    features_df = patients_df.drop(columns=cols_to_drop)
    print(f"Removidas {len(cols_to_drop)} colunas de vazamento. Restam {features_df.shape[1]} atributos clínicos.")
    
    return features_df


def process_drugs(drugs_df):
    """
    Transforma a base de medicamentos do formato longo (long) para o formato largo (wide).
    Cada medicamento torna-se uma coluna binária (1 = usou, 0 = não usou).
    
    Parâmetros:
        drugs_df (pd.DataFrame): DataFrame de medicamentos (dat_md).
        
    Retorna:
        pd.DataFrame: DataFrame indexado por inpatient.number com variáveis binárias para cada fármaco.
    """
    # Cria uma tabela de contingência: linhas = internações, colunas = medicamentos
    drugs_wide = pd.crosstab(
        index=drugs_df["inpatient.number"], 
        columns=drugs_df["Drug_name"]
    )
    
    # Garante que os valores são estritamente binários (caso haja duplicados no original)
    drugs_wide = (drugs_wide > 0).astype(int)
    
    # Adiciona um prefixo para facilitar a identificação dos atributos
    drugs_wide.columns = [f"drug_{col}" for col in drugs_wide.columns]
    
    print(f"Base de medicamentos processada: {drugs_wide.shape[1]} fármacos extraídos.")
    return drugs_wide


def add_missing_indicators(features_df, threshold=20.0):
    """
    Cria colunas indicadoras binárias (Missing Flags) para variáveis clínicas
    cuja percentagem de dados em falta seja superior a um determinado limiar.
    Esta técnica capta o padrão NMAR (Not Missing At Random) identificado na EDA.
    
    Parâmetros:
        features_df (pd.DataFrame): DataFrame com os atributos clínicos.
        threshold (float): Limiar percentual de dados em falta para criar o indicador.
        
    Retorna:
        pd.DataFrame: DataFrame com as novas colunas indicadoras adicionadas.
    """
    df_out = features_df.copy()
    missing_pct = df_out.isna().mean() * 100
    
    # Filtra as colunas que ultrapassam o limiar de faltantes
    cols_with_high_missing = missing_pct[missing_pct > threshold].index.tolist()
    
    for col in cols_with_high_missing:
        indicator_col_name = f"{col}_is_missing"
        df_out[indicator_col_name] = df_out[col].isna().astype(int)
        
    print(f"Adicionados {len(cols_with_high_missing)} indicadores de dados em falta (threshold > {threshold}%).")
    
    return df_out

def build_feature_set(patients_df, drugs_df, variable_groups):
    """
    Orquestra a construção do conjunto de dados final para modelagem,
    executando a limpeza, processamento de fármacos e junção das bases.
    """
    # 1. Processa e roda a base de medicamentos (index = inpatient.number)
    df_drugs_wide = process_drugs(drugs_df)
    
    # 2. Fundo as bases originais usando inpatient.number 
    # (Definimos temporariamente o inpatient.number como índice na clínica)
    df_clinical_indexed = patients_df.set_index("inpatient.number")
    X_features = df_clinical_indexed.join(df_drugs_wide, how="left")
    
    # Preenche com 0 os medicamentos para os pacientes que não têm registo na base de fármacos
    drug_cols = [col for col in X_features.columns if col.startswith("drug_")]
    X_features[drug_cols] = X_features[drug_cols].fillna(0).astype(int)
    
    # 3. Remove vazamento de dados (Desfechos, Info Alta, e as "IDs" se ainda existissem como colunas)
    X_features = remove_leakage(X_features, variable_groups)
    
    # 4. Cria indicadores para exames não solicitados (captura a intenção médica)
    X_features = add_missing_indicators(X_features, threshold=20.0)
    
    # Remove colunas 100% ausentes ou constantes (variância zero)
    constant_cols = [col for col in X_features.columns if X_features[col].nunique(dropna=True) <= 1]
    if constant_cols:
        X_features = X_features.drop(columns=constant_cols)
        print(f"Removidas {len(constant_cols)} colunas vazias/constantes: {constant_cols}")
    
    return X_features

def build_preprocessor(variable_types):
    """
    Constrói o ColumnTransformer para pré-processamento matemático das features.
    Deve ser ajustado (.fit) exclusivamente dentro da validação cruzada para evitar data leakage.
    """
    num_cols = variable_types["continuas"] + variable_types["discretas"]
    cat_cols = variable_types["categoricas"]
    bin_cols = variable_types["binarias"]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    # Imputação pela moda nas binárias protege contra nulos residuais (< 0.2% em comorbidades)
    bin_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("bin", bin_pipeline, bin_cols),
        ],
        remainder="drop",
    )

    return preprocessor