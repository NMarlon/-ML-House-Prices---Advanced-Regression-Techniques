# Fase 1: Exploracao e Limpeza de Dados
# House Prices - Advanced Regression Techniques
# Rode com: py -3.14 scripts/fase1_exploracao.py

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. CARREGAR OS DADOS
# ============================================================
print("=" * 60)
print("FASE 1: EXPLORACAO E LIMPEZA DE DADOS")
print("=" * 60)

# Caminho do dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Carregar train e test
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"\n1. DADOS CARREGADOS")
print(f"   Train: {train.shape[0]} linhas x {train.shape[1]} colunas")
print(f"   Test:  {test.shape[0]} linhas x {test.shape[1]} colunas")

# ============================================================
# 2. VISÃO GERAL
# ============================================================
print(f"\n2. VISÃO GERAL")
print(f"\n   Primeiras 5 linhas do train:")
print(train.head().to_string(index=False))

print(f"\n   Tipos de dados:")
print(train.dtypes.value_counts().to_string())

print(f"\n   Estatísticas descritivas (numéricas):")
print(train.describe().to_string())

# ============================================================
# 3. ANALISE DO TARGET (SalePrice)
# ============================================================
print(f"\n3. ANALISE DO TARGET (SalePrice)")
print(f"   Media:    R$ {train['SalePrice'].mean():,.2f}")
print(f"   Mediana:  R$ {train['SalePrice'].median():,.2f}")
print(f"   Min:      R$ {train['SalePrice'].min():,.2f}")
print(f"   Max:      R$ {train['SalePrice'].max():,.2f}")
print(f"   Desvio:   R$ {train['SalePrice'].std():,.2f}")

# Assimetria (skewness)
skew = train['SalePrice'].skew()
print(f"   Assimetria (skew): {skew:.4f}")
if skew > 1:
    print("   -> Assimetria ALTA: dados muito concentrados em valores baixos")
    print("   -> Vamos aplicar log transform depois!")
elif skew > 0.5:
    print("   -> Assimetria moderada")
else:
    print("   -> Distribuição relativamente simétrica")

# ============================================================
# 4. VALORES FALTANTES
# ============================================================
print(f"\n4. VALORES FALTANTES (Missing Values)")

# Contar missing por coluna
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).round(1)

print(f"\n   Colunas com valores faltantes ({len(missing)} de {len(train.columns)}):")
print(f"   {'Coluna':<25} {'Missing':>8} {'%':>6}")
print(f"   {'-'*25} {'-'*8} {'-'*6}")
for col in missing.index:
    print(f"   {col:<25} {missing[col]:>8} {missing_pct[col]:>5.1f}%")

total_missing = train.isnull().sum().sum()
print(f"\n   Total de valores faltantes: {total_missing:,}")

# ============================================================
# 5. CORRELACOES COM SalePrice
# ============================================================
print(f"\n5. CORRELACOES COM SalePrice")

# Correlação apenas com colunas numéricas
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlations = train[numeric_cols].corr()["SalePrice"].sort_values(ascending=False)

print(f"\n   Top 10 features MAIS correlacionadas com SalePrice:")
print(f"   {'Feature':<25} {'Correlacao':>10}")
print(f"   {'-'*25} {'-'*10}")
for i, (col, corr) in enumerate(correlations.head(11).items()):
    if col == "SalePrice":
        continue
    bar = "█" * int(abs(corr) * 20)
    print(f"   {col:<25} {corr:>10.4f} {bar}")

print(f"\n   Top 5 features MENOS correlacionadas:")
for col, corr in correlations.tail(5).items():
    print(f"   {col:<25} {corr:>10.4f}")

# ============================================================
# 6. ANALISE DE OUTLIERS
# ============================================================
print(f"\n6. ANALISE DE OUTLIERS")

# Usar IQR (Interquartile Range) para detectar outliers
key_features = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "LotArea"]

print(f"\n   Outliers nas features principais (metodo IQR):")
print(f"   {'Feature':<25} {'Outliers':>10} {'%':>6}")
print(f"   {'-'*25} {'-'*10} {'-'*6}")

for col in key_features:
    if col in train.columns:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((train[col] < lower) | (train[col] > upper)).sum()
        pct = outliers / len(train) * 100
        print(f"   {col:<25} {outliers:>10} {pct:>5.1f}%")

# Outliers no preço
Q1_price = train["SalePrice"].quantile(0.25)
Q3_price = train["SalePrice"].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price
outliers_price = ((train["SalePrice"] < lower_price) | (train["SalePrice"] > upper_price)).sum()
print(f"\n   SalePrice: {outliers_price} outliers ({outliers_price/len(train)*100:.1f}%)")
print(f"   Limite inferior: R$ {lower_price:,.0f}")
print(f"   Limite superior: R$ {upper_price:,.0f}")

# ============================================================
# 7. FEATURES CATEGORICAS IMPORTANTES
# ============================================================
print(f"\n7. FEATURES CATEGORICAS IMPORTANTES")

cat_features = ["Neighborhood", "HouseStyle", "SaleCondition", "OverallQual"]

for col in cat_features:
    if col in train.columns:
        print(f"\n   {col}:")
        if train[col].dtype == "object":
            counts = train[col].value_counts().head(5)
            for val, cnt in counts.items():
                avg_price = train[train[col] == val]["SalePrice"].mean()
                print(f"     {val:<20} {cnt:>4} casas  |  Preco medio: R$ {avg_price:,.0f}")
        else:
            # OverallQual é numérico mas funciona como categórico
            for val in sorted(train[col].unique()):
                cnt = (train[col] == val).sum()
                avg_price = train[train[col] == val]["SalePrice"].mean()
                print(f"     {val:<20} {cnt:>4} casas  |  Preco medio: R$ {avg_price:,.0f}")

# ============================================================
# 8. RESUMO E PROXIMOS PASSOS
# ============================================================
print(f"\n" + "=" * 60)
print(f"RESUMO DA FASE 1")
print(f"=" * 60)

print(f"""
O que aprendemos ate agora:

1. O dataset tem {train.shape[0]} casas com {train.shape[1]} features
2. SalePrice varia de R$ {train['SalePrice'].min():,.0f} a R$ {train['SalePrice'].max():,.0f}
3. A distribuicao do preco e assimetrica (skew={skew:.2f}) -> precisa de log transform
4. {len(missing)} colunas tem valores faltantes -> precisamos tratar isso
5. As features mais importantes sao: {', '.join(correlations.head(4).index[1:])}
6. Existem outliers em areas (GrLivArea, LotArea) -> podemos remover

PROXIMOS PASSOS (Fase 2):
- Tratar valores faltantes (preencher ou remover)
- Remover outliers
- Aplicar log transform no SalePrice
- Feature engineering (criar novas features)
- Codificar variáveis categóricas
""")
