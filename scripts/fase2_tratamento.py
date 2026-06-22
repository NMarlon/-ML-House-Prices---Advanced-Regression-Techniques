# Fase 2: Tratamento de Dados
# House Prices - Advanced Regression Techniques
# Rode com: py -3.14 scripts/fase2_tratamento.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # modo sem GUI (salva PNG sem abrir janela)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("FASE 2: TRATAMENTO DE DADOS")
print("=" * 60)

# Carregar dados
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"\nDataset original: {train.shape[0]} linhas x {train.shape[1]} colunas")

# ============================================================
# 1. TRATAR VALORES FALTANTES
# ============================================================
print(f"\n1. TRATANDO VALORES FALTANTES")

# Estratégia: colunas com >80% missing -> remover
# Colunas com menos -> preencher com valor apropriado

missing_pct = (train.isnull().sum() / len(train) * 100)
high_missing = missing_pct[missing_pct > 80].index.tolist()
print(f"\n   Colunas com >80% missing (removidas): {high_missing}")

# Remover colunas com >80% missing
train.drop(columns=high_missing, inplace=True)
test.drop(columns=high_missing, inplace=True)

# Preencher colunas numéricas restantes com mediana
num_cols = train.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if train[col].isnull().any():
        median_val = train[col].median()
        train[col].fillna(median_val, inplace=True)
        test[col].fillna(median_val, inplace=True)
        print(f"   {col}: preenchido com mediana ({median_val:.1f})")

# Preencher colunas categóricas restantes com "None"
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if train[col].isnull().any():
        train[col].fillna("None", inplace=True)
        test[col].fillna("None", inplace=True)
        print(f"   {col}: preenchido com 'None'")

# Verificar se sobrou algum missing
remaining = train.isnull().sum().sum()
print(f"\n   Valores faltantes restantes: {remaining}")

# ============================================================
# 2. REMOVER OUTLIERS
# ============================================================
print(f"\n2. REMOVENDO OUTLIERS")

# Regras baseadas na análise da Fase 1:
# - GrLivArea > 4000 (casas muito grandes com preço baixo)
# - LotArea > 100000 (terrenos muito grandes)
# - TotalBsmtSF > 3000 (porão muito grande)

outlier_rules = [
    ("GrLivArea", 4000, ">"),
    ("LotArea", 100000, ">"),
    ("TotalBsmtSF", 3000, ">"),
]

before = len(train)
for col, threshold, direction in outlier_rules:
    if direction == ">":
        mask = train[col] > threshold
    else:
        mask = train[col] < threshold
    count = mask.sum()
    if count > 0:
        print(f"   {col} {direction} {threshold}: {count} outliers removidos")
        train = train[~mask]

after = len(train)
print(f"\n   Total removidos: {before - after} linhas")
print(f"   Dataset agora: {after} linhas")

# ============================================================
# 3. LOG TRANSFORM NO SALEPRICE
# ============================================================
print(f"\n3. LOG TRANSFORM NO SALEPRICE")

# Antes
skew_before = train['SalePrice'].skew()
print(f"   Skew antes: {skew_before:.4f}")

# Aplicar log1p (log(1+x) para evitar log(0))
train['SalePrice'] = np.log1p(train['SalePrice'])

# Depois
skew_after = train['SalePrice'].skew()
print(f"   Skew depois: {skew_after:.4f}")
print(f"   Redução de assimetria: {(1 - abs(skew_after)/abs(skew_before))*100:.1f}%")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print(f"\n4. FEATURE ENGINEERING")

# Criar novas features
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
print(f"   TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF")

train['TotalBath'] = (train['FullBath'] + 0.5 * train['HalfBath'] +
                      train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath'])
test['TotalBath'] = (test['FullBath'] + 0.5 * test['HalfBath'] +
                     test['BsmtFullBath'] + 0.5 * test['BsmtHalfBath'])
print(f"   TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath")

train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']
print(f"   HouseAge = YrSold - YearBuilt")

train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']
test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']
print(f"   RemodAge = YrSold - YearRemodAdd")

train['HasPool'] = (train['PoolArea'] > 0).astype(int)
test['HasPool'] = (test['PoolArea'] > 0).astype(int)
print(f"   HasPool = 1 se PoolArea > 0")

train['HasGarage'] = (train['GarageArea'] > 0).astype(int)
test['HasGarage'] = (test['GarageArea'] > 0).astype(int)
print(f"   HasGarage = 1 se GarageArea > 0")

# ============================================================
# 5. CODIFICAR VARIÁVEIS CATEGÓRICAS
# ============================================================
print(f"\n5. CODIFICANDO VARIÁVEIS CATEGÓRICAS")

# Label Encoding para variáveis com ordem natural
ordinal_cols = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
}

for col, order in ordinal_cols.items():
    if col in train.columns:
        mapping = {val: i for i, val in enumerate(order)}
        train[col] = train[col].map(mapping).fillna(0).astype(int)
        test[col] = test[col].map(mapping).fillna(0).astype(int)
        print(f"   {col}: Label Encoding ({len(order)} níveis)")

# One-Hot Encoding para o resto das categóricas
cat_remaining = train.select_dtypes(include=['object']).columns.tolist()
print(f"\n   One-Hot Encoding para {len(cat_remaining)} colunas categóricas restantes")

# Combinar train e test pra garantir mesmas colunas
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=cat_remaining, drop_first=True)

# Separar de volta
train = combined.iloc[:len(train)].reset_index(drop=True)
test = combined.iloc[len(train):].reset_index(drop=True)

print(f"\n   Dataset final: {train.shape[0]} linhas x {train.shape[1]} colunas")
print(f"   Test final: {test.shape[0]} linhas x {test.shape[1]} colunas")

# ============================================================
# 6. VISUALIZAÇÕES
# ============================================================
print(f"\n6. GERANDO VISUALIZAÇÕES")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SalePrice antes vs depois do log
# (simulando o "antes" com exp)
original_prices = np.expm1(train['SalePrice'])
axes[0].hist(original_prices, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('SalePrice Original (simulado)')
axes[0].set_xlabel('Preço')

axes[1].hist(train['SalePrice'], bins=50, color='coral', edgecolor='white')
axes[1].set_title('SalePrice após Log Transform')
axes[1].set_xlabel('log(Preço)')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'saleprice_log_transform.png'), dpi=150)
plt.close()
print(f"   reports/saleprice_log_transform.png")

# Correlação das novas features com SalePrice
new_features = ['TotalSF', 'TotalBath', 'HouseAge', 'RemodAge', 'HasPool', 'HasGarage']
correlations = train[new_features + ['SalePrice']].corr()['SalePrice'].drop('SalePrice').sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['coral' if v < 0 else 'steelblue' for v in correlations.values]
correlations.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Correlação das Novas Features com SalePrice')
ax.set_xlabel('Correlação')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'new_features_correlation.png'), dpi=150)
plt.close()
print(f"   reports/new_features_correlation.png")

# Heatmap das top features
top_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                'FullBath', 'YearBuilt', 'TotalSF', 'TotalBath', 'HouseAge', 'SalePrice']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(train[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Heatmap - Top Features vs SalePrice')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'heatmap_top_features.png'), dpi=150)
plt.close()
print(f"   reports/heatmap_top_features.png")

# ============================================================
# 7. SALVAR DATASET LIMPO
# ============================================================
print(f"\n7. SALVANDO DATASET LIMPO")

train.to_csv(os.path.join(DATA_DIR, "train_clean.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test_clean.csv"), index=False)
print(f"   data/train_clean.csv ({train.shape[0]} x {train.shape[1]})")
print(f"   data/test_clean.csv ({test.shape[0]} x {test.shape[1]})")

# Salvar metadados
metadata = {
    "original_rows": 1460,
    "clean_rows": train.shape[0],
    "columns": train.shape[1],
    "removed_columns": high_missing,
    "outliers_removed": before - after,
    "new_features": new_features,
    "skew_before": round(skew_before, 4),
    "skew_after": round(skew_after, 4),
}
with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   data/metadata.json")

# ============================================================
# 8. RESUMO
# ============================================================
print(f"\n{'=' * 60}")
print(f"RESUMO DA FASE 2")
print(f"{'=' * 60}")
print(f"""
O que fizemos:

1. Missing values:
   - Removidas {len(high_missing)} colunas com >80% missing
   - Preenchidos numéricos com mediana
   - Preenchidos categóricos com 'None'

2. Outliers:
   - Removidos {before - after} outliers (GrLivArea>4000, LotArea>100000, TotalBsmtSF>3000)

3. Log transform:
   - SalePrice: skew {skew_before:.2f} -> {skew_after:.2f} (redução de {(1-abs(skew_after)/abs(skew_before))*100:.1f}%)

4. Feature engineering:
   - TotalSF (área total)
   - TotalBath (banheiros totais)
   - HouseAge (idade da casa)
   - RemodAge (idade da reforma)
   - HasPool, HasGarage (flags binárias)

5. Encoding:
   - Label Encoding para variáveis ordinais (qualidade, condição)
   - One-Hot Encoding para categóricas nominais

6. Dataset final:
   - Train: {train.shape[0]} linhas x {train.shape[1]} colunas
   - Test: {test.shape[0]} linhas x {test.shape[1]} colunas

PROXIMOS PASSOS (Fase 3):
- Treinar modelo baseline (Regressão Linear, Random Forest)
- Comparar métricas (RMSE, R²)
- Feature selection
- Cross-validation
""")
