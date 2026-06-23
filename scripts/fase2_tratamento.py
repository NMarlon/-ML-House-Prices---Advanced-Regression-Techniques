import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("FASE 2: TRATAMENTO DE DADOS")
print("=" * 60)

# ============================================================
# 1. CARREGAR DADOS ORIGINAIS
# ============================================================
train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"Train: {train_raw.shape[0]} linhas x {train_raw.shape[1]} colunas")
print(f"Test:  {test_raw.shape[0]} linhas x {test_raw.shape[1]} colunas")

# Separar target do train
y_train = train_raw['SalePrice'].copy()
X_train = train_raw.drop('SalePrice', axis=1).copy()

# Guardar IDs do test
test_ids = test_raw['Id'].copy()

# ============================================================
# 2. COMBINAR POR TRATAMENTO CONJUNTO
# ============================================================
combined = pd.concat([X_train, test_raw], axis=0, ignore_index=True)
print(f"Combined: {combined.shape[0]} linhas x {combined.shape[1]} colunas")

# ============================================================
# 3. TRATAR VALORES FALTANTES
# ============================================================
print("\n1. TRATANDO VALORES FALTANTES")

missing = combined.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(combined) * 100).round(1)

high_missing = missing_pct[missing_pct > 80].index.tolist()
print(f"   Removendo: {high_missing}")
combined.drop(columns=high_missing, inplace=True)

num_cols = combined.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if combined[col].isnull().any():
        combined[col] = combined[col].fillna(combined[col].median())

cat_cols = combined.select_dtypes(include=['object']).columns
for col in cat_cols:
    if combined[col].isnull().any():
        combined[col] = combined[col].fillna('None')

print(f"   Missing restantes: {combined.isnull().sum().sum()}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n2. FEATURE ENGINEERING")

combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalBath'] = (combined['FullBath'] + 0.5 * combined['HalfBath'] +
                         combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath'])
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
combined['HasPool'] = (combined['PoolArea'] > 0).astype(int)
combined['HasGarage'] = (combined['GarageArea'] > 0).astype(int)

print("   Criadas: TotalSF, TotalBath, HouseAge, RemodAge, HasPool, HasGarage")

# ============================================================
# 5. ENCODING
# ============================================================
print("\n3. ENCODING")

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
    if col in combined.columns:
        mapping = {val: i for i, val in enumerate(order)}
        combined[col] = combined[col].map(mapping).fillna(0).astype(int)

cat_remaining = combined.select_dtypes(include=['object']).columns.tolist()
print(f"   Label Encoding: {len(ordinal_cols)} colunas")
print(f"   One-Hot Encoding: {len(cat_remaining)} colunas")

combined = pd.get_dummies(combined, columns=cat_remaining, drop_first=True)
print(f"   Dataset apos encoding: {combined.shape}")

# ============================================================
# 6. SEPARAR DE VOLTA (ANTES DE REMOVER OUTLIERS)
# ============================================================
print("\n4. SEPARANDO TRAIN/TEST")

X_train_combined = combined.iloc[:len(X_train)].copy()
X_test_combined = combined.iloc[len(X_train):].copy()

# ============================================================
# 7. LOG TRANSFORM (SO NO TRAIN)
# ============================================================
print("\n5. LOG TRANSFORM")

skew_before = y_train.skew()
y_train = np.log1p(y_train)
skew_after = y_train.skew()

print(f"   Skew antes: {skew_before:.4f}")
print(f"   Skew depois: {skew_after:.4f}")
print(f"   Reducao: {(1 - abs(skew_after)/abs(skew_before))*100:.1f}%")

# ============================================================
# 8. REMOVER OUTLIERS (SO DO TRAIN)
# ============================================================
print("\n6. REMOVENDO OUTLIERS (SO DO TRAIN)")

X_train_combined['SalePrice'] = y_train.values
before = len(X_train_combined)

outlier_rules = [
    ('GrLivArea', 4000),
    ('LotArea', 100000),
    ('TotalBsmtSF', 3000),
]

for col, threshold in outlier_rules:
    mask = X_train_combined[col] > threshold
    count = mask.sum()
    if count > 0:
        print(f"   {col} > {threshold}: {count} removidos")
    X_train_combined = X_train_combined[~mask]

X_train_combined = X_train_combined.reset_index(drop=True)
y_train = X_train_combined['SalePrice'].copy()
X_final = X_train_combined.drop('SalePrice', axis=1).copy()
X_test_final = X_test_combined.copy()

print(f"   Train: {before} -> {len(X_final)} linhas")
print(f"   Test:  {len(X_test_final)} linhas (nao removidas)")

# Verificar colunas identicas
assert list(X_final.columns) == list(X_test_final.columns), "Colunas diferentes!"
print(f"   Colunas identicas: OK ({X_final.shape[1]} colunas)")

# ============================================================
# 9. SALVAR
# ============================================================
print("\n7. SALVANDO DADOS")

X_final.to_csv(os.path.join(DATA_DIR, 'X_train_clean.csv'), index=False)
y_train.to_csv(os.path.join(DATA_DIR, 'y_train_clean.csv'), index=False)
X_test_final.to_csv(os.path.join(DATA_DIR, 'X_test_clean.csv'), index=False)

metadata = {
    'train_rows': len(X_final),
    'test_rows': len(X_test_final),
    'columns': X_final.shape[1],
    'skew_before': round(float(skew_before), 4),
    'skew_after': round(float(skew_after), 4),
    'outliers_removed': before - len(X_final),
    'high_missing_removed': high_missing,
    'new_features': ['TotalSF', 'TotalBath', 'HouseAge', 'RemodAge', 'HasPool', 'HasGarage'],
}
with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   X_train_clean.csv ({X_final.shape})")
print(f"   y_train_clean.csv ({y_train.shape})")
print(f"   X_test_clean.csv ({X_test_final.shape})")
print(f"   metadata.json")

# ============================================================
# 10. VISUALIZACOES
# ============================================================
print("\n8. GERANDO VISUALIZACOES")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(np.expm1(y_train), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title(f'SalePrice Original\nSkew: {skew_before:.2f}')
axes[0].set_xlabel('Preco (R$)')

axes[1].hist(y_train, bins=50, color='coral', edgecolor='white')
axes[1].set_title(f'SalePrice Log\nSkew: {skew_after:.2f}')
axes[1].set_xlabel('log(Preco)')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'saleprice_log_transform.png'), dpi=150)
plt.close()
print("   saleprice_log_transform.png")

print("\n" + "=" * 60)
print("FASE 2 CONCLUIDA!")
print("=" * 60)
