import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Carregar dados
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_clean.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train_clean.csv')).squeeze()
submission = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

print("Gerando graficos da Fase 3...")

# 1. Feature Importance (Top 15)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top15 = importances.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top15)))
top15.sort_values().plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Top 15 Features - Random Forest', fontsize=16, fontweight='bold')
ax.set_xlabel('Importancia', fontsize=12)
for i, (feat, imp) in enumerate(top15.sort_values().items()):
    ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f03_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  f03_feature_importance.png")

# 2. Model Comparison
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

ridge = Ridge(alpha=10.0, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_scores = np.sqrt(-cross_val_score(ridge, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
rf_scores = np.sqrt(-cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

data = pd.DataFrame({'Ridge': ridge_scores, 'Random Forest': rf_scores})
bp = axes[0].boxplot([data['Ridge'], data['Random Forest']], labels=['Ridge', 'Random Forest'],
                     patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
axes[0].set_title('Comparacao: 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RMSE (log scale)', fontsize=12)
axes[0].grid(True, alpha=0.3)

means = [ridge_scores.mean(), rf_scores.mean()]
stds = [ridge_scores.std(), rf_scores.std()]
bars = axes[1].bar(['Ridge\n(alpha=10)', 'Random Forest\n(100 trees)'], means, 
                   yerr=stds, color=['steelblue', 'coral'], edgecolor='white',
                   capsize=10, alpha=0.8)
axes[1].set_title('RMSE Medio + Desvio Padrao', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RMSE (log scale)', fontsize=12)
for bar, mean, std in zip(bars, means, stds):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                 f'{mean:.4f}', ha='center', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f03_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  f03_model_comparison.png")

# 3. Predictions Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(submission['SalePrice'], bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribuicao das Previsoes (Submission)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Preco (R$)', fontsize=12)
axes[0].set_ylabel('Numero de Casas', fontsize=12)
axes[0].axvline(submission['SalePrice'].mean(), color='red', linestyle='--', 
                label=f'Media: R$ {submission["SalePrice"].mean():,.0f}')
axes[0].axvline(submission['SalePrice'].median(), color='orange', linestyle='--',
                label=f'Mediana: R$ {submission["SalePrice"].median():,.0f}')
axes[0].legend(fontsize=10)

axes[1].hist(np.expm1(y_train), bins=50, color='steelblue', edgecolor='white', alpha=0.8, label='Real')
axes[1].set_title('Distribuicao do Preco Real (Train)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Preco (R$)', fontsize=12)
axes[1].set_ylabel('Numero de Casas', fontsize=12)
axes[1].axvline(np.expm1(y_train).mean(), color='red', linestyle='--',
                label=f'Media: R$ {np.expm1(y_train).mean():,.0f}')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f03_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  f03_predictions.png")

# 4. Top Features vs SalePrice (scatter)
top3 = ['OverallQual', 'TotalSF', 'GrLivArea']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feat in enumerate(top3):
    axes[i].scatter(X_train[feat], np.expm1(y_train), alpha=0.3, s=10, color='steelblue')
    axes[i].set_title(f'{feat} vs SalePrice', fontsize=13, fontweight='bold')
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel('Preco (R$)', fontsize=11)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f03_top_features_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  f03_top_features_scatter.png")

# 5. Residuos do modelo
ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(X_train, y_train)
y_pred_train = ridge.predict(X_train)
residuos = y_train - y_pred_train

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_pred_train, residuos, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Residuos vs Previsao (Ridge)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Previsao (log)', fontsize=12)
axes[0].set_ylabel('Residuo', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuos, bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribuicao dos Residuos (Ridge)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Residuo', fontsize=12)
axes[1].set_ylabel('Frequencia', fontsize=12)
axes[1].axvline(residuos.mean(), color='red', linestyle='--', label=f'Media: {residuos.mean():.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f03_residuals.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  f03_residuals.png")

print("\n" + "="*60)
print("TODOS OS GRAFICOS DA FASE 3 GERADOS!")
print("="*60)
