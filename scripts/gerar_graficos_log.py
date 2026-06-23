import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 4)
plt.rcParams['font.size'] = 12

DATA_DIR = 'data'
REPORTS_DIR = 'reports'

# Carregar dados
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
price = train['SalePrice']
price_log = np.log1p(price)

print('=' * 60)
print('GERANDO GRAFICOS DE COMPARACAO: SEM LOG vs COM LOG')
print('=' * 60)

# 1. Histograma
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(price, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title(f'SalePrice - SEM Log\nSkew: {price.skew():.2f} (assimetria ALTA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Preco (R$)')
axes[0].set_ylabel('Numero de Casas')
axes[0].axvline(price.mean(), color='red', linestyle='--', label=f'Media: R$ {price.mean():,.0f}')
axes[0].axvline(price.median(), color='orange', linestyle='--', label=f'Mediana: R$ {price.median():,.0f}')
axes[0].legend(fontsize=10)

axes[1].hist(price_log, bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_title(f'SalePrice - COM Log\nSkew: {price_log.skew():.2f} (aproximadamente normal)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(Preco)')
axes[1].set_ylabel('Numero de Casas')
axes[1].axvline(price_log.mean(), color='red', linestyle='--', label=f'Media: {price_log.mean():.2f}')
axes[1].axvline(price_log.median(), color='orange', linestyle='--', label=f'Mediana: {price_log.median():.2f}')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'histograma_comparacao.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  reports/histograma_comparacao.png')

# 2. QQ-Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

stats.probplot(price, dist='norm', plot=axes[0])
axes[0].set_title('QQ-Plot: SEM Log\n(Fora da reta = nao normal)', fontsize=13, fontweight='bold')
axes[0].get_lines()[0].set(color='steelblue', markersize=3, alpha=0.6)
axes[0].get_lines()[1].set(color='red', linewidth=2)

stats.probplot(price_log, dist='norm', plot=axes[1])
axes[1].set_title('QQ-Plot: COM Log\n(Proximo da reta = mais normal)', fontsize=13, fontweight='bold')
axes[1].get_lines()[0].set(color='coral', markersize=3, alpha=0.6)
axes[1].get_lines()[1].set(color='red', linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'qqplot_comparacao.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  reports/qqplot_comparacao.png')

# 3. BoxPlot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

bp1 = axes[0].boxplot(price, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[0].set_title('BoxPlot: SEM Log', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Preco (R$)')
n_out_sem = (price > 340000).sum()
axes[0].annotate(f'Outliers: {n_out_sem} casas', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=11, color='red', fontweight='bold')

bp2 = axes[1].boxplot(price_log, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='coral', alpha=0.7))
axes[1].set_title('BoxPlot: COM Log', fontsize=14, fontweight='bold')
axes[1].set_ylabel('log(Preco)')
upper_log = np.log1p(340000)
n_out_com = (price_log > upper_log).sum()
axes[1].annotate(f'Outliers: {n_out_com} casas', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'boxplot_comparacao.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  reports/boxplot_comparacao.png')

# 4. Exemplo numerico visual
example_prices = [50000, 100000, 150000, 200000, 300000, 400000, 500000, 750000]
example_logs = [np.log1p(p) for p in example_prices]

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = range(len(example_prices))

ax.bar([x - 0.2 for x in x_pos], example_prices, width=0.4, color='steelblue', alpha=0.8, label='Preco Real')
ax.bar([x + 0.2 for x in x_pos], [l*50000 for l in example_logs], width=0.4, color='coral', alpha=0.8, label='log(Preco) escalado')

ax.set_xticks(list(x_pos))
ax.set_xticklabels([f'R${p//1000}k' for p in example_prices], fontsize=10)
ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Exemplo Numerico: Preco Real vs Log(Preco)\n(Note como o log comprime os valores grandes)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'exemplo_numerico_log.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  reports/exemplo_numerico_log.png')

print()
print('=' * 60)
print('TODOS OS GRAFICOS GERADOS COM SUCESSO!')
print('=' * 60)
