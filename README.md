# House Prices: Advanced Regression Techniques

## Objetivo
Prever o preço de venda de casas com base em 79 variáveis explicativas.
Este projeto faz parte do portfólio de Análise de Dados / ML.

## Dataset
- Fonte: [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- train.csv: 1460 registros, 81 colunas (incluindo SalePrice)
- test.csv: 1459 registros, 80 colunas

## Estrutura
```
data/           - Datasets (train.csv, test.csv)
notebooks/      - Jupyter notebooks por fase
models/         - Modelos treinados
reports/        - Visualizações e relatórios
src/            - Código fonte reutilizável
```

## Fases do Projeto

### Fase 1: Exploração e Limpeza
- [ ] Carregar dataset
- [ ] Análise de valores faltantes
- [ ] Tipos de dados
- [ ] Distribuição do target (SalePrice)
- [ ] Limpeza inicial

### Fase 2: Análise Estatística
- [ ] Estatística descritiva
- [ ] Correlações com SalePrice
- [ ] Análise de outliers
- [ ] Visualizações (histogramas, boxplots, scatter)
- [ ] Feature engineering inicial

### Fase 3: Modelo Baseline
- [ ] Regressão Linear
- [ ] Ridge/Lasso
- [ ] Random Forest
- [ ] XGBoost
- [ ] Comparação de métricas (RMSE, R²)

### Fase 4: Deep Learning com TensorFlow
- [ ] Pré-processamento (normalização, encoding)
- [ ] Arquitetura da rede neural
- [ ] Treinamento e validação
- [ ] Comparação com modelos clássicos

### Fase 5: Documentação
- [ ] README final
- [ ] Notebook limpo e comentado
- [ ] Post para LinkedIn

## Métrica
RMSLE (Root Mean Squared Log Error) — métrica oficial da competição.

## Tecnologias
- Python 3.11
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
