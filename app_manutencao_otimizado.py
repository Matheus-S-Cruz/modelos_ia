# ==============================================================================
# PROJETO: MANUTENÇÃO PREDITIVA - COMPARATIVO MULTI-MODELOS (MLOps) - OTIMIZADO
# Arquivo: app_manutencao_otimizado.py
#
# Este script implementa as melhorias da Parte 2 da atividade:
#   - Tarefa 2: Engenharia de Features Avançada (variáveis temporais de
#               tendência de degradação).
#   - Tarefa 3: Balanceamento de Dados com SMOTE (aplicado SOMENTE no treino).
#   - Tarefa 4: Análise de Custo Financeiro Industrial (matriz de custo).
#
# Contexto de negócio (fábrica TechIndustries):
#   - Falha Perdida  (Falso Negativo / FN) custa R$ 15.000,00 por ocorrência.
#   - Alarme Falso   (Falso Positivo / FP) custa R$    500,00 por ocorrência.
# Objetivo: encontrar o modelo com o menor custo financeiro total nos 3
# últimos meses do ano.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Tarefa 3: biblioteca para o balanceamento de classes com SMOTE
from imblearn.over_sampling import SMOTE


# ------------------------------------------------------------------------------
# PARÂMETROS DE NEGÓCIO (Tarefa 4 - Matriz de Custo Financeiro)
# ------------------------------------------------------------------------------
CUSTO_FALHA_PERDIDA = 15_000  # Custo de um Falso Negativo (FN) em R$
CUSTO_ALARME_FALSO = 500      # Custo de um Falso Positivo (FP) em R$


# ------------------------------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS (1 ANO / 8.760 HORAS)
# ------------------------------------------------------------------------------
print("--- [1/7] Gerando base de dados sintética de 1 ano (8.760 horas) ---")
np.random.seed(42)
datas = pd.date_range(
    start="2025-01-01 00:00:00",
    end="2025-12-31 23:00:00",
    freq="h",
)

n_amostras = len(datas)

# Variáveis de sensores da máquina
temperatura = np.random.normal(loc=65.0, scale=5.0, size=n_amostras)
vibracao = np.random.normal(loc=2.5, scale=0.4, size=n_amostras)
pressao = np.random.normal(loc=101.3, scale=2.0, size=n_amostras)

df_raw = pd.DataFrame(
    {
        "timestamp": datas,
        "temperatura": temperatura,
        "vibracao": vibracao,
        "pressao": pressao,
    }
)

# Injeção de regras operacionais de falha e ruído aleatório
condicao_falha = (df_raw["temperatura"] > 75) & (df_raw["vibracao"] > 3.2)
ruido = np.random.choice([0, 1], size=n_amostras, p=[0.98, 0.02])
df_raw["falha"] = (condicao_falha | (ruido == 1)).astype(int)
df_raw.to_csv("telemetria_maquina_2025.csv", index=False)
print("-> Base bruta salva em 'telemetria_maquina_2025.csv'")


# ------------------------------------------------------------------------------
# 2. ENGENHARIA DE FEATURES AVANÇADA (Tarefa 2)
# ------------------------------------------------------------------------------
# Além das features base (médias móveis de 6h, desvios-padrão de 24h e a
# diferença de temperatura), criamos novas variáveis temporais que capturam
# TENDÊNCIAS DE DEGRADAÇÃO da máquina.
# ------------------------------------------------------------------------------
print("\n--- [2/7] Executando Engenharia de Features Avançada ---")
df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

# --- Features base (já existentes na versão original) ---
for col in ["temperatura", "vibracao", "pressao"]:
    df_raw[f"{col}_media_6h"] = df_raw[col].rolling(window=6).mean()
    df_raw[f"{col}_std_24h"] = df_raw[col].rolling(window=24).std()

df_raw["diff_temp"] = df_raw["temperatura"].diff()

# --- NOVAS features (Tarefa 2) ---

# a) Taxa de variação (diferença percentual) para vibração e pressão.
#    Captura a velocidade com que a leitura do sensor está mudando.
df_raw["taxa_var_vibracao"] = df_raw["vibracao"].pct_change()
df_raw["taxa_var_pressao"] = df_raw["pressao"].pct_change()

# b) Janelas móveis de máximo e mínimo de 12 horas para temperatura e vibração.
#    Captura os picos/vales recentes (indícios de estresse mecânico).
df_raw["temperatura_max_12h"] = df_raw["temperatura"].rolling(window=12).max()
df_raw["temperatura_min_12h"] = df_raw["temperatura"].rolling(window=12).min()
df_raw["vibracao_max_12h"] = df_raw["vibracao"].rolling(window=12).max()
df_raw["vibracao_min_12h"] = df_raw["vibracao"].rolling(window=12).min()

# c) Feature de interação: temperatura * vibração.
#    A falha ocorre justamente quando temperatura E vibração estão altas juntas.
df_raw["temp_x_vibracao"] = df_raw["temperatura"] * df_raw["vibracao"]

# pct_change pode gerar valores infinitos caso o denominador seja ~0; tratamos
# como ausentes para que o dropna os remova junto com os NaN das janelas móveis.
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
df_preprocessed = df_raw.dropna().reset_index(drop=True)
print(f"-> {df_preprocessed.shape[1] - 2} features disponíveis após a engenharia.")


# ------------------------------------------------------------------------------
# 3. DIVISÃO TEMPORAL (TREINO VS. TESTE)
# ------------------------------------------------------------------------------
print("\n--- [3/7] Divisão Temporal dos Dados ---")
data_corte = pd.to_datetime("2025-10-01 00:00:00")
df_treino = df_preprocessed[df_preprocessed["timestamp"] < data_corte].copy()
df_teste = df_preprocessed[df_preprocessed["timestamp"] >= data_corte].copy()
df_teste.to_csv("dados_teste_ultimos_3_meses.csv", index=False)
print("-> Base de teste (3 últimos meses) salva em 'dados_teste_ultimos_3_meses.csv'")

features = [col for col in df_preprocessed.columns if col not in ["timestamp", "falha"]]
target = "falha"
X_treino, y_treino = df_treino[features].copy(), df_treino[target].copy()
X_teste, y_teste = df_teste[features].copy(), df_teste[target].copy()


# ------------------------------------------------------------------------------
# 4. BALANCEAMENTO DE DADOS COM SMOTE (Tarefa 3)
# ------------------------------------------------------------------------------
# O SMOTE gera amostras sintéticas da classe minoritária (falhas).
# ATENÇÃO MLOps: o SMOTE é aplicado APENAS na base de treino. Aplicá-lo na base
# de teste causaria DATA LEAKAGE e invalidaria a avaliação do modelo.
# ------------------------------------------------------------------------------
print("\n--- [4/7] Aplicando SMOTE apenas na base de TREINO ---")
print(f"-> Distribuição ANTES do SMOTE (treino): {dict(y_treino.value_counts().sort_index())}")

smote = SMOTE(random_state=42)
X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)

print(f"-> Distribuição DEPOIS do SMOTE (treino): {dict(pd.Series(y_treino_bal).value_counts().sort_index())}")

# Normalização de dados (Escalonamento exigido para a Regressão Logística).
# O scaler é ajustado SOMENTE no treino balanceado e reaplicado ao teste.
scaler = StandardScaler()
X_treino_bal_scaled = scaler.fit_transform(X_treino_bal)
X_teste_scaled = scaler.transform(X_teste)


# ------------------------------------------------------------------------------
# 5. TREINAMENTO DOS MODELOS (sobre a base reamostrada pelo SMOTE)
# ------------------------------------------------------------------------------
# Como o SMOTE já equilibra as classes no treino, os parâmetros de ponderação
# de classe (scale_pos_weight / class_weight) não são mais necessários.
# ------------------------------------------------------------------------------
print("\n--- [5/7] Treinando os 5 Modelos sobre a base balanceada (SMOTE) ---")
modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=4,
        random_seed=42,
        verbose=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        verbosity=-1,
    ),
    "Regressão Logística": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
}

predicoes = {}
for nome, model in modelos.items():
    if nome == "Regressão Logística":
        model.fit(X_treino_bal_scaled, y_treino_bal)
        predicoes[nome] = model.predict(X_teste_scaled)
    else:
        model.fit(X_treino_bal, y_treino_bal)
        predicoes[nome] = model.predict(X_teste)

    print(f"-> Modelo: {nome} treinado com sucesso!")


# ------------------------------------------------------------------------------
# 6. TABELA COMPARATIVA DE DESEMPENHO + CUSTO FINANCEIRO (Tarefa 4)
# ------------------------------------------------------------------------------
print("\n--- [6/7] Tabela Comparativa de Métricas (Avaliação nos 3 Últimos Meses) ---")
resumo = []
resumo_custo = []
for nome, y_pred in predicoes.items():
    cm = confusion_matrix(y_teste, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rec = recall_score(y_teste, y_pred, zero_division=0)
    prec = precision_score(y_teste, y_pred, zero_division=0)
    f1 = f1_score(y_teste, y_pred, zero_division=0)
    acc = accuracy_score(y_teste, y_pred)

    # Tarefa 4: Custo Financeiro Total de Erro
    # Custo Total (R$) = (FN * R$ 15.000) + (FP * R$ 500)
    custo_total = (fn * CUSTO_FALHA_PERDIDA) + (fp * CUSTO_ALARME_FALSO)

    resumo.append(
        {
            "Modelo": nome,
            "Recall (Sensibilidade)": f"{rec * 100:.2f}%",
            "Precisão": f"{prec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "Acurácia": f"{acc * 100:.2f}%",
            "Alarmes Falsos (FP)": fp,
            "FALHAS PERDIDAS (FN)": fn,
            "Falhas Detectadas (TP)": tp,
        }
    )

    resumo_custo.append(
        {
            "Modelo": nome,
            "Recall": f"{rec * 100:.2f}%",
            "Precisão": f"{prec * 100:.2f}%",
            "Alarmes Falsos (FP)": fp,
            "Falhas Perdidas (FN)": fn,
            "Custo Financeiro Total (R$)": custo_total,
        }
    )

df_resumo = pd.DataFrame(resumo)
print("\n", df_resumo.to_string(index=False))

# --- Tabela de Custo Financeiro (Tarefa 4) ---
print("\n\n--- TABELA DE CUSTO FINANCEIRO INDUSTRIAL (Tarefa 4) ---")
print(f"(FN = R$ {CUSTO_FALHA_PERDIDA:,.0f} por falha perdida | "
      f"FP = R$ {CUSTO_ALARME_FALSO:,.0f} por alarme falso)".replace(",", "."))

df_custo = pd.DataFrame(resumo_custo).sort_values(
    "Custo Financeiro Total (R$)"
).reset_index(drop=True)

# Formatação monetária em padrão brasileiro para exibição
df_custo_exibicao = df_custo.copy()
df_custo_exibicao["Custo Financeiro Total (R$)"] = df_custo_exibicao[
    "Custo Financeiro Total (R$)"
].apply(lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

print("\n", df_custo_exibicao.to_string(index=False))

melhor = df_custo.iloc[0]
print(
    f"\n>>> MENOR CUSTO FINANCEIRO: {melhor['Modelo']} "
    f"(R$ {melhor['Custo Financeiro Total (R$)']:,.2f})".replace(",", "X").replace(".", ",").replace("X", ".")
)


# ------------------------------------------------------------------------------
# 7. VISUALIZAÇÃO GRÁFICA COMPARATIVA (GRID 2x3 DE MATRIZES)
# ------------------------------------------------------------------------------
print("\n--- [7/7] Exibindo e Salvando Painel Comparativo de Matrizes de Confusão ---")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()

for idx, (nome, y_pred) in enumerate(predicoes.items()):
    cm = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal (0)", "Falha (1)"],
    )
    disp.plot(cmap="Blues", values_format="d", ax=axes[idx])
    axes[idx].set_title(f"{nome}", fontsize=12, fontweight="bold")

# Ocultar o 6º submódulo do grid que ficou vazio
fig.delaxes(axes[5])
plt.suptitle(
    "Comparativo de Matrizes de Confusão nos Últimos 3 Meses\n"
    "(Features Avançadas + SMOTE)",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
plt.savefig("matriz_comparativa_otimizada.png", bbox_inches="tight")
print("-> Imagem 'matriz_comparativa_otimizada.png' salva com sucesso!")
plt.show()
