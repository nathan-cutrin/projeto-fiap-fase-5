[![codecov](https://codecov.io/gh/nathan-cutrin/projeto-fiap-fase-5/branch/main/graph/badge.svg)](https://codecov.io/gh/SEU_USUARIO/SEU_REPO)

# 🎓 Passos Mágicos 

> Datathon FIAP Pós Tech — Machine Learning Engineering (Fase 5)

---

## 1. Visão Geral do Projeto

**Problema de negócio:** agrupar alunos utilizando informações sobre suas avaliações acadêmicas e psicossociais para 
priorizar esforços em grupos de alunos similares.

**Solução proposta:** pipeline completa de Machine Learning — do pré-processamento ao deploy de uma API REST com frontend interativo — que agrupa os alunos em 4 perfis de risco com base em suas dimensões acadêmica e psicossocial.

**Stack tecnológica:**

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.12+ |
| ML | scikit-learn, pandas, numpy |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Serialização | joblib |
| Testes | pytest + pytest-cov |
| Empacotamento | Docker |
| Gerenciador de pacotes | uv |

**Links de produção:**
- API: https://projeto-fiap-fase-5.onrender.com/docs
- Frontend: https://projeto-fiap-fase-5.streamlit.app

---

## 2. Estrutura do Projeto

```
projeto-5/
├── api/                          # Código da API (FastAPI)
│   ├── main.py                   # Entrypoint da aplicação
│   ├── routes.py                 # Endpoints: /predict, /student, /clusters
│   ├── schemas.py                # Modelos Pydantic
│   ├── services/
│   │   └── model_services.py     # Carregamento do modelo e scaler
│   └── database/
│       └── alunos_db.csv         # Base de alunos processada
├── frontend/                     # Interface Streamlit
│   ├── app.py                    # Aplicação principal
│   ├── monitoring.py             # Painel de monitoramento de drift
│   └── utils.py                  # Funções auxiliares
├── src/train/                    # Pipeline de treinamento
│   ├── config.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── main.py
│   └── utils.py
├── scripts/
│   └── data_processing_api.py    # Geração do alunos_db.csv
├── models/                       # Artefatos do modelo
│   ├── model.pkl
│   └── scaler.pkl
├── data/raw/
│   └── base_dados_passos_magicos.xlsx
├── tests/                        # Testes unitários — 89% de cobertura
│   ├── api/
│   ├── frontend/
│   └── scripts/
├── Dockerfile
└── pyproject.toml
```

---

## 3. Instruções de Deploy

**Pré-requisitos:** Python 3.12+ e [uv](https://docs.astral.sh/uv/) instalado.

```bash
# Instalar dependências
uv sync

# Gerar base de alunos processada
uv run python scripts/data_processing_api.py

# Treinar o modelo
uv run python src/train/main.py

# Subir a API (http://localhost:8000)
uv run task api

# Subir o frontend (http://localhost:8501)
uv run task front

# Testes com cobertura
pytest tests/ --cov=. --cov-config=.coveragerc --cov-report=term-missing

# Docker
uv run task docker-build
uv run task docker-run
```

---

## 4. Exemplos de Chamadas à API

### `POST /predict`
```bash
curl -X POST "https://projeto-fiap-fase-5.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"dimensao_academica": 7.5, "dimensao_psicossocial": 8.0}'
```
```json
{ "classe_predita": 2, "metodo": "machine_learning_kmeans" }
```

### `GET /student/{ra}`
```bash
curl "https://projeto-fiap-fase-5.onrender.com/student/904"
```
```json
{
  "ra": "904", "fase": "Fase 3", "idade": 12.0,
  "indicador_desempenho_academico": 7.2, "indicador_engajamento": 8.1,
  "indicador_psicossocial": 6.9, "indicador_autoavaliacao": 7.5,
  "dimensao_academica": 7.65, "dimensao_psicossocial": 7.2
}
```

### `GET /clusters/stats`
```bash
curl "https://projeto-fiap-fase-5.onrender.com/clusters/stats"
```
```json
{
  "1": { "dimensao_academica": { "mean": 5.8, "median": 5.9, "min": 0.0, "max": 7.4 }, "n_alunos": 281 },
  "2": { "...": "..." },
  "3": { "...": "..." },
  "4": { "...": "..." }
}
```

---

## 5. Etapas do Pipeline de Machine Learning

**Pré-processamento:**
- Leitura das abas PEDE 2022, 2023 e 2024 do Excel
- Padronização da coluna `fase` e renomeação dos indicadores (IDA, IEG, IPS, IAA)
- Filtragem para o ano de referência 2024 e remoção de nulos

**Engenharia de Features:**
- `dimensao_academica = média(indicador_desempenho_academico, indicador_engajamento)`
- `dimensao_psicossocial = média(indicador_psicossocial, indicador_autoavaliacao)`

**Treinamento e Validação:**
- Algoritmo: KMeans (k=4) com StandardScaler
- Métrica: Silhouette Score — avalia coesão interna e separação entre clusters
- Artefatos serializados com `joblib`

**Seleção do Modelo:**
- K=4 selecionado via análise do Silhouette Score e interpretabilidade dos perfis gerados

**Monitoramento:**
- Painel interativo no frontend com KS-Test por feature e análise de distribuição de clusters, comparando a linha de base do treino com cenários simulados de produção