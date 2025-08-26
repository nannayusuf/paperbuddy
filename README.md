# PaperBuddy

Assistente para ler e responder perguntas sobre PDFs usando RAG + CrewAI.

## Requisitos

* Python gerenciado pelo **uv** (3.12 recomendado)
* **OPENAI\_API\_KEY** (para respostas do agente)

## Quickstart

```bash
# 1) Ambiente
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync

# 2) Variáveis de ambiente (opcional via arquivo)
cp .env.example .env   # edite sua chave
export $(grep -v '^#' .env | xargs -d '\n')
# ou exporte direto:
# export OPENAI_API_KEY="sk-..."
# export OPENAI_MODEL="gpt-4o-mini"

# 3) Garantir que o Python enxerga o pacote
export PYTHONPATH="$PWD/src"
# (opcional) tornar permanente:
echo 'export PYTHONPATH="$PWD/src:$PYTHONPATH"' >> .venv/bin/activate

# 4) PDFs e indexação
mkdir -p data/pdfs
# coloque seus PDFs em data/pdfs
ppython -m paperbuddy.rag.index --reset

# 5) Rodar UI (Streamlit)
uv run streamlit run app.py
```

## Offline test (sem OPENAI\_API\_KEY)

Valida que a indexação e a busca vetorial funcionam sem LLM.

```bash
# venv e deps
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync

# pacote no path
export PYTHONPATH="$PWD/src"

# preparar dados
mkdir -p data/pdfs data/chroma
# se não tiver PDFs, crie um dummy rápido:
if ! ls data/pdfs/*.pdf >/dev/null 2>&1; then
  printf '%s\n' '%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 55>>stream
BT /F1 12 Tf 72 120 Td (Hello PaperBuddy) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R>>%%EOF' > data/pdfs/dummy.pdf
fi

# indexar
python -m paperbuddy.rag.index --reset

# buscar
python - <<'PY'
from paperbuddy.tools.vector_search import search
res = search("hello", k=2)
print("[ok] resultados:", len(res))
# mostra o primeiro score já normalizado em 0..1
if res:
    print("score:", res[0][1])
    print("preview:", res[0][0].page_content[:200].replace("\\n"," "))
PY
```

## Estrutura

```
paperbuddy/
├─ app.py
├─ pyproject.toml
├─ data/
│  ├─ pdfs/          # seus PDFs (não versionar)
│  └─ chroma/        # índice vetorial (não versionar)
└─ src/
   └─ paperbuddy/
      ├─ __init__.py
      ├─ crew.py                  # agentes PT-BR (make_crew)
      ├─ rag/
      │  ├─ __init__.py
      │  └─ index.py              # builder do índice
      └─ tools/
         ├─ __init__.py
         ├─ vector_search.py      # busca (scores normalizados 0..1)
         └─ vector_search_tool.py # tool p/ CrewAI
```

## Notas

* Requer **OPENAI\_API\_KEY** para o agente responder na UI. Sem a key, use o *Offline test* para validar o pipeline.
* Se aparecer `ModuleNotFoundError: paperbuddy`, exporte `PYTHONPATH="$PWD/src"`.
* Se faltar dependência, rode `uv sync` (ou `uv add <pacote>` + `uv sync`).

## .env.example

```env
OPENAI_API_KEY=coloque_sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini
```

## .gitignore sugerido

```gitignore
.venv/
__pycache__/
*.pyc
.env
.envrc
uv_sync.log
data/chroma/
data/pdfs/*.pdf
```
