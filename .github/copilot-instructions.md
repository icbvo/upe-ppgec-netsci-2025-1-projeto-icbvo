# Copilot / AI agent instructions — Projeto GNN Link Prediction

Este repositório implementa um pipeline simples de link prediction em uma
rede de colaboração usando Graph Neural Networks (PyTorch Geometric). Estas
instruções foram escritas para ajudar agentes AI a serem imediatamente
produtivos no código.

Principais pontos (rápido):

- **Entrada de dados:** `data/collaboration.edgelist.txt` — arquivo texto
  whitespace-separated com duas colunas inteiras `u v`. O código assume IDs
  inteiros densos começando em 0 e calcula `num_nodes = max_id + 1`.
- **Script de treino:** `gnn/train_link_prediction_gnn.py` — carregamento,
  splits (train/val/test), GNN encoder, MLP predictor, treinamento e salvamento
  de métricas e modelo em `results/`.
- **Modelos:** `gnn/models.py` — encodeurs disponíveis: `gcn` e `sage` via
  `get_encoder(name=...)`. Edge predictor é um MLP (`LinkPredictor`).
- **Utilitários:** `gnn/utils.py` — funções para seed (`set_seed`), salvar
  métricas, plots e helpers pequenos.
- **Notebooks:** `notebooks/01_exploration.ipynb` mostra como o edge list é
  carregado e limpo (remoção de self-loops, canonicalização `u < v`). Use-o
  para explorações rápidas e verificações de sanity-check.

Arquitetura e fluxo de dados (o "porquê"):

- O pipeline é propositalmente simples e orientado a pesquisa: dados → splits
  → GNN (aprender embeddings) → MLP (predizer arestas). Não há atributos de
  nó externos; características de nó usadas são construídas a partir do grafo
  (grau normalizado, ver `make_node_features` em `train_link_prediction_gnn.py`).
- As arestas são tratadas como undirected para construir a topologia usada na
  agregação do GNN, mas nos passos de classificação as arestas são empilhadas
  em ambas as direções (`undirected=False` para criar `pos_*_edge_index`) —
  isto é importante ao construir batches para o classificador.
- Negative sampling: usa `torch_geometric.utils.negative_sampling` com método
  `sparse`. Validação/test usam amostras negativas fixas; treino amostras
  negativas são re-geradas a cada época.

Convenções específicas do projeto (evitar surpresas):

- Canonicalização de arestas: sempre normalize `u < v` ao manipular o
  arquivo de arestas. Várias partes do código dependem dessa normalização.
- IDs de nós: assumem-se índices densos [0..num_nodes-1]. Se você introduzir
  novos atributos ou fontes externas, mapeie/reescreva IDs para este formato.
- Salvar/ler modelos e métricas: `results/linkpred_gnn_best.pth` e
  `results/linkpred_metrics.json`. Históricos de treino em
  `results/training_history.json`.

Comandos úteis / workflow de desenvolvimento

- Criar ambiente (conda):

  ```bash
  conda env create -f environment.yml
  conda activate <env-name>
  ```

- Ou instalar dependências por pip (gestão local):

  ```bash
  pip install -r requirements_full.txt
  ```

- Treinar localmente (pré-requisito: `data/collaboration.edgelist.txt` presente):

  ```bash
  python gnn/train_link_prediction_gnn.py
  ```

- Rodar via Docker (exemplo usado no repositório):

  ```bash
  docker build -t netsci-gnn-linkpred .
  docker run --rm -v "$(pwd)":/workspace netsci-gnn-linkpred python gnn/train_link_prediction_gnn.py
  ```

Padrões de código e onde encontrar exemplos

- Data loading / limpeza: `notebooks/01_exploration.ipynb` e
  `gnn/train_link_prediction_gnn.py::load_edge_list` — seguem a mesma rotina
  de canonicalização. Copie esse padrão ao adicionar novos loaders.
- Construção de `edge_index` para PyG: função `df_to_edge_index` em
  `train_link_prediction_gnn.py` (usa direção duplicada para tornar
  arestas compatíveis com operações de mensagem bidirecionais).
- Node features: básico — grau normalizado (ver `make_node_features`). Se for
  adicionar features, siga a convenção `Data.x` = tensor float shape `[N, F]`.

Cuidados ao modificar ou estender o código

- Ao alterar a indexação de nós, atualize todas as funções que usam
  `num_nodes = max_id + 1` e as funções que constroem `edge_index`.
- Evite usar IDs arbitrários não-densos sem um mapeamento explícito salvo no
  disco (p.ex. `node_map.json`).
- Testes manuais: use os notebooks para validar transformações de dados e
  `gnn/train_link_prediction_gnn.py` como script de integração rápida.

Onde olhar primeiro (arquivos chave)

- `gnn/train_link_prediction_gnn.py` — script principal de treino e utilitários
- `gnn/models.py` — implementações de encoder (`GCNEncoder`, `GraphSAGEEncoder`) e `LinkPredictor`
- `gnn/utils.py` — helpers (seed, salvar métricas, plots)
- `notebooks/01_exploration.ipynb` — limpeza e sanity-check do edge list
- `data/collaboration.edgelist.txt` — formato de entrada esperado

Seções que podem precisar de colaboração humana / verificações

- Instalação de `torch-geometric` e dependências GPU pode falhar em alguns
  ambientes; confirme o ambiente Python/CUDA correto antes de treinar.
- Se os dados mudarem (ex.: IDs não iniciam em 0), peça ao mantenedor para
  esclarecer como mapear IDs ou adapte o loader e salve o mapeamento.

Se algo não estiver claro, pergunte especificamente sobre:

- formato do arquivo de entrada (IDs densos vs. rótulos arbitrários),
- se deve adicionar atributos de nó externos, ou
- preferências para experiment tracking (p.ex. MLflow, Weights & Biases).

---
Por favor, revise estas instruções e diga se quer que eu acrescente exemplos
de PRs, checks de pre-commit, ou um pequeno script de `run_example.sh`.
