# Contributing Guidelines

Este projeto faz parte da disciplina **Network Science (PPGEC/UPE - 2025.1)**.  
Aqui estão as regras para manter a organização e consistência do repositório.

---

## 🔹 Fluxo de Branches
- **main** → branch estável, sempre com versão funcional/documentada.  
- **dev** → branch de integração (features/testes vão primeiro para cá).  
- **feature/** → crie uma branch para cada funcionalidade ou experimento.  
  - Exemplo: `feature/graph-builder`, `feature/gnn-forecasting`.  

---

## 🔹 Mensagens de Commit
Este repositório segue a convenção [Conventional Commits](https://www.conventionalcommits.org/).

Formato: <tipo>[escopo opcional]: <mensagem curta no imperativo>


### Tipos principais:
- **feat:** nova funcionalidade  
  - `feat(graph): implement correlation network builder`
- **fix:** correção de bug  
  - `fix(preprocessing): handle missing values`
- **docs:** documentação  
  - `docs: update README with project objectives`
- **style:** formatação (sem mudar código)  
  - `style(notebook): apply black formatting`
- **refactor:** mudança de código sem alterar comportamento  
  - `refactor(model): simplify GCN layers`
- **test:** adição/ajuste de testes  
  - `test: add unit tests for graph metrics`
- **chore:** tarefas auxiliares  
  - `chore: update .gitignore for data files`

---

upe-ppgec-netsci-2025-1-projeto-icbvo/
├── README.md             # visão geral do projeto
├── CONTRIBUTING.md       # guia de contribuição
├── .gitignore            # arquivos ignorados pelo git
│
├── data/                 # dados brutos (não versionados)
│   └── .gitkeep          # arquivo vazio para manter a pasta no git
│
├── notebooks/            # análises em Jupyter/Colab
│   └── exemplo.ipynb
│
├── src/                  # código-fonte em Python
│   ├── __init__.py
│   └── graph_builder.py
│
└── docs/                 # documentação adicional
    └── projeto.pdf

---

## 🔹 Como contribuir
1. Crie uma branch a partir de `dev`:  
   ```bash
   git checkout dev
   git checkout -b feature/nome-da-feature
2. Faça commits claros e pequenos seguindo a convenção.
3. Abra um Pull Request para dev.
4. Após revisão/testes, a branch será integrada em main.
