# 🔍 Sistema de Busca Semântica com IA e FAISS

Este projeto implementa um motor de busca inteligente que compreende o **contexto** das perguntas, indo além da simples busca por palavras-chave. Se um usuário questionar sobre "Python", o sistema localiza a explicação técnica mais relevante, mesmo em textos extensos e complexos.

## 🛠️ Tecnologias Utilizadas

* **Python**: Linguagem principal.
* **FAISS (Facebook AI Similarity Search)**: Biblioteca para busca eficiente em grandes conjuntos de vetores.
* **Sentence-Transformers**: Modelos de Deep Learning para converter texto em embeddings.
* **NumPy**: Processamento de matrizes e vetores.

## 💡 Destaque Técnico

> "Durante o desenvolvimento, implementei a conversão de dados para a tipagem `float32`, garantindo compatibilidade total com o índice **L2 do FAISS**. Isso assegura precisão matemática na busca por similaridade de vetores."

---
## 🚀 Como Executar o Projeto

1. Instale as dependências:
   `pip install -r requirements.txt`

2. Execute o script principal:
   `python main.py`

3. O sistema irá carregar os documentos locais e realizar uma busca semântica baseada na sua pergunta.