import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentVectorSearch:
    def __init__(self):
        # Inicializa o modelo de embedding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.index = None

    def load_documents(self, file_paths):
        # Carrega o texto dos arquivos .txt
        for path in file_paths:
            with open(path, 'r') as f:
                text = f.read().strip()
                self.documents.append(text)
        print(f"Loaded {len(file_paths)} documents")

    def process_documents(self):
        # 1. Transforma os textos em vetores (embeddings)
        embeddings = self.embedder.encode(self.documents)
        embeddings = np.array(embeddings).astype('float32')
        
        # 2. Cria o índice FAISS baseado na dimensão do vetor
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # 3. Adiciona os vetores ao índice
        self.index.add(embeddings)

    def query(self, question, num_results=2):
        # 1. Transforma a pergunta em um vetor numérico
        question_embedding = self.embedder.encode([question])
        question_embedding = np.array(question_embedding).astype('float32')
        
        # 2. Busca os documentos mais similares no FAISS
        distances, indices = self.index.search(question_embedding, num_results)
        
        # 3. Retorna os textos originais correspondentes aos índices
        return [self.documents[i] for i in indices[0]]

# --- Demonstração de uso (Final do arquivo) ---
dvs = DocumentVectorSearch()
dvs.load_documents(['doc1.txt', 'doc2.txt'])

# Processa os documentos ANTES de fazer a busca
dvs.process_documents()

print('Query: "What is Python?"')
# Realiza a busca semântica
results = dvs.query("What is Python?")
print("Top matches:", results)