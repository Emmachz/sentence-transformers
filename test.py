from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
import faiss
import os

# 1. Charger le modèle multilingue
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 2. Extraire le texte des PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text

# 3. Encoder et stocker les embeddings
folder_path = "chemin/vers/tes_pdfs"
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]
embeddings = model.encode(texts, normalize_embeddings=True)

# 4. Créer un index FAISS pour recherche ANN
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = inner product (similaire cosinus si normalisé)
index.add(embeddings)

# 5. Pour chaque document, retrouver les plus proches voisins
k = 5  # nombre de plus proches voisins à retrouver
D, I = index.search(embeddings, k)  # D = scores, I = indices

# 6. Filtrer les doublons avec un seuil
threshold = 0.8
for idx, (scores, neighbors) in enumerate(zip(D, I)):
    for score, neighbor in zip(scores[1:], neighbors[1:]):  # ignorer soi-même (index 0)
        if score > threshold:
            print(f"Doublon: {pdf_files[idx]} <--> {pdf_files[neighbor]} (score: {score:.2f})")
