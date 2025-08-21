from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import fitz  # PyMuPDF pour lire les PDF

# 1. Extraire le texte des PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. Charger le modèle
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# Remarque : L12-v2 est plus adapté pour multilingue (fr/en) que L6-v2, mais on peut tester les deux.

# 3. Lister les PDF dans un dossier
folder_path = "chemin/vers/tes_pdfs"
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

# 4. Extraire et encoder les documents
texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]
embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

# 5. Calculer la similarité cosinus entre tous les documents
similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

# 6. Déterminer les doublons (seuil à ajuster selon les tests, ex. 0.8)
threshold = 0.8
duplicates = []
for i in range(len(pdf_files)):
    for j in range(i+1, len(pdf_files)):
        if similarity_matrix[i][j] > threshold:
            duplicates.append((pdf_files[i], pdf_files[j], similarity_matrix[i][j]))

# 7. Afficher les résultats
for dup in duplicates:
    print(f"Doublon trouvé: {dup[0]} <--> {dup[1]} (similarité: {dup[2]:.2f})")
