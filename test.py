from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
model.save("./paraphrase-MiniLM-L6-v2")
