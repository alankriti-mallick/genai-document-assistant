from sentence_transformers import SentenceTransformer


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(model, texts):
    return model.encode(texts)