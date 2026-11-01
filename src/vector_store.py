import chromadb


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="document_chunks"
    )

    return collection


def store_chunks(collection, chunks, embeddings, metadatas):
    collection.add(
        ids=[f"chunk_{i}" for i in range(collection.count(), collection.count() + len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def search_chunks(collection, question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count())
    )

    return results