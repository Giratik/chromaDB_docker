## Phase 2 : Indexation dans ChromaDB

### 2.1 Structure des Documents Stockés

Pour chaque chunk, on stocke :

```json
{
  "id": "pdf__rapport_annuel_2024__p3__c2",
  "document": "Texte du chunk (500 caractères max)...",
  "metadata": {
    "source": "rapport_annuel_2024.pdf",
    "page": 3,
    "chunk_idx": 2,
    "imported_at": "2026-05-27T14:32:00.000Z",
    "doc_date": "2024-12-31"
  }
}
```

### 2.2 Processus d'Indexation

```python
# 1. Extraction PDF en chunks
chunks = pdf_to_chunks(pdf_bytes, filename)

# 2. Génération des IDs stables
existing_ids = collection.get()["ids"]
ids = generate_ids(chunks, existing_ids)

# 3. Ajout à ChromaDB
collection.add(
    ids=ids,
    documents=[c["document"] for c in chunks],
    metadatas=[c["metadata"] for c in chunks]
)
```

### 2.3 Embedding et Stockage

ChromaDB gère automatiquement l'embedding :

```python
embedding_fn = OllamaEmbeddingFunction(
    url="http://ollama:11434/api/embeddings",
    model_name="embeddinggemma"
)

collection = client.get_or_create_collection(
    name="documents_pdf",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # Distance cosinus
)
```

**Détails** :
- **Modèle d'embedding** : `embeddinggemma` (Ollama)
- **Dimension** : 768D
- **Distance** : Cosinus (HNSW index)
- **Indexation** : Automatique lors du `.add()`

---