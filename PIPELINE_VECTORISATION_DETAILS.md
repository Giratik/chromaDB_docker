# 📚 Pipeline de Vectorisation et Récupération des Chunks

**Document détaillé** — Explique le flux complet de traitement des PDFs et la stratégie d'interrogation des données.

---

## 📖 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Phase 1 : Vectorisation des PDFs](#phase-1--vectorisation-des-pdfs)
3. [Phase 2 : Indexation dans ChromaDB](#phase-2--indexation-dans-chromadb)
4. [Phase 3 : Interrogation des Données](#phase-3--interrogation-des-données)
5. [Architecture Générale](#architecture-générale)
6. [Optimisations et Considérations](#optimisations-et-considérations)

---

## Vue d'ensemble

Le système suit une architecture **RAG (Retrieval-Augmented Generation)** avec recherche hybride :

```
PDF (fichier brut)
    ↓
[EXTRACTION] Texte natif + OCR
    ↓
[CHUNKING] Découpe en fragments
    ↓
[INDEXATION] ChromaDB + Embeddings
    ↓
[BASE VECTORIELLE] Stockage persistant
    ↓
[REQUÊTE] Query Augmentation + Hybrid Search
    ↓
[RÉCUPÉRATION] Chunks pertinents + Métadonnées
    ↓
[GÉNÉRATION] LLM avec contexte
```

---

## Phase 1 : Vectorisation des PDFs

### 1.1 Extraction du Texte (`pdf_engine.py`)

L'extraction du texte suit une stratégie progressive avec fallback :

#### **Étape 1a : Extraction Texte Natif**

```python
text = page.extract_text()  # Via pdfplumber
```

- **Avantage** : Rapide (< 10ms par page)
- **Cas d'usage** : PDFs créés numériquement avec texte embarqué
- **Résultat** : Texte précis et bien structuré

**Seuil de validation** : `MIN_NATIVE_CHARS = 30` caractères
- Si le texte natif < 30 caractères → la page est considérée comme problématique
- Bascule vers OCR

---

#### **Étape 1b : OCR Intelligent (Deux niveaux de fallback)**

Quand le texte natif est insuffisant, le système tente :

**Priorité 1 : PaddleOCR** (si disponible)

```python
_paddle = PaddleOCR(
    use_angle_cls=True,    # Détecte et corrige l'orientation automatique
    lang="fr",             # Français
    use_gpu=True,          # Accélération CUDA
    show_log=False
)
```

**Caractéristiques** :
- ✅ Correction d'orientation **automatique** (0°, 90°, 180°, 270°)
- ✅ Accélération GPU (CUDA) → ~5-20ms par page
- ✅ Multilangue (100+ langues dont FR/EN)
- ✅ Filtrage confiance : `confidence >= 0.5`
- 🎯 **Préféré pour pages scannées mal orientées**

**Priorité 2 : Tesseract** (fallback)

```python
pytesseract.image_to_osd(img)  # Orientation
pytesseract.image_to_string(img, lang="fra+eng")
```

**Caractéristiques** :
- Détection d'orientation (OSD)
- Correction automatique de rotation
- Plus lent que PaddleOCR (~50-100ms)
- Disponibilité garantie (système standard)

---

#### **Processus d'Extraction Complet**

```
Pour chaque page PDF :

┌─────────────────────────────────────
│ Essayer extraction texte natif
└─────────────────────────────────────
                ↓
        Longueur >= 30 car ?
          YES ↓         ↗ NO
              │    
        Retourner le texte
              │
              └─→ RASTERISER page (200 DPI)
                      ↓
                  PaddleOCR dispo ?
                  YES ↓      ↗ NO
                      │
                  Appliquer PaddleOCR
                  (+ correction angle)
                      ↓
                    Résultat > natif ?
                    YES ↓    ↗ NO
                        │
                    Utiliser OCR
                        │
                        └─→ Tesseract fallback
                            └─→ Retourner meilleur
```

---

### 1.2 Nettoyage du Texte

Après extraction, nettoyage standard :

```python
# Suppression espaces multiples
text = re.sub(r'\n{3,}', '\n\n', raw)
text = re.sub(r' {2,}', ' ', text).strip()
```

- Réduit le bruit
- Normalise la structure
- Prépare au chunking

---

### 1.3 Chunking avec Overlap (`_chunk_text`)

Paramètres par défaut :
- **Taille chunk** : 500 caractères
- **Overlap** : 100 caractères
- **Min chunk** : 20 caractères (filtrage)

#### **Stratégie de Coupure (Priorité)**

La coupure respecte les frontières naturelles pour garder le contexte :

```
PRIORITÉ 1 : Paragraphe
  ↓ rfind('\n\n') dans les 50% du chunk
  └─ Coupure après '\n\n'

PRIORITÉ 2 : Phrase
  ↓ rfind(['. ', '! ', '? ', '.\n']) dans les 50% du chunk
  └─ Coupure après la ponctuation

PRIORITÉ 3 : Mot
  ↓ rfind(' ') n'importe où
  └─ Coupure après l'espace
```

**Exemple** :

```
Texte original (600 caractères) :
"Le machine learning est une branche... [+ 400 car] ... paragraphe suivant."

Chunk 1 (500 car) : "Le machine learning... avec overlap"
  └─ Coupure détectée à ponctuation (. )
  
Chunk 2 (400 car) : "avec overlap + nouvel extrait..."
  └─ Overlap = 100 caractères = contexte partagé
```

**Avantages du overlapping** :
- ✅ Évite la perte de contexte aux frontières
- ✅ Améliore la retrouvabilité (requête peut matcher à la jonction)
- ✅ Réduit les biais de chunking

---

### 1.4 Génération des IDs Stables

Chaque chunk reçoit un ID **unique et stable** :

```python
ID_FORMAT = f"pdf__{slug}__p{page}__c{chunk_idx}"

Exemple : pdf__rapport_annuel_2024__p3__c2
```

**Composition** :
- `pdf__` : Préfixe identifiant le type (PDF vs Lexique)
- `{slug}` : Nom du fichier (alphanumérique, max 40 car)
- `p{page}` : Numéro de page
- `c{chunk_idx}` : Index du chunk sur cette page

**Avantages** :
- ✅ IDs prévisibles → mise à jour sans collision
- ✅ Traçabilité complète (retrouver source + contexte)
- ✅ Gestion des doublons (suffixe `_0`, `_1` si collision)

---

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

## Phase 3 : Interrogation des Données

### 3.1 Vue d'ensemble du Flux de Requête

```
USER: "Quelles étaient les revenus en 2024 ?"
    ↓
[QUERY AUGMENTATION]
  ├─ Expansion : générer variantes
  ├─ HyDE : générer réponse hypothétique
  └─ Résultat : 3-4 variantes de requête
    ↓
[RECHERCHE VECTORIELLE] ChromaDB
  ├─ Embedding requête
  ├─ Recherche KNN (top-50)
  └─ Distance cosinus calculée
    ↓
[RECHERCHE LEXICALE] BM25
  ├─ Tokenization requête
  ├─ Corpus indexé
  └─ Scores BM25 calculés
    ↓
[FUSION HYBRIDE]
  ├─ Normalisation vectorielle
  ├─ Normalisation BM25
  ├─ Alpha blend (70/30 ou custom)
  └─ Top-100 hybrid scores
    ↓
[RERANKING] BGE Local
  ├─ Modèle local (bge-reranker-v2-gemma)
  ├─ Relevance scoring
  └─ Top-10 final
    ↓
[CONTEXTE] Formatage
  ├─ Métadonnées sources
  ├─ Dates documents
  └─ Résultat → LLM
```

---

### 3.2 Query Augmentation

#### **A. Expansion (`expand_query`)**

Génère 3 reformulations avec synonymes :

```
Query: "Quelles étaient les revenus en 2024 ?"

Variantes générées par LLM:
1. "Quels ont été les gains financiers de 2024 ?"
2. "Informations chiffre d'affaires 2024"
3. "Bénéfices exercice 2024"

Total requêtes : 1 (original) + 3 (variantes) = 4
```

**Bénéfices** :
- ✅ Capture variantes de vocabulaire
- ✅ Réduction synonyme "gain" vs "revenu"
- ✅ Meilleure couverture sémantique

#### **B. HyDE (Hypothetical Document Embedding)**

Génère une réponse hypothétique supposée être dans le corpus :

```
Query: "Quelles étaient les revenus en 2024 ?"

Réponse hypothétique générée :
"En 2024, les revenus de l'entreprise se sont élevés à 
1,2 milliards d'euros, en augmentation de 8% par rapport 
à 2023. Ce chiffre inclut les ventes nationales (700M€) 
et l'export (500M€)."

Utilisation : Embedding de cette réponse 
           → Améliore la recherche vectorielle
```

**Raison** :
- Les réponses hypothétiques sont souvent plus similaires aux chunks que la requête brute
- Augmente la probabilité de retrouver les vrais chunks pertinents

---

### 3.3 Recherche Vectorielle (ChromaDB)

```python
# Requête multi-query
results = collection.query(
    query_texts=[q1, q2, q3, q4],  # 4 requêtes (orig + 3 variantes + HyDE)
    n_results=5  # Top-5 par requête
)

# Résultat : Jusqu'à 20 chunks candidates (en éliminant doublons)
```

**Résultat pour chaque chunk** :
- Document texte
- Distance cosinus [0, 2]
- Métadonnées complet

**Conversion distance → score** :
```
vecto_score = 1 - (distance / 2)

Distance 0.2  →  Score 0.90 ✅ Très proche
Distance 1.0  →  Score 0.50 ✅ Moyen
Distance 1.8  →  Score 0.10 ⚠️  Loin
```

---

### 3.4 Recherche Lexicale (BM25)

**BM25 = Ranking algorithm lexical** (Okapi)

```python
# Tokenization
tokens_query = tokenize("revenus 2024")
# → ["revenus", "2024"]

tokens_corpus = [tokenize(doc) for doc in candidates]
# → [["revenus", "compte", "annuel", ...], ...]

# BM25 ranking
bm25 = BM25Okapi(tokens_corpus)
bm25_scores = bm25.get_scores(tokens_query)
# → [0.45, 0.78, 0.12, ...]  Scores pour chaque doc
```

**Avantages BM25** :
- ✅ Indépendant des embeddings
- ✅ Excelle sur requêtes simples ("2024", "revenue")
- ✅ Rapidité (aucun réseau/GPU)
- ✅ Complémenter recherche vectorielle

---

### 3.5 Fusion Hybride

**Alpha blending** = Combinaison vectoriel + BM25

```python
# Normalisation des scores [0-1]
vecto_scores_norm = [s / max(vecto_scores) for s in vecto_scores]
bm25_scores_norm = [s / max(bm25_scores) for s in bm25_scores]

# Alpha = poids vectoriel (défaut 0.7)
hybrid_score = 0.7 * vecto_norm + 0.3 * bm25_norm

Interprétation:
- 0.7 : 70% confiance à l'embedding (sémantique profonde)
- 0.3 : 30% confiance au BM25 (mots clés)
```

**Avantages** :
- ✅ Robustesse : bénéficie des 2 approches
- ✅ Requêtes simples → BM25 boost
- ✅ Requêtes sémantiques → Vectoriel domine
- ✅ Configurable (alpha = paramètre)

---

### 3.6 Reranking (BGE Local)

Utilise le modèle local **bge-reranker-v2-gemma** pour score de relevance :

```python
# Prompt structuré
prompt = f"""Given a query and a passage, predict whether 
the passage is relevant to the query.
Query: {query}
Passage: {document_excerpt}
Relevant (Yes/No):"""

# Inference
score = model.logit("Yes") / (logit("Yes") + logit("No"))
# → Score [0, 1] : probabilité relevance
```

**Stratégie de reranking** :

```
Avant reranking : 100 candidats (top hybrid)
  ↓
Batch inference : 4 documents / batch (GPU efficiency)
  ↓
Score relevance : 0.92, 0.87, 0.65, 0.34, ...
  ↓
Tri décroissant : Meilleurs d'abord
  ↓
Top-10 final : Ressortie réelle (meilleure qualité)
```

**Résultat** :
- Élimine les faux positifs (score bas)
- Priorise la vrai relevance sémantique
- Latence acceptable (~50-500ms pour 10-50 docs)

---

### 3.7 Construction du Contexte Final

Pour chaque chunk gardé (top-10) :

```python
context_line = f"Extrait de 📄 {filename} (Page {page})"
if doc_date:
    context_line += f" [Document du {doc_date}]"
context_line += f":\n{document_text}"
```

**Résultat = Liste de contextes annotés** :

```
Extrait de 📄 rapport_annuel_2024.pdf (Page 3) 
[Document du 2024-12-31] :
"Les revenus en 2024 se sont élevés à 1,2 milliards 
d'euros, en augmentation de 8% par rapport à 2023."

Extrait de 📄 bilan_trimestriel_q4_2024.pdf (Page 1) 
[Document du 2024-12-31] :
"Le trimestre Q4 a enregistré une croissance de revenus 
de 12%, porté par..."

[Continuer pour top-10]
```

---

## Architecture Générale

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTÈME RAG COMPLET                          │
└─────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────┐
        │          UPLOAD & INDEXATION (One-time)               │
        └──────────────────────────────────────────────────────┘
               PDF File
                  ↓
          pdf_engine.py
          ├─ extract_pages()      → Texte + OCR
          ├─ _chunk_text()        → Chunks overlap
          └─ generate_ids()       → IDs stables
                  ↓
          ChromaDB Collection
          ├─ Document text
          ├─ Metadata (source, page, date)
          └─ Embeddings (embeddinggemma)
                  ↓
          HNSW Index (cosine)
          
        ┌──────────────────────────────────────────────────────┐
        │          QUERY & RETRIEVAL (Per request)              │
        └──────────────────────────────────────────────────────┘
               User Query
                  ↓
          Query Augmentation
          ├─ expand_query()    → Variantes
          └─ hyde_query()      → Réponse hypothétique
                  ↓
          Dual Search
          ├─ Vectorial (ChromaDB) → Distance cosinus
          └─ Lexical (BM25)       → Scores BM25
                  ↓
          Hybrid Fusion
          └─ alpha_blend()        → 0.7V + 0.3B
                  ↓
          Reranking (Local BGE)
          └─ bge-reranker-v2-gemma → Relevance score
                  ↓
          Top-10 Chunks
          ├─ Texte
          ├─ Source + Page
          ├─ Date document
          └─ Scores (V|B|H|R)
                  ↓
          Context String
          └─ Formaté pour LLM
                  ↓
          LLM Generation (Ollama)
          └─ Streaming response
```

---

## Optimisations et Considérations

### Performance

| Opération | Temps | Notes |
|-----------|-------|-------|
| Extraction texte natif | ~10ms/page | Très rapide |
| OCR PaddleOCR | ~5-20ms/page | GPU CUDA |
| OCR Tesseract | ~50-100ms/page | CPU only |
| Chunking | ~1ms/100 chunks | Rapide |
| Vectorisation (embedding) | ~100-200ms/doc | Batch Ollama |
| Recherche vectorielle | ~50-100ms | HNSW index |
| BM25 | ~10-30ms | Lexical only |
| Reranking BGE | ~50-500ms/10-50docs | GPU optional |
| **Total requête** | **~500-1000ms** | Tout compris |

### Qualité

| Facteur | Impact |
|---------|--------|
| Chunk size | Plus grand = moins de chunks / mieux contexte, mais moins précis |
| Overlap | Plus d'overlap = plus redondance, mais meilleure couverture |
| Alpha | 0.7+ = préfère sémantique; 0.3+ = préfère lexical |
| Query augmentation | Augmente couverture au prix de latence |
| Reranking | Améliore drastiquement qualité (~20-30% boost) |

### Limites Connues

⚠️ **Pages extrêmement mal scannées** (flou, rotation extrême)
- Peut nécessiter retraitement manuel ou meilleur OCR

⚠️ **PDFs avec images/tableaux**
- Texte extrait, images perdues
- Solution : OCR sur raster

⚠️ **Très longs documents**
- Fragmentation en 1000+ chunks
- Peut surcharger le reranker
- Mitigation : Top-50 avant reranking

⚠️ **Requêtes très longues ou complexes**
- Query augmentation peut générer variantes ambiguës
- Mitigation : Limiter temperature LLM

---

## Résumé

**Pipeline résumé** :

1. **PDF → Texte** : Extraction native + OCR fallback
2. **Texte → Chunks** : Coupure intelligente avec overlap
3. **Chunks → Embeddings** : Ollama + HNSW storage
4. **Requête → Augmentation** : Variantes + HyDE
5. **Recherche Dual** : Vectoriel + BM25
6. **Fusion Hybride** : Alpha blend
7. **Reranking** : BGE local
8. **Génération** : LLM avec contexte

**Résultat** : Système RAG robuste avec recherche haute qualité, tolérant aux PDFs mal formés et optimisé pour les requêtes en français.

