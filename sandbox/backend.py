"""
backend.py — Logique RAG : ChromaDB, Ollama, BM25, Hybrid Search
─────────────────────────────────────────────────────────────────
Aucune dépendance Streamlit ici. Importable indépendamment.

Prérequis :
    pip install chromadb ollama rank-bm25
"""

import re
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client

from config import CHROMA_HOST, CHROMA_PORT, OLLAMA_HOST, EMBEDDING_MODEL


# ─── CLIENTS ──────────────────────────────────────────────────────────────────

def make_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def make_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


def make_embedding_fn() -> embedding_functions.OllamaEmbeddingFunction:
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST + "/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )


def get_collection(chroma_client: chromadb.HttpClient, collection_name: str):
    return chroma_client.get_collection(
        name=collection_name,
        embedding_function=make_embedding_fn(),
    )


def list_collections(chroma_client: chromadb.HttpClient) -> list[str]:
    raw = chroma_client.list_collections()
    return [c.name if hasattr(c, "name") else c for c in raw]


def list_doc_dates(collection) -> list[str]:
    result = collection.get(include=["metadatas"])
    dates = {
        meta.get("doc_date", "")
        for meta in (result.get("metadatas") or [])
        if meta and meta.get("doc_date")
    }
    return sorted(dates)


def list_generative_models(ollama_client: Client) -> list[str]:
    raw = ollama_client.list()
    models = raw.models if hasattr(raw, "models") else raw.get("models", [])
    result = []
    for m in models:
        name = m.model if hasattr(m, "model") else m.get("model", m.get("name", ""))
        if name and "embed" not in name:
            result.append(name)
    return result


# ─── QUERY AUGMENTATION ───────────────────────────────────────────────────────

def expand_query(ollama_client: Client, model: str, query: str) -> list[str]:
    """Génère des reformulations synonymiques de la question."""
    prompt = (
        "Reformule cette question en 3 variantes courtes avec des synonymes différents.\n"
        "Retourne UNIQUEMENT les 3 reformulations, une par ligne, sans numérotation ni explication.\n"
        f"Question : {query}"
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        variants = [l.strip() for l in resp["message"]["content"].split("\n") if l.strip()]
        return [query] + variants[:3]
    except Exception:
        return [query]


def hyde_query(ollama_client: Client, model: str, query: str) -> str:
    """HyDE : génère une réponse hypothétique pour améliorer la recherche vectorielle."""
    prompt = (
        "Rédige un court paragraphe (3-4 phrases) qui serait une réponse plausible à cette question.\n"
        "Utilise un vocabulaire précis et varié. N'indique pas que c'est hypothétique.\n"
        f"Question : {query}"
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return resp["message"]["content"].strip()
    except Exception:
        return query


# ─── TOKENISATION ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenisation simple : minuscules, suppression ponctuation, split."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ─── HYBRID SEARCH ────────────────────────────────────────────────────────────

def retrieve_context_hybrid(
    collection,
    query: str,
    ollama_client: Client,
    model: str,
    n_results: int,
    seuil: float,
    alpha: float,
    use_hyde: bool,
    use_expansion: bool,
    use_reranker: bool = True,
    reranker_model: str = "BAAI/bge-reranker-v2-gemma",
    doc_date_filter: str = "",
) -> tuple[list[str], list[tuple], list[dict]]:
    """
    Hybrid Search : fusion score vectoriel ChromaDB + score BM25.

    Retourne :
        contexts       — liste de strings prêts à injecter dans le prompt
        sources        — liste de tuples (source_name, hybrid_score, vecto_dist, doc_date)
        detailed_chunks — liste de dicts avec toutes les métriques (pour visualisation)
    """
    queries = [query]
    if use_expansion:
        queries = expand_query(ollama_client, model, query)
    if use_hyde:
        queries.append(hyde_query(ollama_client, model, query))

    per_query = max(5, n_results // len(queries))

    # ── Récupération vectorielle ──────────────────────────────────────────────
    candidates: dict[str, dict] = {}
    where_filter = {"doc_date": doc_date_filter} if doc_date_filter else None
    for q in queries:
        try:
            if where_filter:
                r = collection.query(query_texts=[q], n_results=per_query, where=where_filter)
            else:
                r = collection.query(query_texts=[q], n_results=per_query)
            for i, doc_id in enumerate(r["ids"][0]):
                dist = r["distances"][0][i]
                if dist <= seuil and doc_id not in candidates:
                    candidates[doc_id] = {
                        "document": r["documents"][0][i],
                        "metadata": r["metadatas"][0][i],
                        "vecto_distance": dist,
                    }
        except Exception:
            continue

    if not candidates:
        return [], [], []

    ids = list(candidates.keys())
    docs = [candidates[i]["document"] for i in ids]
    metas = [candidates[i]["metadata"] for i in ids]
    vecto_distances = [candidates[i]["vecto_distance"] for i in ids]

    # ── Scores normalisés ─────────────────────────────────────────────────────
    vecto_scores = [1 - d / 2 for d in vecto_distances]
    max_v = max(vecto_scores) or 1
    vecto_scores_norm = [s / max_v for s in vecto_scores]

    corpus_tokens = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokenize(query))
    max_b = max(bm25_scores) or 1
    bm25_scores_norm = [s / max_b for s in bm25_scores]

    hybrid_scores = [
        alpha * vecto_scores_norm[i] + (1 - alpha) * bm25_scores_norm[i]
        for i in range(len(ids))
    ]

    # ── Hybrid ranking initial ────────────────────────────────────────────────
    ranked = sorted(
        zip(hybrid_scores, vecto_distances, bm25_scores, docs, metas),
        key=lambda x: x[0],
        reverse=True,
    )[:n_results * 2]  # on garde 2× plus de candidats pour le reranker
    # ── Reranking ─────────────────────────────────────────────────────────────
    if use_reranker:
        ranked_with_rerank = rerank_chunks(query, ranked, top_n=n_results)
    else:
        ranked_with_rerank = [(*item, 0.0) for item in ranked[:n_results]]


    # ── Construction des résultats ────────────────────────────────────────────
    contexts: list[str] = []
    sources: list[tuple] = []
    detailed_chunks: list[dict] = []
    seen_sources: set[str] = set()

    for hybrid_score, vecto_dist, bm25_score, doc, meta, rerank_score in ranked_with_rerank:
        if "source" in meta and "page" in meta:
            source_name = f"📄 {meta['source']} (Page {meta['page']})"
            chunk_type = "pdf"
            doc_date = meta.get("doc_date", "")
        elif "acronyme" in meta:
            source_name = f"📚 Lexique : {meta['acronyme']}"
            chunk_type = "lexique"
            doc_date = ""
        else:
            source_name = "Document inconnu"
            chunk_type = "unknown"
            doc_date = ""

        context_line = f"Extrait de {source_name}"
        if doc_date:
            context_line += f" [Document du {doc_date}]"
        context_line += f" :\n{doc}"
        contexts.append(context_line)

        if source_name not in seen_sources:
            sources.append((source_name, hybrid_score, vecto_dist, doc_date))
            seen_sources.add(source_name)

        detailed_chunks.append({
            "source": source_name,
            "type": chunk_type,
            "document": doc,
            "metadata": meta,
            "hybrid_score": hybrid_score,
            "vecto_distance": vecto_dist,
            "bm25_score": bm25_score,
            "doc_date": doc_date,
            "rerank_score": rerank_score,
        })

    return contexts, sources, detailed_chunks


# ─── GÉNÉRATION LLM ───────────────────────────────────────────────────────────

def build_system_prompt(context_str: str) -> str:
    return f"""Tu es un assistant IA expert, concis et professionnel.
Ta mission est de répondre à la question de l'utilisateur en utilisant UNIQUEMENT le contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis poliment "Je ne trouve pas cette information dans les documents fournis", et n'invente rien.
Réponds en français.

RÈGLES IMPORTANTES :
- Nous sommes en Mai 2026.
- Les dates des documents sont indiquées entre crochets [Document du YYYY-MM-DD].
- Si plusieurs documents traitent le même sujet avec des dates différentes, PRIORISE TOUJOURS le document le plus récent.

CONTEXTE :
{context_str}
"""


def stream_answer(ollama_client: Client, model: str, system_prompt: str, user_question: str):
    """Générateur de tokens pour la réponse en streaming."""
    for chunk in ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        stream=True,
        options={"temperature": 0.0},
    ):
        yield chunk["message"]["content"]


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_reranker_model = None
_reranker_tokenizer = None

def get_reranker(model_name: str = "BAAI/bge-reranker-v2-gemma"):
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        _reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _reranker_model = AutoModelForCausalLM.from_pretrained(  # ← CausalLM, pas SequenceClassification
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        _reranker_model.eval()
        if torch.cuda.is_available():
            _reranker_model = _reranker_model.cuda()
    return _reranker_tokenizer, _reranker_model


def rerank_chunks(
    query: str,
    ranked: list[tuple],
    top_n: int | None = None,
    model_name: str = "BAAI/bge-reranker-v2-gemma",
    batch_size: int = 4,
) -> list[tuple]:
    if not ranked:
        return []

    tokenizer, model = get_reranker(model_name)

    # bge-reranker-v2-gemma utilise un prompt structuré et lit le logit du token "Yes"
    YES_TOKEN_ID = tokenizer.encode("Yes", add_special_tokens=False)[0]

    def score_pair(query: str, doc: str) -> float:
        prompt = (
            f"Given a query and a passage, predict whether the passage is relevant "
            f"to the query.\nQuery: {query}\nPassage: {doc}\nRelevant (Yes/No):"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab_size)
            # logit du dernier token prédit → prob "Yes"
            last_logits = logits[0, -1, :]  # (vocab_size,)
            yes_no_logits = last_logits[[YES_TOKEN_ID]]
            score = torch.softmax(yes_no_logits, dim=-1)[0].item()
        return score

    docs = [doc for _, _, _, doc, _ in ranked]
    rerank_scores = [score_pair(query, doc) for doc in docs]

    reranked = [
        (*item, score)
        for item, score in zip(ranked, rerank_scores)
    ]
    reranked.sort(key=lambda x: x[-1], reverse=True)

    if top_n:
        reranked = reranked[:top_n]

    return reranked


# ─── 1. QUERY REWRITING ───────────────────────────────────────────────────────
 
def rewrite_query(
    ollama_client,
    model: str,
    query: str,
    chat_history: list[dict],
) -> str:
    """
    Reformule la question de l'utilisateur en une query autonome et complète,
    en tenant compte de l'historique de conversation.
 
    Exemple :
        Historique : "Où déposer l'accord d'intéressement ?"
        Question   : "y a t il un délai ?"
        → Résultat : "Quel est le délai de dépôt de l'accord d'intéressement ?"
 
    Si la question est déjà autonome (pas de pronom anaphorique, pas d'ellipse),
    le LLM la retourne telle quelle — pas de reformulation inutile.
    """
    if not chat_history:
        # Pas d'historique → rien à résoudre
        return query
 
    # On formate les N derniers échanges (évite des prompts trop longs)
    MAX_TURNS = 4
    recent = chat_history[-(MAX_TURNS * 2):]
    history_str = "\n".join(
        f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'} : {m['content']}"
        for m in recent
    )
 
    prompt = (
        "Tu es un assistant qui reformule des questions.\n"
        "Voici l'historique récent de la conversation :\n"
        f"{history_str}\n\n"
        "Nouvelle question de l'utilisateur : « {query} »\n\n"
        "Ta tâche : si cette question contient des pronoms, ellipses ou références "
        "implicites à l'historique (ex: 'y a t il un délai ?', 'et pour lui ?', "
        "'quel est ce montant ?'), reformule-la en incluant explicitement "
"le sujet principal de la conversation et toute population ou cas particulier "
"mentionné dans l'historique.\n"
        "Si la question est déjà autonome, retourne-la EXACTEMENT telle quelle.\n"
        "Retourne UNIQUEMENT la question reformulée, sans explication ni ponctuation "
        "supplémentaire."
    ).format(query=query)
 
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        rewritten = resp["message"]["content"].strip().strip("«»\"'")
        # Garde la query originale si la reformulation est vide ou aberrante
        return rewritten if len(rewritten) > 5 else query
    except Exception:
        return query
 
 
# ─── 2. STREAM ANSWER — version avec historique ───────────────────────────────
 
def stream_answer(
    ollama_client,
    model: str,
    system_prompt: str,
    user_question: str,
    chat_history: list[dict] | None = None,
):
    """
    Générateur de tokens pour la réponse en streaming.
 
    chat_history : liste de dicts {"role": "user"|"assistant", "content": str}
                   représentant les tours PRÉCÉDENTS (sans le tour en cours).
                   Si None ou vide → comportement identique à l'original.
 
    Le system_prompt (contexte RAG) est injecté en premier message système.
    L'historique est ensuite rejoué tel quel, puis la question courante est ajoutée.
    """
    MAX_HISTORY_TURNS = 6  # nb de tours (user+assistant) à conserver
 
    messages = [{"role": "system", "content": system_prompt}]
 
    if chat_history:
        # Tronque pour rester dans le context window du modèle
        trimmed = chat_history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(trimmed)
 
    messages.append({"role": "user", "content": user_question})
 
    for chunk in ollama_client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": 0.0},
    ):
        yield chunk["message"]["content"]