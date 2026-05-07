"""
rag_demo.py — Démo RAG avec ChromaDB et Ollama
──────────────────────────────────────────────
Prérequis :
    pip install streamlit chromadb ollama
    
Il vous faut un modèle de génération (ex: llama3, mistral, ou gemma)
à télécharger via : ollama pull llama3
"""

import re
from rank_bm25 import BM25Okapi

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8100)) # Adaptez à votre port
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ─── CLIENTS OLLAMA & CHROMA ──────────────────────────────────────────────────
@st.cache_resource
def get_chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

@st.cache_resource
def get_ollama_client():
    return Client(host=OLLAMA_HOST)

@st.cache_resource
def get_embedding_fn():
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_HOST + "/api/embeddings",
        model_name="embeddinggemma"
    )

def get_collection(client, collection_name: str):
    return client.get_collection(
        name=collection_name,
        embedding_function=get_embedding_fn()
    )

#def retrieve_context(collection, query: str, n_results: int, seuil: float):
#    """Recherche les chunks pertinents et extrait les sources."""
#    results = collection.query(query_texts=[query], n_results=n_results)
#    
#    contexts = []
#    sources_utilisees = set()
#    
#    if not results["ids"][0]:
#        return contexts, sources_utilisees
#        
#    for i, dist in enumerate(results["distances"][0]):
#        if dist <= seuil:
#            doc = results["documents"][0][i]
#            meta = results["metadatas"][0][i]
#            
#            # Formater la source selon le type de collection (PDF ou Lexique)
#            if "source" in meta and "page" in meta:
#                source_name = f"📄 {meta['source']} (Page {meta['page']})"
#            elif "acronyme" in meta:
#                source_name = f"📚 Lexique : {meta['acronyme']}"
#            else:
#                source_name = "Document inconnu"
#                
#            contexts.append(f"Extrait de {source_name} :\n{doc}")
#            sources_utilisees.add(source_name)
#            
#    return contexts, list(sources_utilisees)
#

############################################################################################################################################
def expand_query(ollama_client, model: str, query: str) -> list[str]:
    """
    Génère des reformulations synonymiques de la question.
    Retourne la question originale + variantes.
    """
    prompt = f"""Reformule cette question en 3 variantes courtes avec des synonymes différents.
Retourne UNIQUEMENT les 3 reformulations, une par ligne, sans numérotation ni explication.
Question : {query}"""
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4}
        )
        variants = [l.strip() for l in resp["message"]["content"].split("\n") if l.strip()]
        return [query] + variants[:3]
    except Exception:
        return [query]


def hyde_query(ollama_client, model: str, query: str) -> str:
    """
    HyDE : génère une réponse hypothétique pour améliorer la recherche vectorielle.
    L'embedding de cette réponse est plus proche des chunks réels que celui de la question.
    """
    prompt = f"""Rédige un court paragraphe (3-4 phrases) qui serait une réponse plausible à cette question.
Utilise un vocabulaire précis et varié. N'indique pas que c'est hypothétique.
Question : {query}"""
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        return resp["message"]["content"].strip()
    except Exception:
        return query


def retrieve_context_enhanced(collection, query: str, ollama_client, model: str,
                               n_results: int, seuil: float, use_hyde: bool, use_expansion: bool):
    """
    Recherche enrichie : HyDE et/ou query expansion + déduplication par id.
    """
    # Construire la liste de requêtes à lancer
    queries = [query]

    if use_expansion:
        queries = expand_query(ollama_client, model, query)

    if use_hyde:
        queries.append(hyde_query(ollama_client, model, query))

    # Lancer toutes les requêtes et fusionner les résultats (dédup par id)
    seen_ids = set()
    all_results = []  # [(distance, document, metadata)]

    for q in queries:
        try:
            per_query = max(5, n_results // len(queries))
            r = collection.query(query_texts=[q], n_results=per_query)
            for i, doc_id in enumerate(r["ids"][0]):
                dist = r["distances"][0][i]
                if dist <= seuil and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append((dist, r["documents"][0][i], r["metadatas"][0][i]))
        except Exception:
            continue

    # Trier par distance croissante
    all_results.sort(key=lambda x: x[0])
    all_results = all_results[:n_results]

    contexts = []
    sources = []
    seen_sources = set()

    for dist, doc, meta in all_results:
        if "source" in meta and "page" in meta:
            source_name = f"📄 {meta['source']} (Page {meta['page']})"
        elif "acronyme" in meta:
            source_name = f"📚 Lexique : {meta['acronyme']}"
        else:
            source_name = "Document inconnu"

        contexts.append(f"Extrait de {source_name} :\n{doc}")
        if source_name not in seen_sources:
            sources.append(source_name)
            seen_sources.add(source_name)

    return contexts, sources




def tokenize(text: str) -> list[str]:
    """Tokenisation simple : minuscules, suppression ponctuation, split."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


def retrieve_context_hybrid(collection, query: str, ollama_client, model: str,
                             n_results: int, seuil: float,
                             alpha: float,
                             use_hyde: bool, use_expansion: bool):
    """
    Hybrid Search : fusion score vectoriel ChromaDB + score BM25.
    alpha=1.0 → 100% vectoriel | alpha=0.0 → 100% BM25
    Score final : alpha * score_vecto + (1-alpha) * score_bm25
    Les deux scores sont normalisés entre 0 et 1 avant fusion.
    """

    # ── 1. Construire la liste de requêtes ────────────────────────────────────
    queries = [query]
    if use_expansion:
        queries = expand_query(ollama_client, model, query)
    if use_hyde:
        queries.append(hyde_query(ollama_client, model, query))

    per_query = max(5, n_results // len(queries))

    # ── 2. Récupérer les candidats vectoriels (toutes requêtes, dédupliqués) ──
    candidates = {}   # id → {document, metadata, vecto_distance}

    for q in queries:
        try:
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
        return [], []

    ids       = list(candidates.keys())
    docs      = [candidates[i]["document"] for i in ids]
    metas     = [candidates[i]["metadata"]  for i in ids]
    vecto_distances = [candidates[i]["vecto_distance"] for i in ids]

    # ── 3. Score vectoriel normalisé (distance → similarité) ─────────────────
    # ChromaDB retourne des distances cosine : 0 = identique, 2 = opposé
    # On convertit en similarité : 1 - dist/2  → [0, 1]
    vecto_scores = [1 - d / 2 for d in vecto_distances]
    max_v = max(vecto_scores) or 1
    vecto_scores_norm = [s / max_v for s in vecto_scores]

    # ── 4. Score BM25 normalisé ───────────────────────────────────────────────
    corpus_tokens = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokenize(query))
    max_b = max(bm25_scores) or 1
    bm25_scores_norm = [s / max_b for s in bm25_scores]

    # ── 5. Score hybride fusionné ─────────────────────────────────────────────
    hybrid_scores = [
        alpha * vecto_scores_norm[i] + (1 - alpha) * bm25_scores_norm[i]
        for i in range(len(ids))
    ]

    # ── 6. Tri décroissant par score hybride ──────────────────────────────────
    ranked = sorted(
        zip(hybrid_scores, vecto_distances, docs, metas),
        key=lambda x: x[0],
        reverse=True
    )[:n_results]

    # ── 7. Formater les résultats ─────────────────────────────────────────────
    contexts = []
    sources  = []
    seen_sources = set()

    for hybrid_score, vecto_dist, doc, meta in ranked:
        if "source" in meta and "page" in meta:
            source_name = f"📄 {meta['source']} (Page {meta['page']})"
        elif "acronyme" in meta:
            source_name = f"📚 Lexique : {meta['acronyme']}"
        else:
            source_name = "Document inconnu"

        contexts.append(f"Extrait de {source_name} :\n{doc}")
        if source_name not in seen_sources:
            sources.append((source_name, hybrid_score, vecto_dist))
            seen_sources.add(source_name)

    return contexts, sources

####################################################################################################################s






# ─── INTERFACE UTILISATEUR ────────────────────────────────────────────────────
st.title("🤖 Chatbot RAG (ChromaDB + Ollama)")

try:
    chroma_client = get_chroma_client()
    ollama_client = get_ollama_client()
    
    # Récupération sécurisée des collections
    raw_cols = chroma_client.list_collections()
    collections_dispo = [c.name if hasattr(c, 'name') else c for c in raw_cols]
    
    # Récupération sécurisée des modèles Ollama (anti-crash)
    ollama_list = ollama_client.list()
    raw_models = ollama_list.models if hasattr(ollama_list, 'models') else ollama_list.get('models', [])
    
    modeles_ollama = []
    for m in raw_models:
        # S'adapte selon que 'm' est un objet (v0.2+) ou un dictionnaire (v0.1)
        nom = m.model if hasattr(m, 'model') else m.get('model', m.get('name', ''))
        if nom and "embed" not in nom:
            modeles_ollama.append(nom)
            
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Paramètres RAG")
    
    if not collections_dispo:
        st.warning("Aucune collection trouvée.")
        st.stop()
        
    if not modeles_ollama:
        st.warning("Aucun modèle génératif trouvé dans Ollama. Faites 'ollama pull llama3'.")
        st.stop()

    selected_collection = st.selectbox("Collection ChromaDB", collections_dispo)
    selected_model = st.selectbox("Modèle LLM Ollama", modeles_ollama)
    
    st.markdown("---")
    n_results = st.slider("Chunks à injecter", 1, 100, 50)
    seuil = st.slider("Seuil de distance (cosine)", 0.1, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.markdown("**🔬 Stratégie de recherche**")
    use_hyde      = st.toggle("HyDE (réponse hypothétique)", value=True)
    use_expansion = st.toggle("Query expansion (synonymes)", value=True)
    alpha         = st.slider("Vectoriel ← → BM25", 0.0, 1.0, 0.5, 0.05,
                               help="0.0 = BM25 pur | 1.0 = vectoriel pur | 0.5 = hybride équilibré")
    st.markdown("---")
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ─── LOGIQUE DE CHAT ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Saisie utilisateur
if prompt := st.chat_input("Posez une question sur vos documents..."):
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        try:
            collection = get_collection(chroma_client, selected_collection)
        except Exception as e:
            st.error(f"Impossible de charger la collection : {e}")
            st.stop()
            
        with st.status("🔍 Recherche dans les documents...", expanded=True) as status:
            contexts, sources = retrieve_context_hybrid(
                collection, prompt, ollama_client, selected_model,
                n_results, seuil, alpha, use_hyde, use_expansion
            )

            if not contexts:
                status.update(label="Aucun document pertinent trouvé.", state="error")
                context_str = "Aucun contexte pertinent trouvé."
            else:
                nb_queries = 1 + (3 if use_expansion else 0) + (1 if use_hyde else 0)
                status.update(
                    label=f"{len(contexts)} extraits (sur {nb_queries} requêtes)",
                    state="complete"
                )
                st.write("**Sources consultées :**")
                for name, h_score, v_dist in sources:
                    st.caption(f"- {name} — hybride: {h_score:.3f} | vecto: {v_dist:.4f}")
                context_str = "\n\n---\n\n".join(contexts)

        # Construction du prompt système
        system_prompt = f"""Tu es un assistant IA expert, concis et professionnel.
        Ta mission est de répondre à la question de l'utilisateur en utilisant UNIQUEMENT le contexte fourni ci-dessous.
        Si la réponse n'est pas dans le contexte, dis poliment "Je ne trouve pas cette information dans les documents fournis", et n'invente rien.
        Réponds en français.
        Nous sommes en Mai 2026.
        Si deux documents donnent des informations sur la même chose, tu dois ignorer le document dont le titre contient la date la plus ancienne.

        CONTEXTE :
        {context_str}
        """

        message_placeholder = st.empty()
        full_response = ""
        
        # Appel à Ollama en mode streaming
        try:
            for chunk in ollama_client.chat(
                model=selected_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
                options={
        "temperature": 0.0 # Force le modèle à être le plus factuel et déterministe possible
    }
            ):
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")
            
            # Affichage final sans le curseur
            message_placeholder.markdown(full_response)
            
            # Sauvegarde dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Erreur avec Ollama : {e}")



