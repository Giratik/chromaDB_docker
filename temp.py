"""
rag_demo.py — Démo RAG Avancé (ChromaDB + FlashRank + Ollama)
─────────────────────────────────────────────────────────────
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client
from flashrank import Ranker, RerankRequest

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8100))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

st.set_page_config(page_title="RAG Chatbot Avancé", page_icon="🧠", layout="wide")

# ─── CLIENTS & MODÈLES ────────────────────────────────────────────────────────
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
        model_name="nomic-embed-text"
    )

@st.cache_resource
def get_reranker():
    # Télécharge et charge le modèle de reranking (la 1ère fois ça peut prendre 1 minute)
    # bge-reranker-v2-m3 est excellent pour le français
    return Ranker(model_name="bge-reranker-v2-m3", cache_dir="./models")

def get_collection(client, collection_name: str):
    return client.get_collection(
        name=collection_name,
        embedding_function=get_embedding_fn()
    )

def retrieve_and_rerank(collection, ranker, query: str, n_final: int):
    """
    Architecture en 2 étapes :
    1. Retrieval large avec Chroma (30 chunks)
    2. Reranking ultra-précis avec FlashRank (Top N)
    """
    # 1. Demander BEAUCOUP de résultats à ChromaDB
    n_chroma = max(30, n_final * 4)
    results = collection.query(query_texts=[query], n_results=n_chroma)
    
    if not results["ids"][0]:
        return [], []
        
    # 2. Préparer les données pour le Reranker
    passages = []
    for i in range(len(results["ids"][0])):
        passages.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "meta": results["metadatas"][0][i]
        })
        
    # 3. Noter chaque passage (Cross-Encoder)
    rerankrequest = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerankrequest)
    
    # 4. Garder seulement les N meilleurs
    top_results = reranked_results[:n_final]
    
    contexts = []
    sources_utilisees = set()
    
    for r in top_results:
        meta = r["meta"]
        doc = r["text"]
        score = r.get("score", 0.0)
        
        if "source" in meta and "page" in meta:
            source_name = f"📄 {meta['source']} (Page {meta['page']})"
        elif "acronyme" in meta:
            source_name = f"📚 Lexique : {meta['acronyme']}"
        else:
            source_name = "Document inconnu"
            
        # On injecte le score de pertinence pour audit visuel
        contexts.append(f"Extrait de {source_name} (Score Reranker: {score:.3f}) :\n{doc}")
        sources_utilisees.add(source_name)
        
    return contexts, list(sources_utilisees)


# ─── INTERFACE UTILISATEUR ────────────────────────────────────────────────────
st.title("🧠 Chatbot RAG (Recherche Hybride : Chroma + FlashRank)")

try:
    chroma_client = get_chroma_client()
    ollama_client = get_ollama_client()
    ranker = get_reranker()
    
    raw_cols = chroma_client.list_collections()
    collections_dispo = [c.name if hasattr(c, 'name') else c for c in raw_cols]
    
    ollama_list = ollama_client.list()
    raw_models = ollama_list.models if hasattr(ollama_list, 'models') else ollama_list.get('models', [])
    modeles_ollama = [m.model if hasattr(m, 'model') else m.get('model', m.get('name', '')) 
                      for m in raw_models if "embed" not in (m.model if hasattr(m, 'model') else m.get('model', m.get('name', '')))]
except Exception as e:
    st.error(f"Erreur d'initialisation : {e}")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Paramètres RAG")
    selected_collection = st.selectbox("Collection ChromaDB", collections_dispo)
    selected_model = st.selectbox("Modèle LLM Ollama", modeles_ollama)
    
    st.markdown("---")
    n_results = st.slider("Chunks finaux (après reranking)", 1, 10, 3, 
                          help="Nombre de documents donnés au LLM. Le reranker analyse toujours 30 chunks en arrière-plan.")
    
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ─── LOGIQUE DE CHAT ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez une question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        try:
            collection = get_collection(chroma_client, selected_collection)
        except Exception as e:
            st.error(f"Impossible de charger la collection : {e}")
            st.stop()
            
        with st.status("🔍 Recherche et Reranking en cours...", expanded=True) as status:
            # Appel à notre nouvelle fonction de Reranking
            contexts, sources = retrieve_and_rerank(collection, ranker, prompt, n_results)
            
            if not contexts:
                status.update(label="Aucun document pertinent trouvé.", state="error")
                context_str = "Aucun contexte pertinent trouvé."
            else:
                status.update(label=f"Top {len(contexts)} extraits validés par le Reranker !", state="complete")
                st.write("**Sources consultées :**")
                for s in sources:
                    st.caption(f"- {s}")
                context_str = "\n\n---\n\n".join(contexts)
                
                with st.expander("Voir le contexte injecté au LLM"):
                    st.text(context_str)

        system_prompt = f"""Tu es un assistant IA expert et strictement factuel.
        Ta mission est de répondre à la question en utilisant UNIQUEMENT le contexte fourni.

        RÈGLE ABSOLUE : 
        Si la réponse ne se trouve pas clairement dans le contexte, tu dois répondre EXACTEMENT par cette phrase unique : "Je ne trouve pas cette information dans les documents fournis." 
        N'ajoute aucune autre phrase, ne donne pas d'explication.

        CONTEXTE :
        {context_str}
        """

        message_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in ollama_client.chat(
                model=selected_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
                options={"temperature": 0.0}
            ):
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Erreur avec Ollama : {e}")