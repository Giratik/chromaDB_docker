# 🧠 ChromaDB Manager

Interface Streamlit pour gérer la base vectorielle ChromaDB utilisée par le lexique EDP.  
ChromaDB est exposé en mode serveur HTTP sur le port **8000**, accessible par n'importe quelle autre application.

---

## 🗂️ Structure

```
chroma-manager/
├── app/
│   ├── main.py            # Interface Streamlit
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── init.sh                # À lancer une fois après le premier démarrage
└── README.md
```

---

## 🚀 Démarrage

### 1. Lancer les services

```bash
docker compose up -d --build
```

### 2. Télécharger le modèle Ollama (première fois uniquement)

```bash
chmod +x init.sh && ./init.sh
```

### 3. Accéder à l'interface

- **Manager Streamlit** → http://localhost:8600  
- **API ChromaDB** → http://localhost:8100/api/v1/heartbeat

---

## 📋 Fonctionnalités

| Page | Description |
|------|-------------|
| Dashboard | Vue globale : nombre d'entrées, aperçu rapide |
| Parcourir | Rechercher/filtrer les entrées du lexique |
| Ajouter | Ajouter un nouvel acronyme manuellement |
| Modifier | Modifier un acronyme ou sa signification |
| Supprimer | Supprimer une entrée ou vider la collection |
| Import JSON | Importer un `lexique.json` (fusion ou remplacement) |
| Export | Télécharger le lexique en JSON ou CSV |

---

## 🔌 Connecter d'autres applications à ChromaDB

### Depuis un autre docker-compose (ex: ton chatbot)

Dans le `docker-compose.yml` du chatbot, déclare le réseau externe :

```yaml
services:
  backend:
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    networks:
      - chroma_net

networks:
  chroma_net:
    external: true   # ← utilise le réseau créé par ce docker-compose
```

### Depuis ton code Python existant

Remplace `chromadb.PersistentClient` par `chromadb.HttpClient` :

```python
import chromadb

# Avant (fichier local)
# client = chromadb.PersistentClient(path="./chromadb")

# Après (serveur HTTP)
client = chromadb.HttpClient(
    host=os.environ.get("CHROMA_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_PORT", 8000))
)
```

---

## ⚙️ Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `CHROMA_HOST` | `chromadb` | Hostname du serveur ChromaDB |
| `CHROMA_PORT` | `8000` | Port HTTP de ChromaDB |
| `OLLAMA_HOST` | `http://ollama:11434` | URL du serveur Ollama |

---

## 💡 Ollama déjà installé sur ta machine ?

Si Ollama tourne déjà en dehors de Docker (sur `localhost:11434`),  
retire le service `ollama` du `docker-compose.yml` et change :

```yaml
environment:
  - OLLAMA_HOST=http://host.docker.internal:11434
```

---

## 🔄 Format du lexique.json

```json
[
  { "acronyme": "CODIR", "signification": "Comité de Direction" },
  { "acronyme": "GPEC", "signification": "Gestion Prévisionnelle des Emplois et Compétences" }
]
```

---

## 🔧 Commandes utiles

```bash
# Voir les logs
docker compose logs -f

# Redémarrer un service
docker compose restart chroma-manager

# Arrêter sans perdre les données
docker compose down

# Tout supprimer (données incluses)
docker compose down -v
```


# switcher entre collections

```python
import chromadb
from chromadb.utils import embedding_functions

# 1. Connexion au serveur
client = chromadb.HttpClient(host="chromadb", port=8000)

# 2. Fonction d'embedding (doit être la même que l'ingestion)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://ollama:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# 3. Choix de la collection SPÉCIFIQUE
collection_rh = client.get_collection(name="documents_ressources_humaines", embedding_function=ollama_ef)

# 4. Requête
resultats = collection_rh.query(
    query_texts=["Quels sont les jours de congés ?"],
    n_results=3
)
```