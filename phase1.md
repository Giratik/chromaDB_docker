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