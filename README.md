# Explainable RAG with Knowledge Graph

An Explainable Retrieval-Augmented Generation (RAG) system that combines **Vector Search** and **Knowledge Graphs** to provide transparent, sourced answers about EU Environmental Policy (Green Deal).

## 📸 Screenshots

### 1. Home Page & Initialization
*The application interface where you configure the database and initialize the system.*
![Home Page](docs\images\home page.PNG)

### 2. Execution & Results
*The system answering a question with hybrid retrieval, showing the answer, explanation, and citations.*
![Execution Result](docs\images\execution.PNG)

## 🚀 Features
- **Hybrid Search**: Combines semantic search (embeddings) with structured graph queries (Neo4j).
- **Explainability**: Every answer includes a reasoning path and specific text citations.
- **Knowledge Graph**: Maps Policies, Targets, Countries, and Sectors.
- **Free Tier Friendly**: Configured to use **Groq (Llama 3)** and **HuggingFace Embeddings**.

## 📋 Prerequisites
- **Python 3.11+**
- **Groq API Key** (Free)
- **Neo4j Database** (Optional - only for Full Mode. Use Neo4j Desktop, Docker, or Aura Free Tier).

## 🛠️ Installation

1. **Clone & Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory:
   ```ini
   GROQ_API_KEY=gsk_...
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   ```

## 🏃 Usage

### Option A : Démo Simplifiée (Sans Neo4j)

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 2. Installer les dépendances minimales
pip install groq sentence-transformers python-dotenv

# 3. Configurer la clé API
# Ajoutez GROQ_API_KEY dans votre fichier .env

# 4. Préparer les données
python src/data_preparation.py

# 5. Lancer la démo
python demo_simple.py
```

### Option B : Système Complet (Avec Neo4j)

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 2. Installer toutes les dépendances
pip install -r requirements.txt

# 3. Configurer les variables d'environnement
# Copiez .env.example vers .env et ajoutez votre GROQ_API_KEY

# 4. Démarrer Neo4j
docker-compose up -d

# 5. Préparer les données
python src/data_preparation.py

# 6. Construire le graphe (utilise l'API Groq)
python src/kg_builder.py

# 7. Lancer l'interface Streamlit
streamlit run app/streamlit_app.py
```

## ❓ Exemples de Questions

Une fois le système lancé, essayez ces questions :

- "Quels sont les objectifs de l'UE en matière d'énergies renouvelables pour 2030 ?"
- "Comment RED III impacte-t-il le secteur des transports ?"
- "Quelles sources d'énergie renouvelable sont promues par le Green Deal ?"
- "Quel est l'objectif de l'Allemagne en matière d'électricité renouvelable pour 2030 ?"

## 📂 Project Structure
- `src/data_preparation.py`: Cleans and chunks text documents.
- `src/kg_builder.py`: Extracts entities using LLM and populates Neo4j.
- `src/rag_system.py`: Main logic for Hybrid Retrieval and Answer Generation.
- `app/streamlit_app.py`: Interactive Web Interface.
- `demo_simple.py`: Lightweight CLI demo.
