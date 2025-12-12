"""
Streamlit Demo Application
Interactive interface for the Explainable RAG system.
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile

import streamlit as st
from pyvis.network import Network

from data_preparation import DocumentProcessor
from rag_system import ExplainableRAGSystem

# Page configuration
st.set_page_config(
    page_title="Explainable RAG - EU Green Deal", page_icon="🌱", layout="wide"
)

# Title and description
st.title("🌱 Explainable RAG System")
st.subheader("EU Green Deal & Renewable Energy Knowledge Assistant")

st.markdown(
    """
This system uses **hybrid retrieval** (vector search + knowledge graph) to answer questions 
about EU environmental policy and renewable energy targets. It provides:
- **Accurate answers** based on official documents
- **Transparent explanations** showing the reasoning path through the knowledge graph
- **Source citations** linking back to original text chunks
"""
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

neo4j_uri = st.sidebar.text_input("Neo4j URI", value="bolt://localhost:7687")
neo4j_user = st.sidebar.text_input("Neo4j User", value="neo4j")
neo4j_password = st.sidebar.text_input(
    "Neo4j Password", value="password", type="password"
)

top_k = st.sidebar.slider(
    "Number of text chunks to retrieve", min_value=1, max_value=10, value=3
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False

# Initialize button
if st.sidebar.button("🚀 Initialize System"):
    with st.spinner("Initializing RAG system..."):
        try:
            # Initialize RAG system
            rag = ExplainableRAGSystem(neo4j_uri, neo4j_user, neo4j_password)

            # Load and process data
            processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
            raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
            chunks_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chunks")

            all_chunks = []
            for file_name in os.listdir(raw_dir):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(raw_dir, file_name)
                    chunks = processor.process_document(file_path, chunks_dir)
                    all_chunks.extend(chunks)

            # Index chunks
            rag.load_and_index_chunks(all_chunks)

            st.session_state.rag_system = rag
            st.session_state.initialized = True

            st.sidebar.success(f"✅ System initialized with {len(all_chunks)} chunks!")

        except Exception as e:
            st.sidebar.error(f"❌ Error initializing system: {e}")

# Main interface
if st.session_state.initialized:
    st.markdown("---")

    # Example questions
    st.markdown("### 💡 Example Questions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("What are the EU's 2030 renewable energy targets?"):
            st.session_state.query = "What are the EU's 2030 renewable energy targets?"
        if st.button("Which countries are mentioned in the Green Deal?"):
            st.session_state.query = "Which countries are mentioned in the Green Deal?"

    with col2:
        if st.button("How does RED III impact the transport sector?"):
            st.session_state.query = "How does RED III impact the transport sector?"
        if st.button("What renewable sources are promoted by EU policies?"):
            st.session_state.query = (
                "What renewable sources are promoted by EU policies?"
            )

    # Query input
    st.markdown("### 🔍 Ask a Question")
    query = st.text_input(
        "Enter your question about EU environmental policy:",
        value=st.session_state.get("query", ""),
        placeholder="e.g., What are the renewable energy targets for 2030?",
    )

    if st.button("🔎 Search", type="primary") or query:
        if query:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Query the RAG system
                    response = st.session_state.rag_system.query(query, top_k=top_k)

                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "📝 Answer",
                            "🧠 Explanation",
                            "📚 Citations",
                            "🕸️ Knowledge Graph",
                        ]
                    )

                    with tab1:
                        st.markdown("### Answer")
                        st.markdown(response["answer"])

                    with tab2:
                        st.markdown("### Explanation")
                        st.markdown(response["explanation"])

                        # Show Cypher query if available
                        if response["graph_data"].get("cypher"):
                            with st.expander("🔧 View Cypher Query"):
                                st.code(
                                    response["graph_data"]["cypher"], language="cypher"
                                )

                    with tab3:
                        st.markdown("### Source Citations")
                        st.markdown(response["citations"])

                        # Show detailed vector sources
                        with st.expander("📄 View Retrieved Text Chunks"):
                            for i, source in enumerate(response["vector_sources"], 1):
                                st.markdown(
                                    f"**Chunk {i}** (Score: {source['score']:.3f})"
                                )
                                st.markdown(
                                    f"*Source: {source['metadata']['source_document']} - {source['metadata']['chunk_id']}*"
                                )
                                st.text(source["text"][:300] + "...")
                                st.markdown("---")

                    with tab4:
                        st.markdown("### Knowledge Graph Visualization")

                        # Display graph data
                        graph_data = response["graph_data"]

                        if graph_data.get("nodes"):
                            st.markdown(
                                f"**Nodes retrieved:** {len(graph_data['nodes'])}"
                            )
                            st.markdown(
                                f"**Relationships retrieved:** {len(graph_data.get('relationships', []))}"
                            )

                            # Create network visualization
                            net = Network(
                                height="500px",
                                width="100%",
                                bgcolor="#222222",
                                font_color="white",
                            )
                            net.barnes_hut()

                            # Add nodes
                            for node in graph_data["nodes"][
                                :15
                            ]:  # Limit for visualization
                                label = node.get("name", "Unknown")
                                node_type = ", ".join(node.get("labels", ["Unknown"]))
                                color = {
                                    "Policy": "#FF6B6B",
                                    "Target": "#4ECDC4",
                                    "Legislation": "#45B7D1",
                                    "Country": "#FFA07A",
                                    "RenewableSource": "#98D8C8",
                                    "Sector": "#F7DC6F",
                                }.get(node_type.split(",")[0].strip(), "#CCCCCC")

                                net.add_node(
                                    node["id"],
                                    label=label,
                                    title=f"{node_type}: {label}",
                                    color=color,
                                )

                            # Add edges
                            for rel in graph_data.get("relationships", [])[:15]:
                                net.add_edge(
                                    rel["start_node"],
                                    rel["end_node"],
                                    title=rel["type"],
                                    label=rel["type"],
                                )

                            # Save and display
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".html", mode="w"
                            ) as f:
                                net.save_graph(f.name)
                                with open(f.name, "r") as html_file:
                                    html_content = html_file.read()
                                    st.components.v1.html(html_content, height=500)

                            # Show node details
                            with st.expander("📊 View Node Details"):
                                for node in graph_data["nodes"][:10]:
                                    st.json(node)
                        else:
                            st.info("No graph data retrieved for this query.")

                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback

                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a question.")

else:
    st.info("👈 Please initialize the system using the sidebar configuration.")

    st.markdown(
        """
    ### 🚀 Getting Started
    
    1. Ensure Neo4j is running locally on `bolt://localhost:7687`
    2. Set your Neo4j credentials in the sidebar
    3. Click **Initialize System** to load the knowledge base
    4. Start asking questions!
    
    ### 📋 System Requirements
    
    - Neo4j database (running locally or remotely)
    - Groq API key (set in environment variable `GROQ_API_KEY`)
    - Python dependencies installed from `requirements.txt`
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>Explainable RAG System | EU Green Deal & Renewable Energy | Built with LlamaIndex, Neo4j, and Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
