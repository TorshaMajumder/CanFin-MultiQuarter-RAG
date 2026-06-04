import os
import yaml
from groq import Groq
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def main(config):
    # --- PAGE CONFIG & STYLING ---
    st.set_page_config(page_title="CanFin Multi-Quarter RAG", layout="wide")
    st.title("🇨🇦 Shopify 2023 Financial Analyst")
    st.markdown("Compare quarterly performance using RAG-powered insights.")

    # --- SIDEBAR - SECRETS & CONFIG ---
    with st.sidebar:
        st.header("Configuration")
        groq_key = st.text_input("Groq API Key", type="password")
        pinecone_key = st.text_input("Pinecone API Key", type="password")
        
        st.divider()
        st.subheader("Comparison Settings")
        q_a = st.selectbox("First Quarter", ["Q1", "Q2", "Q3", "Full Year"], index=0)
        q_b = st.selectbox("Second Quarter", ["Q1", "Q2", "Q3", "Full Year"], index=3)

    # --- INITIALIZE CLIENTS ---
    if groq_key and pinecone_key:
        
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(config["pinecone_index_name"])
        embed_model = SentenceTransformer(config["embedding_model_name"])
        groq_client = Groq(api_key=groq_key)

        # --- THE CHAT INTERFACE ---
        user_query = st.text_input("Ask a comparison question:", 
                                placeholder="e.g. How did gross margins evolve between these periods?")

        if st.button("Run Analysis"):
            with st.spinner(f"Analyzing {q_a} vs {q_b}..."):
                # Embedding the query
                query_vec = embed_model.encode(user_query).tolist()

                # Retrieval with Metadata Filtering
                res_a = index.query(vector=query_vec, top_k=3, namespace="shopify-2023",
                                    filter={"quarter": {"$eq": q_a}}, include_metadata=True)
                
                res_b = index.query(vector=query_vec, top_k=3, namespace="shopify-2023",
                                    filter={"quarter": {"$eq": q_b}}, include_metadata=True)

                context_a = "\n".join([m['metadata']['text'] for m in res_a['matches']])
                context_b = "\n".join([m['metadata']['text'] for m in res_b['matches']])

                # Generation
                system_prompt = f"You are a helpful Financial Analyst. Use the provided context to compare {q_a} and {q_b}."
                user_prompt = f"USER QUESTION: {user_query}\n\nCONTEXT {q_a}:\n{context_a}\n\nCONTEXT {q_b}:\n{context_b}"

                response = groq_client.chat.completions.create(
                                                                model=config["groq_model_name"],
                                                                messages=[
                                                                    {"role": "system", "content": system_prompt},
                                                                    {"role": "user", "content": user_prompt}
                                                                ],
                                                                temperature=0.1
                                                            )

                # Display Results
                st.markdown("### 📊 Analyst Comparison")
                st.write(response.choices[0].message.content)
                
                with st.expander("View Retrieved Sources"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Sources for {q_a}")
                        st.info(context_a)
                    with col2:
                        st.caption(f"Sources for {q_b}")
                        st.info(context_b)
    else:
        st.warning("Please enter your API keys in the sidebar to begin.")


if __name__ == "__main__":

    path_to_config = os.path.join(os.getcwd(), "config.yaml")
    path_to_doc = os.path.join(os.getcwd(), "shopify_data")
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
