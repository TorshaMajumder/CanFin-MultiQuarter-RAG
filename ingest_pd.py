import os
import yaml
import instructor
from google.generativeai import configure
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import FinancialAnalysis # Import the schema we just made
# from groq import Groq
import pymupdf4llm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main(config, path_to_doc):


    # INITIALIZE ALL CLIENTS
    pc = Pinecone(api_key=config["PC_API_KEY"])
    index = pc.Index(config["pinecone_index_name"])
    embed_model = SentenceTransformer(config["embedding_model_name"])
    client = instructor.from_google(
    client=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=config["GOOGLE_API_KEY"]),
    mode=instructor.Mode.GEMINI_JSON,
    )

    # Define the files and their labels
    docs = [
        {"path": f"{path_to_doc}/shopify_q1.pdf", "quarter": "Q1"},
        {"path": f"{path_to_doc}/shopify_q2.pdf", "quarter": "Q2"},
        {"path": f"{path_to_doc}/shopify_q3.pdf", "quarter": "Q3"},
        {"path": f"{path_to_doc}/shopify_annual.pdf", "quarter": "Full Year"},
    ]


    # Define the Chunker
    # We use a slightly larger chunk size for Markdown to keep tables intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    def process_and_upload(doc_info):
        file_path = doc_info["path"]
        quarter = doc_info["quarter"]
        
        print(f"--- Processing {file_path} ({quarter}) ---")
        
        # Convert PDF to Markdown
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Split Markdown into Chunks
        chunks = text_splitter.split_text(md_text)
        print(f"Created {len(chunks)} chunks.")
        
        # Prepare for Pinecone
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            vectors_to_upsert.append({
                "id": f"shopify_2023_{quarter}_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "quarter": quarter,
                    "year": 2023,
                    "company": "Shopify"
                }
            })
        # ====================================================================
        # Delete any exixting namespace "shopify-2023"
        index.delete(delete_all=True, namespace="shopify-2023")
        # Upsert to namespace "shopify-2023"
        index.upsert(vectors=vectors_to_upsert, namespace="shopify-2023")
        print(f"\n Successfully uploaded {quarter} to Pinecone!")
    
    
    def compare_quarters(query, q_a="Q1", q_b="Full Year"):
        
        query_vec = embed_model.encode(query).tolist()
        
        # Targeted Search for Quarter A
        res_a = index.query(
                                vector=query_vec, 
                                top_k=3, 
                                namespace="shopify-2023",
                                filter={"quarter": {"$eq": q_a}}, # THE KEY: Metadata filtering
                                include_metadata=True
                            )
        
        # Targeted Search for Quarter B
        res_b = index.query(
                                vector=query_vec, 
                                top_k=3, 
                                namespace="shopify-2023",
                                filter={"quarter": {"$eq": q_b}}, # THE KEY: Metadata filtering
                                include_metadata=True
                            )
        
        # Combine Context
        context_a = "\n".join([m['metadata']['text'] for m in res_a['matches']])
        context_b = "\n".join([m['metadata']['text'] for m in res_b['matches']])
        
        # Construct the Analysis Prompt
        prompt = f"""
        You are a Senior Financial Analyst. Analyze the performance of Shopify 
        by comparing {q_a} data with {q_b} data based ONLY on the context below.

        --- CONTEXT FOR {q_a} ---
        {context_a}

        --- CONTEXT FOR {q_b} ---
        {context_b}

        TASK:
        Identify the key differences in financial metrics, management tone, 
        or strategic priorities between these two periods. 
        Use a bulleted list for the comparison.
        """

        response = client.chat.completions.create(
                                                        model=FinancialAnalysis,
                                                        messages=[{"role": "user", "content": prompt}],
                                                        temperature=0.1
                                                    )
        
        return response.metrics[0].value

    # ======================================================================================
    # Run the pipeline
    # Upload the PDF to pinecone
    for doc in docs:
        if os.path.exists(doc["path"]):
            process_and_upload(doc)
        else:
            print(f"\nFile not found: {doc['path']}")

    print(f"\n--- Describe index:\n{index.describe_index_stats()}")
    #
    # Run a query
    analysis_query = "What were the main revenue drivers and how did they change?"
    print(f"\nANALYZING: {analysis_query}\n")
    ans = compare_quarters(analysis_query) 
    print(f"\n---\n{ans}")
    # =======================================================================================



if __name__ == "__main__":

    path_to_config = os.path.join(os.getcwd(), "config.yaml")
    path_to_doc = os.path.join(os.getcwd(), "shopify_data")
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)

    main(config, path_to_doc)