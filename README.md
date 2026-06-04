## A Multimodal, Temporal RAG System for Canadian Financial Reports

#### 🚀 Setup & Installation

**Prerequisites**
* Python 3.9+
* A Pinecone API Key ([Get one here](https://www.pinecone.io/))
* A Groq API Key ([Get one here](https://console.groq.com/))

**1. Clone the Repository**
```bash
git clone https://github.com/TorshaMajumder/CanFin-MultiQuarter-RAG.git
cd CanFin-MultiQuarter-RAG
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Data**
You find all the Shopify 2023 Quartarly and Annual reports in [Shopify Investor Relations](https://www.shopify.com/investors/financial-reports) page.

**4. Data Ingestion**
Place your Shopify 2023 PDFs (`shopify_q1.pdf`, `shopify_q2.pdf`, etc.) in the root directory and run the ingestion pipeline. This script parses the PDFs into Markdown, chunks the text, and upserts vectors to Pinecone.
```bash
python ingest.py
```

**5. Launch the App**
```bash
streamlit run app.py
```

---

#### 🛠 Technical Architecture

This application is built to solve the "Temporal Knowledge" problem in RAG systems, specifically for complex financial documents.

*   **Parsing (Multimodal-Aware):** Used `PyMuPDF4LLM` to convert financial PDFs into **Markdown**. This preserves the structural integrity of complex tables (Balance Sheets/Income Statements) which standard text-extractors often corrupt.
*   **Vector Storage:** **Pinecone (Serverless)**.
*   **Temporal Metadata Strategy:** Implemented a metadata schema involving `quarter`, `year`, and `company` tags. This allows for **Conditional Retrieval**, enabling the system to isolate and compare specific fiscal periods using Pinecone's `$eq` filters.
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`. This model was chosen for its efficiency in mapping semantic meaning within a 384-dimensional space.
*   **LLM Inference:** **Llama-3.1-8b-instant** via **Groq**. The model is grounded using a strict "System Prompt" to ensure answers are derived exclusively from the retrieved financial context, minimizing hallucinations.
*   **UI Framework:** **Streamlit**, designed with a dual-column source viewer to provide transparency into the "Reasoning" phase of the RAG pipeline.

---

#### 📊 Engineering Challenges Solved
*   **Table Integrity:** Solved the loss of tabular data relationships by utilizing Markdown-based chunking.
*   **Comparison Logic:** Developed a targeted retrieval loop that queries separate namespaces/filters for different quarters, ensuring the LLM receives a balanced context for comparative analysis.

---

## 🚀 Live Demo

<div align="center">
  <video src="YOUR_DRAGGED_LINK_HERE" width="100%" autoplay loop muted playsinline></video>

  <br/>

[Streamlit App](https://github.com/user-attachments/assets/e1320f49-5be3-42d1-9e99-10569a1de8a6)

</div>


---
