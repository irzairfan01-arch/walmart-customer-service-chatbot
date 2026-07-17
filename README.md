# 🛒 Walmart Customer Service ChatBot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green" />
  <img src="https://img.shields.io/badge/GPT--2-Powered-orange" />
</p>

<p align="center">
  An <strong>Agentic AI-powered Customer Service Chatbot</strong> for Walmart, built using <strong>RAG (Retrieval-Augmented Generation)</strong>, Hugging Face Transformers, Sentence Transformers, and Streamlit. Designed to handle customer queries intelligently with context-aware responses.
</p>

---

## ✨ Features

- 🤖 **Agentic AI** – Intelligent response generation using GPT-2 fine-tuned logic
- 🔍 **RAG (Retrieval-Augmented Generation)** – Retrieves the most relevant answers from a customer service dataset before generating a response
- 🧠 **Sentence Transformers** – Uses semantic similarity to find the best-matching response
- 💬 **ChatGPT-style UI** – Clean Streamlit chat interface with Walmart branding
- 📜 **Chat History** – Sidebar with session-based conversation history
- 🛒 **Walmart-branded Design** – Blue & yellow themed UI matching Walmart's identity
- 💾 **Persistent Chat Logs** – Saves and loads chat history via JSON

---

## 🗂️ Project Structure

```
walmart-customer-service-chatbot/
│
├── WalmartCustomerServiceChatBot.ipynb   # Main Jupyter Notebook (development & experiments)
├── chatbot_app.py                        # Streamlit Web App (production-ready UI)
├── CustomerServiceDataSet.csv            # Dataset used for RAG retrieval
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/irzairfan01-arch/walmart-customer-service-chatbot.git
cd walmart-customer-service-chatbot
```

### 2. Install Dependencies
```bash
pip install streamlit pandas torch transformers sentence-transformers numpy
```

### 3. Run the Streamlit App
```bash
streamlit run chatbot_app.py
```

---

## 🧠 How It Works

```
User Query
    │
    ▼
Sentence Transformer (Encodes Query)
    │
    ▼
Semantic Search on CustomerServiceDataSet.csv (RAG Retrieval)
    │
    ▼
Best Matching Context Retrieved
    │
    ▼
GPT-2 Generates Final Response
    │
    ▼
Response Displayed in Chat UI
```

1. The user types a customer service query.
2. The query is encoded using **Sentence Transformers** (`all-MiniLM-L6-v2`).
3. Cosine similarity is computed against all entries in `CustomerServiceDataSet.csv`.
4. The most relevant entry is retrieved (**RAG**).
5. **GPT-2** generates a natural language response based on the retrieved context.
6. The response is displayed in the Walmart-branded Streamlit chat interface.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web UI framework |
| **Hugging Face Transformers** | GPT-2 language model |
| **Sentence Transformers** | Semantic search & embeddings |
| **Pandas** | Dataset handling |
| **PyTorch** | Deep learning backend |
| **JSON** | Chat history persistence |

---

## 📊 Dataset

The `CustomerServiceDataSet.csv` contains Walmart-specific customer service Q&A pairs used for retrieval. The chatbot matches user queries against this dataset using semantic similarity before generating a response.

---

## 👤 Author

**Irza Ali**  
BSAI Student | SZABIST  
GitHub: [@irzairfan01-arch](https://github.com/irzairfan01-arch)

---

## 📄 License

This project is for educational purposes as part of an AI coursework project.
