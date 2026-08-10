# 🔎 RAG-Lab

### A Practical RAG Engineering & Experimentation Platform

**RAG-Lab** is a hands-on laboratory for building, evaluating, and benchmarking different **Retrieval-Augmented Generation (RAG)** architectures and their underlying components.

Instead of implementing a single RAG pipeline, this project explores the complete RAG stack:

> **Chunking → Embeddings → Retrieval → Query Transformation → Reranking → Generation → Evaluation → Observability → Experimentation**

The repository progressively evolves from foundational RAG implementations into an **interactive RAG experimentation platform** using **RAGAS, MLflow, LangSmith, Streamlit, multiple embedding models, vector databases, rerankers, and free/open LLMs**.

---

## ⭐ Why This Project?

A typical RAG project looks like:

```text
Documents
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retriever
    ↓
LLM
    ↓
Answer
```

RAG-Lab goes beyond this baseline:

```text
                         ┌──────────────────────┐
                         │       RAG-Lab        │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
        RAG Architectures      RAG Components          Evaluation
             │                      │                      │
      ┌──────┼──────┐        ┌──────┼──────┐       ┌──────┼──────┐
      │      │      │        │      │      │       │      │      │
   Basic  Advanced Specialized Embeddings DBs  Rerankers RAGAS  Metrics
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
                                    ▼
                             Experiment Tracking
                                    │
                              ┌─────┴─────┐
                              │           │
                           MLflow     LangSmith
                              │           │
                              └─────┬─────┘
                                    ▼
                              Streamlit UI
```

The goal is to answer engineering questions such as:

* Which RAG architecture performs best for a particular dataset?
* Does hybrid retrieval improve recall?
* How much does reranking improve context precision?
* Which embedding model provides the best quality/latency trade-off?
* Does Multi-Query RAG justify its additional LLM calls?
* Which vector database performs best for a particular workload?
* Which free/open LLM provides the best answer quality?
* How do retrieval quality, answer quality, latency, and token usage change together?

---

# 🚀 Key Engineering Highlights

This repository demonstrates practical experience with:

### 🧠 RAG Architecture

* Naive RAG
* Semantic RAG
* Parent Document RAG
* Hybrid RAG
* Multi-Query RAG
* Query Expansion RAG
* Reranking RAG
* Contextual Compression RAG
* Corrective RAG
* Self-RAG
* Adaptive RAG
* Graph RAG
* Multimodal RAG
* SQL RAG

### 🔍 Retrieval Engineering

* Dense vector retrieval
* Sparse/BM25 retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion
* Query transformation
* Query expansion
* Multi-query retrieval
* Cross-encoder reranking
* Context compression
* Parent-child retrieval

### 🧩 RAG Components

* Multiple embedding models
* Multiple vector databases
* Multiple rerankers
* Multiple LLMs
* Configurable retrieval parameters
* Configurable chunking strategies

### 📊 Evaluation

* RAGAS
* Faithfulness
* Context Precision
* Context Recall
* Answer Relevancy
* Retrieval metrics
* End-to-end evaluation

### 🔬 Experimentation

* MLflow experiment tracking
* Parameter tracking
* Metric tracking
* Artifact tracking
* Model/component comparisons
* Reproducible experiments

### 🔭 Observability

* LangSmith tracing
* Retrieval traces
* LLM traces
* Latency tracking
* Token usage tracking
* Failure analysis

### 🖥️ Application

* Streamlit
* Interactive RAG configuration
* Retrieval visualization
* Evaluation dashboards
* Experiment comparison

### 🛠️ Engineering

* Python
* LangChain
* LangGraph where appropriate
* Hugging Face
* Chroma
* BM25
* Pytest
* Docker
* Structured configuration
* Centralized logging

---

# 🏗️ High-Level Architecture

```text
                         ┌─────────────────────┐
                         │      Streamlit      │
                         │         UI          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    RAG Pipeline     │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
             Query Layer       Retrieval Layer    Generation
                  │                 │                 │
          ┌───────┼───────┐   ┌────┼────┐            │
          │       │       │   │         │             │
       Original Multi   Expansion Dense   Sparse      LLM
       Query    Query      │     │       │
                           │     └───┬───┘
                           │         │
                           └─────────┘
                               RRF
                                │
                                ▼
                            Reranker
                                │
                                ▼
                          Final Context
                                │
                                ▼
                               LLM
                                │
                                ▼
                              Answer
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
            RAGAS           LangSmith          MLflow
          Evaluation       Observability     Experiments
```

---

# 📚 RAG Architecture Comparison

| RAG Architecture       | Main Technique         | Primary Benefit              | Complexity | Status |
| ---------------------- | ---------------------- | ---------------------------- | ---------- | ------ |
| Naive RAG              | Dense retrieval        | Baseline RAG                 | 🟢 Low     | 🟢     |
| Semantic RAG           | Semantic chunking      | Better document segmentation | 🟢 Low     | 🟢     |
| Parent Document RAG    | Parent-child retrieval | Better context               | 🟡 Medium  | 🟢     |
| Hybrid RAG             | Dense + BM25           | Better retrieval coverage    | 🟡 Medium  | 🟢     |
| Multi-Query RAG        | Multiple queries       | Better recall                | 🟡 Medium  | 🟢     |
| Query Expansion RAG    | Query enrichment       | Better terminology matching  | 🟡 Medium  | 🟢     |
| Reranking RAG          | Cross-encoder          | Better precision             | 🟡 Medium  | 🔵     |
| Contextual Compression | Context filtering      | Reduce irrelevant context    | 🟡 Medium  | 🔵     |
| Corrective RAG         | Retrieval evaluation   | Correct poor retrieval       | 🔴 High    | 🔵     |
| Self-RAG               | Reflection             | Self-evaluation              | 🔴 High    | 🔵     |
| Adaptive RAG           | Dynamic routing        | Query-dependent retrieval    | 🔴 High    | 🔵     |
| Graph RAG              | Knowledge graph        | Relationship-aware retrieval | 🔴 High    | 🔵     |
| Multimodal RAG         | Multi-modal retrieval  | Images + text                | 🔴 High    | 🔵     |
| SQL RAG                | Structured retrieval   | Database querying            | 🔴 High    | 🔵     |

**Legend**

```text
🟢 Implemented
🔵 Planned
```

---

# 🔬 RAG Strategy Comparison

The repository deliberately separates techniques that are often incorrectly treated as the same thing.

### Hybrid RAG

Combines different retrieval paradigms:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Dense Retrieval     BM25 Retrieval
          │                 │
          └────────┬────────┘
                   ▼
                  RRF
                   │
                   ▼
              Final Context
```

### Multi-Query RAG

Generates multiple interpretations:

```text
                    Query
                      │
                      ▼
                 Query LLM
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         Q1          Q2          Q3 ... Q5
          │           │           │
          └───────────┼───────────┘
                      ▼
                  Retrieval
                      │
                      ▼
                     RRF
```

### Query Expansion RAG

Enriches the original query:

```text
Original Query
      │
      ▼
Query Expansion
      │
      ▼
Expanded Query
      │
      ▼
Retrieval
      │
      ▼
RRF with Original Retrieval
```

### Reranking RAG

Uses a second-stage model:

```text
Query
  │
  ▼
Vector Search
  │
  ▼
Top-N Candidates
  │
  ▼
Cross Encoder
  │
  ▼
Top-K Documents
  │
  ▼
LLM
```

---

# 🧠 Foundational RAG

## Naive RAG

The baseline architecture:

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector DB
    ↓
Similarity Search
    ↓
Context
    ↓
LLM
```

This implementation establishes the common RAG pipeline used throughout the repository.

---

## Semantic RAG

Instead of relying exclusively on fixed-size chunks, semantic relationships between neighboring content are considered when constructing chunks.

Focus:

* Semantic chunking
* Embedding similarity
* Context preservation
* Retrieval quality

---

## Parent Document RAG

Uses smaller child chunks for retrieval while returning larger parent documents as context.

```text
Large Parent Document
          │
          ├── Child Chunk 1
          ├── Child Chunk 2
          ├── Child Chunk 3
          └── Child Chunk 4
                  │
                  ▼
             Retrieval
                  │
                  ▼
           Matching Child
                  │
                  ▼
            Parent Document
                  │
                  ▼
                 LLM
```

---

# 🔍 Retrieval Strategies

## Hybrid RAG

Combines:

* Dense vector retrieval
* Sparse BM25 retrieval
* Reciprocal Rank Fusion

This helps address cases where semantic similarity alone may miss exact terminology, identifiers, names, or keywords.

---

## Multi-Query RAG

Generates multiple query variants using an LLM.

Current design:

```text
User Query
    ↓
LLM
    ↓
5 Query Variants
    ↓
5 Retrieval Operations
    ↓
RRF
    ↓
Final Documents
```

---

## Query Expansion RAG

Generates one enriched query containing:

* Synonyms
* Related concepts
* Domain terminology
* Important implied terms

The original and expanded retrieval results are combined using RRF.

---

# 🧪 Evaluation Framework

A central goal of RAG-Lab is moving from:

> "The answer looks good."

to:

> "The system can be quantitatively evaluated."

---

## RAGAS

The evaluation layer will use RAGAS to measure:

```text
                  RAG Response
                       │
                       ▼
                     RAGAS
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
 Faithfulness    Context Recall    Context Precision
       │
       ▼
 Answer Relevancy
```

The evaluation framework will support both individual runs and batch experiments.

---

# 📈 MLflow

MLflow will track experiments across different configurations.

Example:

```text
Experiment
──────────────────────────────────────
RAG Architecture:    Hybrid RAG
Embedding:           BGE-small
Vector DB:           Chroma
Reranker:            BGE Reranker
LLM:                 Llama
Top-K:               5

Metrics
──────────────────────────────────────
Faithfulness:        0.91
Context Recall:      0.88
Context Precision:   0.89
Latency:             1.42 sec
```

This makes it possible to compare:

```text
RAG Architecture
       ×
Embedding Model
       ×
Vector Database
       ×
Reranker
       ×
LLM
```

---

# 🔭 LangSmith

LangSmith will provide detailed traces of RAG execution.

```text
User Query
    │
    ▼
Query Transformation
    │
    ▼
Retriever
    │
    ├── Retrieved Documents
    │
    ▼
Reranker
    │
    ▼
Prompt
    │
    ▼
LLM
    │
    ▼
Final Answer
```

Useful observability signals include:

* Retrieval latency
* Number of retrieved documents
* Prompt size
* Output tokens
* LLM latency
* Intermediate outputs
* Failed operations
* End-to-end latency

---

# 🧩 Component Benchmarking

One of the major goals of RAG-Lab is to isolate individual components.

## Embedding Benchmark

```text
Dataset
   │
   ├── Model A
   ├── Model B
   ├── Model C
   └── Model D
          │
          ▼
     Evaluation
          │
   ┌──────┼──────┐
   ▼      ▼      ▼
 Recall  Latency Memory
```

---

## Vector Database Benchmark

Potential vector stores:

* Chroma
* FAISS
* Qdrant
* Weaviate
* Milvus

Comparison dimensions:

| Metric            | Description                         |
| ----------------- | ----------------------------------- |
| Retrieval Latency | Time required for search            |
| Recall            | Ability to retrieve relevant chunks |
| Filtering         | Metadata filtering capabilities     |
| Persistence       | Data persistence support            |
| Scalability       | Behavior as dataset size grows      |
| Setup Complexity  | Developer experience                |

---

## Reranker Benchmark

Compare:

```text
No Reranker
     vs
Reranker A
     vs
Reranker B
```

using:

* Context Precision
* Context Recall
* Latency
* End-to-end answer quality

---

# 🤖 LLM Benchmarking

RAG-Lab prioritizes **free/open models and accessible inference providers**.

Potential models include:

* Llama
* Qwen
* Gemma
* Mistral
* Other open-weight models

The objective is to compare models under identical retrieval conditions.

```text
                 Same Dataset
                      │
                      ▼
                Same Retrieval
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Llama        Qwen       Gemma
          │           │           │
          └───────────┼───────────┘
                      ▼
                   RAGAS
                      │
                      ▼
                MLflow Results
```

---

# 🖥️ Streamlit RAG Playground

The final application will expose RAG configuration through a Streamlit interface.

Example:

```text
┌──────────────────────────────────────────────┐
│                  RAG-Lab                     │
├──────────────────────────────────────────────┤
│                                              │
│ RAG Architecture                            │
│ [ Hybrid RAG                         ▼ ]     │
│                                              │
│ Embedding Model                             │
│ [ BGE-small                         ▼ ]     │
│                                              │
│ Vector Database                             │
│ [ Chroma                            ▼ ]     │
│                                              │
│ Reranker                                    │
│ [ BGE Reranker                      ▼ ]     │
│                                              │
│ LLM                                         │
│ [ Llama                             ▼ ]     │
│                                              │
│ Top-K                                       │
│ [ 5 ]                                        │
│                                              │
│ Query                                       │
│ ┌──────────────────────────────────────────┐ │
│ │ What is the company's leave policy?      │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│               [ Run RAG ]                    │
└──────────────────────────────────────────────┘
```

The result page will expose:

```text
Answer
──────────────────────────────────────────────
...

Retrieved Documents
──────────────────────────────────────────────
1. ...
2. ...
3. ...

Evaluation
──────────────────────────────────────────────
Faithfulness:       0.91
Context Recall:     0.88
Context Precision:  0.89

Performance
──────────────────────────────────────────────
Retrieval:          120 ms
Generation:         840 ms
Total:              960 ms
```

---

# 📊 RAG Experiment Matrix

The final experimentation framework is intended to support comparisons across multiple dimensions.

```text
                  RAG Architecture
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Retrieval      Components       Generation
          │              │              │
          │        ┌─────┼─────┐        │
          │        │     │     │        │
          │   Embeddings DB  Reranker   LLM
          │
          └──────────────┬──────────────┘
                         │
                         ▼
                      RAGAS
                         │
                         ▼
                      MLflow
                         │
                         ▼
                 Experiment Report
```

This enables experiments such as:

```text
Hybrid RAG
+
BGE Embeddings
+
Chroma
+
BGE Reranker
+
Llama
```

versus:

```text
Naive RAG
+
MiniLM Embeddings
+
FAISS
+
No Reranker
+
Qwen
```

---

# 📁 Repository Structure

```text
rag-lab/
│
├── foundational_rag/
│   ├── naive_rag/
│   ├── semantic_rag/
│   └── parent_document_rag/
│
├── retrieval_strategies/
│   ├── hybrid_rag/
│   ├── multi_query_rag/
│   ├── query_expansion_rag/
│   ├── reranking_rag/
│   └── contextual_compression_rag/
│
├── advanced_rag/
│   ├── corrective_rag/
│   ├── self_rag/
│   └── adaptive_rag/
│
├── specialized_rag/
│   ├── graph_rag/
│   ├── multimodal_rag/
│   └── sql_rag/
│
├── evaluation/
│   ├── datasets/
│   ├── ragas/
│   ├── benchmarks/
│   └── reports/
│
├── experiments/
│   ├── embeddings/
│   ├── vector_databases/
│   ├── rerankers/
│   ├── llms/
│   └── mlflow/
│
├── shared/
│   └── utils/
│       └── logging_config.py
│
├── app/
│   ├── streamlit_app.py
│   ├── components/
│   └── utils/
│
├── configs/
│   ├── embeddings.yaml
│   ├── vector_databases.yaml
│   ├── llms.yaml
│   ├── rerankers.yaml
│   └── experiments.yaml
│
├── tests/
│   ├── foundational/
│   ├── retrieval/
│   ├── advanced/
│   └── evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── rag_types.md
│   ├── retrieval_strategies.md
│   ├── evaluation.md
│   ├── embeddings.md
│   ├── vector_databases.md
│   ├── rerankers.md
│   ├── llm_comparison.md
│   └── experiments.md
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

# 🗺️ Development Roadmap

### Phase 1 — Foundations

* [x] Naive RAG
* [x] Semantic RAG
* [x] Parent Document RAG

### Phase 2 — Retrieval

* [x] Hybrid RAG
* [x] Multi-Query RAG
* [x] Query Expansion RAG
* [ ] Reranking RAG
* [ ] Contextual Compression RAG

### Phase 3 — Advanced RAG

* [ ] Corrective RAG
* [ ] Self-RAG
* [ ] Adaptive RAG

### Phase 4 — Specialized RAG

* [ ] Graph RAG
* [ ] Multimodal RAG
* [ ] SQL RAG

### Phase 5 — Evaluation

* [ ] Evaluation dataset
* [ ] RAGAS integration
* [ ] Retrieval metrics
* [ ] Generation metrics
* [ ] Automated evaluation reports

### Phase 6 — Experiment Tracking

* [ ] MLflow integration
* [ ] Parameter tracking
* [ ] Metric tracking
* [ ] Artifact tracking
* [ ] Experiment comparison

### Phase 7 — Observability

* [ ] LangSmith
* [ ] Retrieval tracing
* [ ] LLM tracing
* [ ] Token tracking
* [ ] Latency tracking
* [ ] Error analysis

### Phase 8 — Component Benchmarking

* [ ] Embedding comparison
* [ ] Vector database comparison
* [ ] Reranker comparison
* [ ] LLM comparison
* [ ] Chunking comparison

### Phase 9 — RAG Playground

* [ ] Streamlit application
* [ ] Architecture selector
* [ ] Embedding selector
* [ ] Vector DB selector
* [ ] Reranker selector
* [ ] LLM selector
* [ ] Evaluation dashboard
* [ ] Experiment comparison

### Phase 10 — Production Engineering

* [ ] Docker
* [ ] Pytest
* [ ] CI/CD
* [ ] Configuration management
* [ ] Reproducible experiments
* [ ] Performance optimization

---

# 🧪 Example Research Questions

RAG-Lab can be used to investigate questions such as:

### Retrieval

> Does hybrid retrieval improve Context Recall compared with dense retrieval?

### Chunking

> Does semantic chunking improve retrieval quality compared with fixed-size chunking?

### Query Transformation

> Does Multi-Query RAG improve recall enough to justify additional LLM calls?

### Query Expansion

> Does enriching queries with domain terminology improve retrieval of relevant documents?

### Reranking

> How much does cross-encoder reranking improve Context Precision?

### Embeddings

> Which embedding model provides the best retrieval quality/latency trade-off?

### Vector Databases

> How does retrieval latency change as the dataset grows?

### LLMs

> Which free/open LLM provides the best answer quality with identical retrieved context?

### System Design

> At what point does additional RAG complexity stop producing meaningful quality improvements?

---

# ⚙️ Getting Started

## 1. Clone

```bash
git clone <repository-url>
cd rag-lab
```

## 2. Create Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment

Create `.env` from `.env.example`.

Example:

```text
GROQ_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-lab
```

Only configure services required by the experiment you are running.

---

# ▶️ Running the Implementations

### Naive RAG

```bash
python foundational_rag/naive_rag/main.py
```

### Semantic RAG

```bash
python foundational_rag/semantic_rag/main.py
```

### Parent Document RAG

```bash
python foundational_rag/parent_document_rag/main.py
```

### Hybrid RAG

```bash
python retrieval_strategies/hybrid_rag/main.py
```

### Multi-Query RAG

```bash
python retrieval_strategies/multi_query_rag/main.py
```

### Query Expansion RAG

```bash
python retrieval_strategies/query_expansion_rag/main.py
```

### Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# 🧱 Design Principles

## 1. Understand Before Abstracting

Individual implementations remain explicit enough to understand the mechanics before common abstractions are introduced.

## 2. Measure Instead of Assuming

Architecture choices should be validated using evaluation metrics.

## 3. Reproducibility

Experiments should record:

* Dataset
* RAG architecture
* Embedding model
* Vector database
* Reranker
* LLM
* Retrieval parameters
* Evaluation metrics

## 4. Complexity Must Justify Itself

More sophisticated RAG does not automatically mean better RAG.

The project therefore evaluates:

```text
Quality
   ×
Latency
   ×
Token Usage
   ×
Cost
   ×
Complexity
```

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience with:

```text
RAG
├── Document Processing
├── Chunking
├── Embeddings
├── Vector Search
├── Sparse Search
├── Hybrid Retrieval
├── Query Transformation
├── Reranking
├── Context Compression
├── Advanced RAG
└── Specialized RAG

Evaluation
├── RAGAS
├── Retrieval Metrics
├── Generation Metrics
└── Benchmarking

Observability
├── LangSmith
├── Logging
├── Token Tracking
└── Latency Analysis

Experimentation
├── MLflow
├── Embedding Benchmarks
├── Vector DB Benchmarks
├── Reranker Benchmarks
└── LLM Benchmarks

Application Engineering
├── Streamlit
├── Configuration
├── Testing
├── Docker
└── CI/CD
```

---

# 📌 Portfolio Value

RAG-Lab is designed to demonstrate more than familiarity with a specific framework.

It demonstrates the ability to:

* Design RAG architectures
* Understand retrieval trade-offs
* Select appropriate retrieval strategies
* Evaluate retrieval quality
* Compare model components
* Instrument AI pipelines
* Track experiments
* Analyze system performance
* Build interactive AI applications
* Progress from prototype to production-oriented architecture

The central philosophy is:

> **Don't just build a RAG system. Understand why it works, measure how well it works, and determine when a different architecture works better.**

---

# 📈 Future Extensions

Potential future additions include:

* Agentic RAG
* Hybrid RAG + Graph RAG
* RAG routing
* Knowledge graph construction
* Long-context vs RAG comparison
* CAG vs RAG comparison
* RAG caching
* Semantic caching
* Query-level routing
* Adaptive retrieval depth
* Cost-aware retrieval
* Token optimization
* Production monitoring
* Online evaluation
* Continuous evaluation

---

# 👤 Author

**Harshit Gupta**

Machine Learning Engineer | Generative AI | Agentic AI | MLOps

---

## ⭐ Support

If you find the project useful, consider giving the repository a ⭐.

Contributions, suggestions, and discussions around RAG evaluation and experimentation are welcome.

---

## 📜 License

This project is licensed under the MIT License.
