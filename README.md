# Drug Repurposing Framework

## Overview
This project aims to develop a **scalable framework** for **drug repurposing** by leveraging **Knowledge Graphs (KGs) and Large Language Models (LLMs)**. The goal is to analyze **biological relationships** (drug-disease, drug-gene) and generate **link predictions** for potential drug candidates.

## Project Workflow

### **1. Data Preparation**
- Extracting data from sources like **PubTator**, **DrugBank**, and other biomedical databases.
- Cleaning and preprocessing extracted datasets.
- Storing structured data in an **SQLite database**.

### **2. Knowledge Graph Construction**
- Transforming data for **Neo4j** integration.
- Building and visualizing **biomedical knowledge graphs**.

### **3. Machine Learning Model Development**
- **Feature Engineering**: Extracting relevant graph-based and textual features.
- **Model Selection**: Evaluating Graph Neural Networks (GNNs) and other ML techniques.
- **Hyperparameter Tuning**: Optimizing the model for accuracy.
- **Model Validation**: Using precision, recall, F1-score, ROC, and AUC.

### **4. Model Deployment & Predictions**
- Deploying the ML model for **link prediction**.
- Storing predicted drug candidates in Neo4j.
- Performing **pathway enrichment analysis** for evaluation.

### **5. User Interface & Hypothesis Testing**



## **Data Sources**
The project utilizes multiple publicly available biomedical datasets for drug repurposing analysis. Key data sources include:

- **[PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/)** – Annotated biomedical text data for named entity recognition.
- **[DrugBank](https://go.drugbank.com/)** – Comprehensive database of drug molecules, targets, interactions, and pathways.
- **[GNBR (Global Network of Biomedical Relationships)](https://zenodo.org/record/834026)** – Literature-based biomedical knowledge graph.
- **[Hetionet](https://het.io/)** – Heterogeneous network integrating diverse biological relationships.
- **[STRING](https://string-db.org/)** – Protein-protein interactions database.
- **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)** – Biomedical literature for extracting additional drug-disease and gene-disease relationships.

## **Preprocessing Steps**
To ensure high-quality data for knowledge graph construction and machine learning, the following preprocessing steps are performed:

1. **Data Cleaning**:
   - Removing duplicates and missing values.
   - Standardizing chemical and gene names using controlled vocabularies (e.g., MeSH, UniProt).
   - Normalizing textual data for NLP-based processing.

2. **Entity Resolution**:
   - Mapping drugs to standard identifiers (e.g., **DrugBank ID, ChEMBL ID**).
   - Linking gene and disease names to **NCBI Gene, OMIM, MeSH**.

3. **Feature Engineering**:
   - Extracting topological features from the **Knowledge Graph (KG)**.
   - Generating **graph embeddings** for machine learning models.
   - Creating **text-based embeddings** using **LLMs (e.g., BioBERT, SciBERT)**.

## **Knowledge Graph Construction**
The Knowledge Graph (KG) is built using **Neo4j**, capturing biological relationships such as:

- **Drug–Gene** interactions.
- **Gene–Disease** associations.
- **Drug–Disease** relationships.
- **Protein–Protein** interactions.

The graph schema is structured as follows:


