# Drug Repurposing Framework Using Knowledge Graphs and Large Language Models

## Project Overview

This repository contains the implementation of a computational framework for drug repurposing that integrates Knowledge Graphs (KGs) and Large Language Models (LLMs). The framework identifies novel therapeutic applications for existing drugs, potentially accelerating drug discovery and reducing development costs.

### Purpose

The purpose of this research is to develop a computational framework that leverages both structured data (using Knowledge Graphs) and unstructured data (using Large Language Models) to identify new uses for existing drugs, thereby:

- Accelerating drug development timelines
- Reducing pharmaceutical research costs
- Addressing unmet medical needs
- Leveraging established safety profiles of existing drugs

## Framework Architecture

The framework consists of several interconnected components:

1. **Data Extraction** - Collection of biomedical data from PubTator and NCBI
2. **Data Preprocessing** - Cleaning and filtering data to ensure quality and relevance
3. **Knowledge Graph Construction** - Building a graph representation of drugs, diseases, and genes
4. **Embedding Generation** - Creating vector representations using TransE
5. **Link Prediction** - Identifying potential drug-disease associations
6. **Classification** - Using LLMs to categorize drugs and diseases
7. **Evidence Synthesis** - Validating predictions through LLM reasoning and web searches
8. **Visualization** - Interactive dashboard for exploring results

## Technologies Used

- **TransE**: Knowledge graph embedding model
- **DGL-KE**: Framework for efficient KG training
- **Ollama**: Local LLM deployment
- **DeepSeek r1**: Primary language model for classification and reasoning
- **Serper**: Web search API for evidence retrieval
- **SQLite**: Database for storing and querying data
- **Metabase**: Interactive dashboard for visualization
- **Python**: Primary programming language
- **GitHub**: Version control

## Dataset

The framework processes a comprehensive biomedical dataset with:

- 12,696 diseases
- 32,670 human genes
- 128,741 chemicals/drugs
- 10,683,605 relationships between entities

## Key Results

The framework successfully:

1. Generated over 1 million potential drug-disease associations
2. Identified 5,003 clinically relevant repurposing candidates (approved drugs for primary treatable diseases)
3. Highlighted 119 high-confidence predictions based on LLM reasoning and web evidence
4. Produced 5 top candidates with dual high confidence ratings from both distance-based and LLM approaches

### Top Drug Repurposing Candidates

The framework identified five high-potential drug repurposing candidates:

1. **Infliximab** for graft-versus-host disease
2. **Unfractionated Heparin Sodium** for myocardial infarction
3. **Epirubicin Hydrochloride** for non-small cell lung carcinoma
4. **Cisplatin** for non-small cell lung carcinoma
5. **(S)-Rivaroxaban** for thrombotic disease

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for model training)
- Windows Subsystem for Linux (WSL) or Linux environment

### Dependencies

```
pip install -r requirements.txt
```

### Data Setup

1. Download PubTator data from [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator/)
2. Download gene data from [NCBI Gene Database](https://www.ncbi.nlm.nih.gov/gene)
3. Place the data in the `data/raw` directory

## Usage

### Data Extraction

```python
python src/data_extraction/extract_data_pubtator.py
python src/data_extraction/extract_data_ncbi.py
```

### Data Preprocessing

```python
python src/data_preprocessing/preprocess_data_nodes.py
python src/data_preprocessing/preprocess_data_edges.py
python src/data_preprocessing/gene_info_select.py
```

### Data Transformation

```python
python src/data_transformation/transform_data_nodes.py
python src/data_transformation/transform_data_edges.py
```

### Model Training

```python
python src/ml_model/model_data_generation.py
$train_model.sh
```

### Link Prediction

```python
python src/ml_model/predict_links.py
```

### Entity Classification

```python
python src/data_postprocessing/data_generation.py
python src/data_postprocessing/disease_classification.py
python src/data_postprocessing/drugs_classification.py
```

### Evidence Synthesis

```python
python src/data_postprocessing/predictions_reasoning.py
```

## Performance Metrics

The TransE model achieved significant improvements during training:

- **Mean Rank (MR)**: Improved from 434.04 to 132.31 (69.52% reduction)
- **Mean Reciprocal Rank (MRR)**: Improved from 0.1062 to 0.2535 (138.79% increase)
- **Hits@1**: Improved to 0.1539
- **Hits@3**: Improved to 0.2787 
- **Hits@10**: Improved to 0.4573

## Limitations

- Reliance on publicly available biomedical databases with potential data quality issues
- Computational resource requirements for LLM-based classification
- TransE model limitations in handling complex biological relationships
- Potential biases in LLM classifications and web evidence
- Focus primarily on approved drugs and primary treatable diseases

## Future Work

- Explore more sophisticated embedding models (RotatE, ComplEx, DistMult)
- Incorporate additional biomedical data modalities
- Develop specialized frameworks for specific therapeutic areas
- Improve computational efficiency for LLM-based components
- Extend validation with experimental or clinical data
- Implement automated updating from new scientific literature

## Citation

If you use this framework in your research, please cite:

```
Arora, M. (2025). Developing a Framework for Drug Repurposing Using Knowledge Graphs and Large Language Models. M.Sc. Thesis, IU International University of Applied Sciences.
```

## Acknowledgments

- Prof. Dr. Markus Hemmer (Thesis Supervisor)
- IU University of Applied Sciences
- PubTator, NCBI, and other open-source data providers
- Ollama, SQLite, and Metabase tools

## Contact

For questions or collaboration opportunities, please contact:
- Manali Arora - [arora.manali@gmail.com]
