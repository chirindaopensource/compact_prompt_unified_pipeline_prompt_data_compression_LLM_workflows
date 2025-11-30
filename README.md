# `README.md`

# CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.18043v1-b31b1b.svg)](https://arxiv.org/abs/2510.18043v1)
[![Journal](https://img.shields.io/badge/Journal-ACM%20ICAIF%202025-003366)](https://arxiv.org/abs/2510.18043v1)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows)
[![Discipline](https://img.shields.io/badge/Discipline-Computer%20Science%20%7C%20AI%20for%20Finance-00529B)](https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows)
[![Data Sources](https://img.shields.io/badge/Data-TAT--QA-lightgrey)](https://github.com/NExTplusplus/TAT-QA)
[![Data Sources](https://img.shields.io/badge/Data-FinQA-lightgrey)](https://github.com/czyssrs/FinQA)
[![Data Sources](https://img.shields.io/badge/Data-Wikipedia%20Dump-lightgrey)](https://dumps.wikimedia.org/)
[![Data Sources](https://img.shields.io/badge/Data-ShareGPT-lightgrey)](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
[![Data Sources](https://img.shields.io/badge/Data-arXiv-lightgrey)](https://arxiv.org/)
[![Core Method](https://img.shields.io/badge/Method-Hard%20Prompt%20Pruning%20%7C%20N--gram%20Abbreviation%20%7C%20Quantization-orange)](https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows)
[![Analysis](https://img.shields.io/badge/Analysis-Cost--Performance%20Trade--offs%20%7C%20Semantic%20Fidelity-red)](https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Spacy](https://img.shields.io/badge/spaCy-%2309A3D5.svg?style=flat&logo=spaCy&logoColor=white)](https://spacy.io/)
[![PyYAML](https://img.shields.io/badge/PyYAML-gray?logo=yaml&logoColor=white)](https://pyyaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows"** by:

*   Joong Ho Choi
*   Jiayang Zhao
*   Jeel Shah
*   Ritvika Sonawane
*   Vedant Singh
*   Avani Appalla
*   Will Flanagan
*   Filipe Condessa

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and offline corpus statistics generation to the core hard prompt pruning, n-gram abbreviation, numeric quantization, and comprehensive semantic fidelity evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_compactprompt_pipeline`](#key-callable-run_compactprompt_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Choi et al. (2025). The core of this repository is the iPython Notebook `compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed to be a generalizable toolkit for optimizing Large Language Model (LLM) inference costs and latency in enterprise environments.

The paper addresses the challenge of processing long, data-rich contexts—combining free-form instructions, large documents, and numeric tables—under strict cost and context-window constraints. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Construct a large offline corpus to compute static self-information scores for tokens.
-   Implement **Hard Prompt Compression** by pruning low-information phrases identified via a hybrid static/dynamic scoring mechanism.
-   Apply **Textual N-gram Abbreviation** to losslessly compress repetitive patterns in attached documents.
-   Execute **Numerical Quantization** (Uniform and K-Means) to reduce the token footprint of tabular data while bounding approximation error.
-   Select representative few-shot exemplars via embedding-based clustering to maximize prompt utility.
-   Evaluate semantic fidelity using both embedding cosine similarity and a simulated human evaluation protocol.
-   Automatically generate performance metrics and cost-savings analysis.

## Theoretical Background

The implemented methods are grounded in information theory, natural language processing, and data compression principles.

**1. Hybrid Self-Information Scoring:**
The core of the prompt pruning strategy relies on identifying the information content of each token.
-   **Static Self-Information ($I_{\text{stat}}$):** Derived from a large offline corpus (Wikipedia, ShareGPT, arXiv), capturing global token rarity: $I_{\text{stat}}(t) = -\log_2 p(t)$.
-   **Dynamic Self-Information ($s_{\text{dyn}}$):** Derived from a scorer LLM's conditional probability, capturing context-specific surprise: $s_{\text{dyn}}(t \mid c) = -\log_2 P_{\text{model}}(t \mid c)$.
-   **Fusion Rule:** A combined score $C(t)$ prioritizes dynamic information when the two metrics diverge significantly ($\Delta > 0.1$), ensuring context-critical tokens are preserved.

**2. Dependency-Driven Pruning:**
To maintain grammatical coherence, tokens are grouped into syntactic phrases (NP, VP, PP) using dependency parsing. Pruning decisions are made at the phrase level based on aggregated importance scores.

**3. Data Compression:**
-   **N-gram Abbreviation:** Frequent multi-token patterns in documents are replaced with short, unique placeholders, leveraging the heavy-tailed distribution of language in specialized domains.
-   **Quantization:** Floating-point numbers in tables are mapped to integer codes ($q_i$) or cluster centroids, reducing token count while maintaining numerical relationships within a bounded error $\varepsilon_{\max}$.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows/blob/main/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_summary.png" alt="CompactPrompt Process Summary" width="100%">
</div>

## Features

The provided iPython Notebook (`compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 35 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks the schema, content integrity, and structural consistency of TAT-QA and Fin-QA datasets.
-   **Advanced NLP Processing:** Integrates `spaCy` for dependency parsing and `tiktoken`/`transformers` for precise token alignment.
-   **Robust LLM Integration:** A unified interface for interacting with OpenAI, Anthropic, and Together AI models, supporting both generation and log-probability extraction.
-   **Comprehensive Evaluation:** Includes automated semantic similarity checks using `all-mpnet-base-v2` and a randomized, bias-mitigated protocol for LLM-based human proxy evaluation.
-   **Reproducible Artifacts:** Generates structured logs, compressed datasets, and detailed metric reports for every run.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Preprocessing (Tasks 1-6):** Ingests raw data, validates schemas, cleanses malformed entries, and normalizes numeric columns.
2.  **Corpus Statistics (Tasks 7-10):** Builds an offline corpus and computes static self-information scores for the vocabulary.
3.  **Prompt Engineering (Tasks 11-13):** Serializes tables to Markdown and constructs prompt templates with few-shot exemplars.
4.  **Dynamic Scoring (Tasks 14-18):** Configures LLM resources, retrieves log-probabilities, and computes combined importance scores.
5.  **Compression Engines (Tasks 19-27):** Executes phrase-level pruning, n-gram abbreviation, and numeric quantization.
6.  **Exemplar Selection (Tasks 28-30):** Embeds candidate examples and selects representative prototypes via K-Means clustering and silhouette optimization.
7.  **Evaluation (Tasks 31-35):** Computes embedding similarities and conducts a rigorous human-proxy evaluation to assess semantic fidelity.

## Core Components (Notebook Structure)

The `compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 35 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_compactprompt_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_compactprompt_pipeline`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between all 35 sub-tasks.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `pyyaml`, `scipy`, `spacy`, `tiktoken`, `transformers`, `sentence-transformers`, `scikit-learn`.
-   LLM Provider SDKs: `openai`, `anthropic`, `together`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows.git
    cd compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy pyyaml scipy spacy tiktoken transformers sentence-transformers scikit-learn openai anthropic together
    ```

4.  **Download spaCy model:**
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Input Data Structure

The pipeline requires two primary DataFrames (`tatqa_raw_df` and `finqa_raw_df`) containing QA examples. Each row must adhere to the following schema:
1.  **`example_id`**: Unique string identifier.
2.  **`split`**: Dataset partition ("train", "dev", "test").
3.  **`question_text`**: The natural language question.
4.  **`tables`**: List of table dictionaries (with `headers` and `rows`).
5.  **`passages`**: List of passage dictionaries (with `text`).
6.  **`answer_type`**, **`answer_value`**, **`answer_unit`**: Ground truth answer fields.

Additionally, an `offline_corpus.jsonl` file is required for static statistics, containing documents with `doc_id`, `source`, and `text`.

## Usage

The `compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `run_compactprompt_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    with open('config.yaml', 'r') as f:
        study_config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from JSON/Parquet: pd.read_json(...)
    tatqa_raw_df = ... 
    finqa_raw_df = ...
    
    # 3. Execute the entire replication study.
    output, log = run_compactprompt_pipeline(
        tatqa_raw_df=tatqa_raw_df,
        finqa_raw_df=finqa_raw_df,
        study_config=study_config,
        condition="compressed_plus_data",
        target_llm="gpt-4o",
        scorer_llm="gpt-4.1-mini"
    )
    
    # 4. Access results
    print(f"Mean Compression Ratio: {output.metrics['mean_compression_ratio']:.2f}x")
```

## Output Structure

The pipeline returns an `OrchestratorOutput` object containing all analytical artifacts:
-   **`tatqa_processed_df` / `finqa_processed_df`**: DataFrames with compressed text and quantized tables.
-   **`pruned_prompts`**: Dictionary mapping example IDs to their hard-pruned prompt text.
-   **`abbreviation_dict`**: The reversible n-gram dictionary.
-   **`quantization_results`**: Metadata and codes for all quantized columns.
-   **`representative_exemplars`**: IDs of selected few-shot prototypes.
-   **`similarity_results`**: Semantic fidelity statistics (cosine similarity).
-   **`human_evaluation`**: Results from the LLM-proxy annotation protocol.
-   **`metrics`**: Aggregate performance indicators (compression ratio, accuracy).

## Project Structure

```
compact_prompt_unified_pipeline/
│
├── compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb  # Main implementation notebook
├── config.yaml                                                                        # Master configuration file
├── requirements.txt                                                                   # Python package dependencies
│
├── compact_prompt_outputs/                                                            # Output directory (generated)
│   ├── corpus_stats/
│   ├── embeddings/
│   ├── human_eval_results/
│   └── processed_data/
│
├── LICENSE                                                                            # MIT Project License File
└── README.md                                                                          # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Compression Budgets:** `prompt_token_budget`.
-   **N-gram Settings:** `ngram_length`, `top_n_T`.
-   **Quantization:** `bit_width_b`, `num_clusters_k`.
-   **Scoring Thresholds:** `delta_relative_difference_threshold`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Adaptive Compression:** Dynamically adjusting the token budget based on query complexity.
-   **Privacy-Aware Pruning:** Integrating PII detection to prioritize removing sensitive entities.
-   **Multimodal Support:** Extending the pipeline to compress image or chart data attachments.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{choi2025compactprompt,
  title={CompactPrompt: A Unified Pipeline for Prompt and Data Compression in LLM Workflows},
  author={Choi, Joong Ho and Zhao, Jiayang and Shah, Jeel and Sonawane, Ritvika and Singh, Vedant and Appalla, Avani and Flanagan, Will and Condessa, Filipe},
  journal={arXiv preprint arXiv:2510.18043v1},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). CompactPrompt: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows
```

## Acknowledgments

-   Credit to **Joong Ho Choi et al.** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, spaCy, and Hugging Face**.

--

*This README was generated based on the structure and content of the `compact_prompt_unified_pipeline_prompt_data_compression_LLM_workflows_draft.ipynb` notebook and follows best practices for research software documentation.*
