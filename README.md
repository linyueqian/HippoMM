# HippoMM: Hippocampal-inspired Multimodal Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Website](https://img.shields.io/badge/Website-HippoMM-blue)](https://hippomultimodalmemory.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.10739-b31b1b.svg)](https://arxiv.org/abs/2504.10739)
[![Dataset](https://img.shields.io/badge/Dataset-HippoVlog-green)](https://github.com/linyueqian/HippoVlog/)

HippoMM is a multimodal memory system designed for **long audiovisual event understanding**, drawing inspiration from the biological processes of hippocampal memory formation and retrieval. It integrates visual, audio, and text modalities to create, consolidate, and query memories in a way that mirrors human cognitive functions.

The system aims to address the challenge of comprehending extended audiovisual experiences by implementing computational mechanisms analogous to hippocampal pattern separation, pattern completion, memory consolidation, and cross-modal associative retrieval.

![HippoMM Architecture](figs/method.png)

## Core Concepts

HippoMM implements a biologically-inspired multimodal memory system with four main processing stages:

* **Temporal Pattern Separation**: Segments continuous audiovisual streams into meaningful units based on visual changes and audio silence detection. This mimics how the hippocampus separates experiences into distinct episodes.

* **Perceptual Encoding**: Transforms raw inputs into rich multimodal representations using cross-modal features from video frames and audio segments.

* **Memory Consolidation**: Filters redundant segments through similarity-based analysis, preserving only distinct memory events to optimize storage.

* **Semantic Replay**: Generates abstract summaries of consolidated memories using Vision-Language Models, transforming detailed perceptual information into semantic representations called `ThetaEvents`.

The system handles queries through a dual-pathway memory retrieval process:

1. **Fast Retrieval**: Quickly checks semantic summaries for information
2. **Detailed Recall**: Performs deeper analysis when fast retrieval isn't sufficient, including feature search, semantic search, and temporal window localization

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/linyueqian/HippoMM.git
    cd HippoMM
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment (e.g., `conda` or `venv`).
    ```bash
    # Using conda (example)
    # conda create -n hippomm python=3.9
    # conda activate hippomm

    pip install -r requirements.txt
    ```

3.  **Configure the system:**
    * Copy the default configuration:
        ```bash
        cp config/default_config.yaml config/config.yaml
        ```
    * Edit `config/config.yaml` and update it with:
        * Paths to your foundation models (ImageBind, Whisper, QwenVL).
        * API keys and base URLs for any external services (e.g., OpenAI API for reasoning or VLM endpoints).
        * Desired storage directories (`storage.base_dir`, etc.).

## Example Usage

1.  **Process videos/audios in a folder:**
    ```bash
    python hippomm/core/batch_process.py --path /path/to/videos/or/audios --memory_store /path/to/memory_store
    ```

2.  **Query the memory system:**
    ```bash
    python hippomm/core/ask_question.py --question "What happened in the video?" --memory_store /path/to/memory_store
    ```

3.  **List available `ThetaEvents`:**
    ```bash
    python hippomm/core/ask_question.py --list --memory_store /path/to/memory_store
    ```

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@misc{lin2025hippommhippocampalinspiredmultimodalmemory,
      title={HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding}, 
      author={Yueqian Lin and Qinsi Wang and Hancheng Ye and Yuzhe Fu and Hai "Helen" Li and Yiran Chen},
      year={2025},
      eprint={2504.10739},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2504.10739}, 
}

```