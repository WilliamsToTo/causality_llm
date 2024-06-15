# Understanding LLMs' Ability in Causal Discovery
This repository explores the factors that influence Large Language Models' (LLMs) understanding of causal discovery questions. We conduct experiments using various datasets and employ state-of-the-art LLMs to analyze the performance of LLMs in identifying causal relationships from textual data.

## Dataset
The datasets used for our experiments are stored in the `./data` folder. Please refer to the documentation within that folder for details on the structure and contents of each dataset.

## Code.
- `interact_llm.py`: Contains the implementation for LLM inference. This script is used to interact with pre-trained language models to evaluate their understanding of causal questions.
  
- `search_causal_relation.py`: Manages the retrieval of queries related to causal discovery. For more details on the retrieval process, visit [WIMBD](https://github.com/allenai/wimbd).

## Model Fine-tuning
We use the official OLMo implementation for fine-tuning the models on causal discovery tasks. The OLMo-7b-instruct model can be fine-tuned using the scripts and guidelines available at [here](https://github.com/allenai/open-instruct).

