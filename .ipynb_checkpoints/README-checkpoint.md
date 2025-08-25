# Chatbot Setup and Usage Guide

This guide will walk you through the steps to set up and run the `stofsChatBot` application.

### Step 1: Clone the Repository

First, clone the project repository from GitHub using the following command:

```
git clone git@github.com:aharooniumd/stofsChatBot.git
```
### Step 2: Set Up the Environment

Navigate into the repository directory. Set up the required Conda environment using the appropriate file for your operating system. 

```
# For Linux 
cd stofsChatBot 
conda env create -f env-linux.yml 
conda activate stofschatbot
```

```
# For macOS 
cd stofsChatBot 
conda env create -f environment.yml 
conda activate stofschatbot
```

### Step 3: Download the LLM Model

Next, you will download the required LLM model using the `huggingface-cli`. The code is configured to use a quantized version of Llama-3.1-8B-Instruct (`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`), which is designed to be run locally.

While this project uses `llama-cpp` to run the model, you can use any package you desire. Just be sure the model you choose has a large enough context window for your needs.

You can grab the models from Hugging Face. The model used in the code can be found at: <https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF>.

```
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --local-dir ./
```
This command downloads the model file and saves it in the current directory.

### Step 4: Run the Chatbot

Before running the chatbot, make sure to update the **LLM model path** in both the **LLMQuery.py** and **LLMChat.py** code files to point to the location where you downloaded the model.

Once the path is updated, you can run the chatbot with this command:

```
python LLMChat.py
```

You can then enter a query, such as:

`What's the RMSD for station 8638901 on 2024-08-15?`

It may take a moment to read and analyze the data from the cloud before it prints the result.

### Additional Information

* **`LLMQuery.py` vs. `LLMChat.py`**: Use `LLMQuery.py` to see how function-calling works. Use `LLMChat.py` to have a chat with the model that includes memory and normal responses.

* **Data Retrieval**: If you want to see the data retrieval process, make sure you set the `quiet_missing` argument to `False` when calling `summarize_metric_by_date_lead`.

* **`metrics.py`**: You do not have to run `metrics.py` to make the chatbot work. However, if you want to have the data files locally, you can run this script.