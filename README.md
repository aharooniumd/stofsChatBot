Be sure to download an LLM model before running the code.
Here we use Llama-cpp to run the model you can use any package you desire.
Make sure the model you are using has a large enough context window (in our case we use
a quantized version of Llama-3.1-8B-Instruct (Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)).
You can grab the models from Huggingface (link to the model used in the code:
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF).

Make sure you run metrics.py before all the others, so you will have the required files
locally (it takes a while to run).
If you want to see the data retrieval process, make sure you set quiet_missing argument to False when calling
summarize_metric_by_date_lead.