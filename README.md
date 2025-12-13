# Contextual Augmentation for EL using LLMs

By:
    - Jakub Jagła
    - Łukasz Borak
    - Maksymilian Żmuda-Trzebiatowski
    - Krzysztof Bryszak

## Startup instructions

1. Install Ollama.

    - On linux you can run the following `bash` command:

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

    - On windows and Mac download an installer from the [ollama website](https://ollama.com/download)

2. Start the Ollama server:

    ```bash
    ollama serve
    ```

3. Install required Python dependencies.

    ```python
    pip install ollama pydantic torch tqdm
    pip install -e .
    ```

4. Run the pipeline.

    ```python
    # Process a single dataset (e.g., KORE50)
    python -m app.run_pipeline --model gemma3:4b --dataset KORE50

    # Process all datasets
    python -m app.run_pipeline --model gemma3:4b --all

    # Limit samples for testing
    python -m app.run_pipeline --model gemma3:4b --dataset KORE50 --limit 5

    # Adjust parallelism
    python -m app.run_pipeline --model gemma3:4b --all --max-workers 2
    ```