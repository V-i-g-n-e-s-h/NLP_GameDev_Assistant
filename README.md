# NLP_GameDev_Assistant

## Setup (Windows)

1. Open PowerShell.
2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Upgrade pip and install the required packages:

   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the scraping script

```
python scrap.py
```

This fetches raw data and saves it locally.

### 2. Load the processed data

```
python load.py
```

This transforms the raw data and prepares it for the app.

### 3. Launch the Streamlit dashboard

```
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

## Note

- Ensure Python 3.11 or newer is on your PATH.
- **Prerequisite:** Make sure you have Ollama installed and the latest Llama 3.2 model downloaded on your system before continuing.
- Reactivate the virtual environment in each new shell with:

  ```
  .\.venv\Scripts\activate
  ```
