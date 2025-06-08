# BondInsights

BondInsights is an AI-powered web application that enables municipal bond insurers and analysts to upload official bond statements (PDFs), ask complex questions, perform advanced keyword searches, and generate automated, source-cited due diligence PDF reports. The app leverages Google Gemini LLM, LangChain, and Gradio for a seamless, interactive experience.

---

## Features

- **Multi-Document Support:** Upload and analyze multiple bond PDFs at once.
- **LLM-Powered Q&A:** Ask natural language questions about the uploaded documents and receive answers with source citations.
- **Advanced Search & Filtering:** Instantly search for keywords or phrases across all uploaded documents, with page-level results.
- **Automated Due Diligence Reports:** Generate 1-2 page PDF reports with clear recommendations and exact source citations.
- **Customizable Reports:** Choose the structure and focus of generated reports.
- **User-Friendly Interface:** Simple, interactive web UI built with Gradio.

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/en/latest/) (recommended)
- API keys for [Google Generative AI](https://ai.google.dev/) and [HuggingFace](https://huggingface.co/)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/BondInsights.git
    cd BondInsights
    ```

2. **Create and activate a conda environment:**
    ```sh
    conda create --name bondinsights python=3.11
    conda activate bondinsights
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your `.env` file:**
    ```
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
    GOOGLE_API_KEY=your_google_api_key
    ```

---

## Usage

1. **Start the app:**
    ```sh
    python main.py
    ```

2. **In your browser:**
    - Upload one or more bond PDF documents.
    - Ask questions or search for keywords.
    - Generate and download a due diligence report with citations.

---

## Example Use Cases

- Rapid due diligence for municipal bond insurers 
- Legal and financial document review
- Academic literature synthesis
- Corporate policy compliance checks

---

## Project Structure

```
BondInsights/
├── main.py
├── llm.py
├── load.py
├── vectorstore.py
├── requirements.txt
├── .env
└── README.md
```

---



---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Generative AI](https://ai.google.dev/)
- [Gradio](https://gradio.app/)
- [HuggingFace](https://huggingface.co/)