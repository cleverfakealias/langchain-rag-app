# LangChain RAG Application

A modern Retrieval-Augmented Generation (RAG) application built with LangChain, featuring document ingestion, vector search, and conversational AI.

## Features

- 📄 **Document Processing**: Support for PDF, TXT, MD, and DOCX documents
- 🔍 **Vector Search**: Efficient semantic search using ChromaDB
- 🤖 **AI Chat**: Conversational interface powered by OpenAI
- 🌐 **Web Interface**: Beautiful Streamlit-based UI
- 📊 **Document Management**: Upload, process, and manage your documents

## Project Structure

```
langchain-rag-app/
├── src/                          # Source code
│   ├── rag_engine/              # Core RAG functionality
│   │   ├── __init__.py
│   │   └── engine.py            # Main RAG engine
│   ├── document_processor/      # Document processing
│   │   ├── __init__.py
│   │   └── processor.py         # Document loading and chunking
│   ├── vector_store/            # Vector database operations
│   │   ├── __init__.py
│   │   └── store.py             # ChromaDB integration
│   ├── utils/                   # Helper functions
│   │   ├── __init__.py
│   │   └── helpers.py           # Utility functions
│   └── web_interface/           # Streamlit web app
│       ├── __init__.py
│       └── app.py               # Main Streamlit application
├── tests/                       # Test files
│   ├── __init__.py
│   └── test_rag.py              # RAG system tests
├── config/                      # Configuration files
│   ├── __init__.py
│   └── setup.py                 # Setup script
├── docs/                        # Documentation
├── examples/                    # Example files
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8+ (latest stable version recommended)
- OpenAI API key

### 2. Quick Setup (Recommended)

```bash
# Run the automated setup script
python config/setup.py
```

### 3. Manual Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd langchain-rag-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your OpenAI API key
# Copy from env_example.txt
```

### 4. Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the Application

**Option 1: Using the launcher script (Recommended)**
```bash
# Windows
run.bat

# Unix/Linux/macOS
./run.sh

# Or using Python directly
python run.py
```

**Option 2: Using the main entry point**
```bash
python main.py
```

**Option 3: Direct Streamlit run**
```bash
streamlit run src/web_interface/app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Upload Documents**: Use the file uploader to add your documents
2. **Process Documents**: Click "Process Documents" to ingest them into the vector database
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **View History**: Check the conversation history and document status

## Testing

Run the test suite to verify everything works:

```bash
python tests/test_rag.py
```

## Development

### Running Tests
```bash
python tests/test_rag.py
```

### Code Structure

- **`src/rag_engine/`**: Core RAG functionality and orchestration
- **`src/document_processor/`**: Document loading and text processing
- **`src/vector_store/`**: Vector database operations
- **`src/utils/`**: Helper functions and utilities
- **`src/web_interface/`**: Streamlit web application

### Adding New Features

1. Create new modules in the appropriate `src/` subdirectory
2. Update the relevant `__init__.py` files to export new classes/functions
3. Add tests in the `tests/` directory
4. Update documentation as needed

## Configuration

You can customize the application by modifying the following parameters:

- **Chunk size and overlap**: In `src/document_processor/processor.py`
- **Number of retrieved documents**: In `src/rag_engine/engine.py`
- **Model parameters**: In `src/rag_engine/engine.py`
- **Vector store settings**: In `src/vector_store/store.py`

## Troubleshooting

- **OpenAI API Error**: Ensure your API key is valid and has sufficient credits
- **Memory Issues**: Reduce chunk size or use smaller documents
- **Performance**: Consider using GPU acceleration for larger document sets
- **Import Errors**: Make sure you're running from the project root directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License. 