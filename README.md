# LangChain RAG Application

A modern Retrieval-Augmented Generation (RAG) application built with LangChain, featuring document ingestion, vector search, and conversational AI.

## Features

- 📄 **Document Processing**: Support for PDF, TXT, MD, and DOCX documents
- 🔍 **Vector Search**: Efficient semantic search using ChromaDB
- 🤖 **AI Chat**: Conversational interface powered by free HuggingFace models
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
│   └── __init__.py
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
- No API keys required - using free HuggingFace models!

### 2. Quick Setup (Recommended)

```bash
# Run the automated setup script
python config/setup.py
```

### 3. Manual Setup

```bash
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

# Create optional .env file for model configuration
# Copy from env_example.txt (optional)
```

### 4. Environment Setup (Optional)

The application uses free HuggingFace models, so no API keys are required! A `.env` file is optional and can be used for model configuration:

#### **Quick Configuration Options:**

**Use Model Presets (Recommended):**
```env
# Choose from: fast, balanced, quality, max_quality, technical, mistral, mixtral, phi, gemma, gpt2, llama2
MODEL_PRESET=fast          # Microsoft Phi-2 (fast, 3GB RAM)
MODEL_PRESET=balanced      # Microsoft Phi-2 (default, 3GB RAM)
MODEL_PRESET=quality       # Mistral 7B (excellent, 8GB RAM)
MODEL_PRESET=max_quality   # Mixtral 8x7B (best, 16GB RAM)
MODEL_PRESET=technical     # Code Llama 7B (coding/docs, 8GB RAM)
MODEL_PRESET=mistral       # Mistral 7B (alternative)
MODEL_PRESET=mixtral       # Mixtral 8x7B (alternative)
MODEL_PRESET=phi           # Microsoft Phi-2 (alternative)
MODEL_PRESET=gemma         # Google Gemma 7B (8GB RAM)
```

**Use Custom Models:**
```env
CUSTOM_LLM_MODEL=your-custom-model-name
CUSTOM_EMBEDDING_MODEL=your-custom-embedding-model
```

**Advanced Settings:**
```env
TEMPERATURE=0.7            # Control response randomness
MAX_NEW_TOKENS=512         # Maximum tokens in response
LOAD_IN_8BIT=true          # Use 8-bit quantization
DEVICE_MAP=auto            # Device mapping
```

**Device Configuration (GPU/CPU):**
```env
FORCE_DEVICE=auto          # Auto-detect (default)
FORCE_DEVICE=cpu           # Force CPU usage
FORCE_DEVICE=cuda          # Force CUDA GPU usage
FORCE_DEVICE=mps           # Force Apple Silicon GPU
EMBEDDING_USE_GPU=true     # Enable GPU for embeddings
```

**View Configuration Options:**
```bash
python config/show_config.py
```

**Check Device Configuration:**
```bash
python show_devices.py
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

The application includes comprehensive functionality testing through the web interface. You can test the system by uploading documents and asking questions.
## Development

### Testing the Application
The application can be tested through the web interface by uploading documents and asking questions to verify the RAG functionality works correctly.

### Code Structure

- **`src/rag_engine/`**: Core RAG functionality and orchestration
- **`src/document_processor/`**: Document loading and text processing
- **`src/vector_store/`**: Vector database operations
- **`src/utils/`**: Helper functions and utilities
- **`src/web_interface/`**: Streamlit web application

### Adding New Features

1. Create new modules in the appropriate `src/` subdirectory
2. Update the relevant `__init__.py` files to export new classes/functions
3. Test new functionality through the web interface
4. Update documentation as needed

## Configuration

You can customize the application by modifying the following parameters:

- **Chunk size and overlap**: In `src/document_processor/processor.py`
- **Number of retrieved documents**: In `src/rag_engine/engine.py`
- **Model parameters**: In `src/rag_engine/engine.py`
- **Vector store settings**: In `src/vector_store/store.py`

## Troubleshooting

- **Model Loading Issues**: Ensure you have sufficient RAM (8GB+ recommended for HuggingFace models)
- **Memory Issues**: Reduce chunk size or use smaller documents
- **Performance**: Consider using GPU acceleration for larger document sets
- **Import Errors**: Make sure you're running from the project root directory
- **First Run**: Initial model download may take a few minutes

### GPU/Device Issues

- **CUDA Not Available**: Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- **Apple Silicon (M1/M2)**: Use `FORCE_DEVICE=mps` for GPU acceleration
- **Force CPU**: Use `FORCE_DEVICE=cpu` if GPU causes issues
- **Check Devices**: Run `python show_devices.py` to see available devices
- **Memory Issues**: Reduce model size or use `LOAD_IN_8BIT=true`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License. 