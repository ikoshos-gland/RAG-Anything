# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-Anything is an All-in-One Multimodal RAG System built on top of LightRAG. It processes various document formats (PDFs, Office documents, images, text files) and extracts multimodal content (text, images, tables, equations) for intelligent retrieval-augmented generation.

## Common Development Commands

### Installation and Setup
```bash
# Install from PyPI
pip install raganything

# Install with optional dependencies
pip install 'raganything[all]'              # All optional features
pip install 'raganything[image]'            # Image format conversion (BMP, TIFF, GIF, WebP)
pip install 'raganything[text]'             # Text file processing (TXT, MD)

# Install from source for development
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything
pip install -e .
pip install -e '.[all]'

# Verify installation and setup environment
python setup_and_config.py --check-all
```

### Package Management
```bash
# Build and publish (for maintainers)
python setup.py sdist bdist_wheel
python -m twine upload dist/*

# Check package metadata
python -c "from raganything import __version__, __author__, __url__; print(f'v{__version__} by {__author__} - {__url__}')"
```

### Running Examples
```bash
# End-to-end processing with parser selection
python examples/raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY --parser mineru

# Direct modal processing
python examples/modalprocessors_example.py --api-key YOUR_API_KEY

# Office document parsing test (no API key required)
python examples/office_document_test.py --file path/to/document.docx

# Image format parsing test (no API key required)
python examples/image_format_test.py --file path/to/image.bmp

# Text format parsing test (no API key required)
python examples/text_format_test.py --file path/to/document.md

# Check external dependencies
python examples/office_document_test.py --check-libreoffice --file dummy
python examples/image_format_test.py --check-pillow --file dummy
python examples/text_format_test.py --check-reportlab --file dummy
```

### Development Scripts (Root Directory)
The repository includes several practical development and testing scripts in the root directory:

```bash
# Setup and configuration helper
python setup_and_config.py                    # Interactive setup guide
python setup_and_config.py --check-all        # Check all dependencies
python setup_and_config.py --create-env       # Create sample .env file
python setup_and_config.py --test-api --api-key YOUR_KEY  # Test API connection

# Basic document processing and experimentation
python quickstart_basic.py --file document.pdf --api-key YOUR_KEY
python quickstart_basic.py --file document.pdf --api-key YOUR_KEY --enable-reranker
python multimodal_query_demo.py --api-key YOUR_KEY --enable-reranker
python batch_processing_demo.py --folder ./documents --api-key YOUR_KEY
python content_insertion_demo.py --api-key YOUR_KEY

# Debug and analysis tools
python debug_storage.py                       # Analyze RAG storage contents
```

### Testing Commands
Note: This project does not have a formal test suite. Testing is done through example scripts:
- Use `examples/office_document_test.py`, `examples/image_format_test.py`, and `examples/text_format_test.py` for testing parser functionality
- These test scripts don't require API keys and only test document parsing capabilities
- Use the root-level scripts for end-to-end testing with real API integration

## Architecture Overview

### Core Components

1. **RAGAnything Main Class** (`raganything/raganything.py`)
   - Central orchestrator that inherits from QueryMixin, ProcessorMixin, and BatchMixin
   - Manages the complete document processing pipeline
   - Integrates with LightRAG for knowledge graph operations

2. **Mixins Architecture**
   - **QueryMixin** (`raganything/query.py`): Handles text and multimodal queries
   - **ProcessorMixin** (`raganything/processor.py`): Document parsing and content processing
   - **BatchMixin** (`raganything/batch.py`): Batch processing capabilities

3. **Parser System** (`raganything/parser.py`)
   - **MineruParser**: Uses MinerU for PDF and document parsing
   - **DoclingParser**: Alternative parser for Office documents and HTML
   - Configurable parser selection via `parser` parameter

4. **Modal Processors** (`raganything/modalprocessors.py`)
   - **ImageModalProcessor**: Processes image content with vision models
   - **TableModalProcessor**: Handles structured table data
   - **EquationModalProcessor**: Processes mathematical equations (LaTeX)
   - **GenericModalProcessor**: Base class for custom content types
   - **ContextExtractor**: Extracts contextual information for multimodal content

5. **Configuration System** (`raganything/config.py`)
   - Environment variable support with defaults
   - Comprehensive configuration for all processing modes
   - Backward compatibility for legacy settings

### Key Processing Workflows

1. **Document Processing Pipeline**:
   Document → Parser (MinerU/Docling) → Content Separation → Modal Processing → LightRAG Knowledge Graph

2. **Multimodal Content Flow**:
   - Text content: Direct insertion into LightRAG
   - Images: Vision model analysis + entity extraction
   - Tables: Structure analysis + data interpretation
   - Equations: LaTeX processing + conceptual mapping

3. **Query Processing**:
   - Pure text queries: Direct LightRAG search
   - VLM enhanced queries: Automatic image analysis in retrieved context
   - Multimodal queries: Integration of specific multimodal content with search results

### Dependencies and External Tools

**Required Python Dependencies** (from requirements.txt):
- `huggingface_hub`: Model access and downloading
- `lightrag-hku`: Core RAG functionality
- `mineru[core]`: Document parsing engine
- `tqdm`: Progress tracking for batch operations

**Optional Dependencies**:
- `Pillow>=10.0.0`: Extended image format support ([image] extra)
- `reportlab>=4.0.0`: Text to PDF conversion ([text] extra)
- `markdown>=3.4.0`, `weasyprint>=60.0`, `pygments>=2.10.0`: Enhanced markdown ([markdown] extra)

**External System Requirements**:
- **LibreOffice**: Required for Office document processing (.doc, .docx, .ppt, .pptx, .xls, .xlsx)
- **CUDA/GPU** (optional): For accelerated document parsing with MinerU

### Environment Configuration

Create `.env` file based on `env.example`:
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url  # Optional
OUTPUT_DIR=./output             # Default output directory
PARSER=mineru                   # Parser selection: mineru or docling
PARSE_METHOD=auto              # Parse method: auto, ocr, or txt
```

### File Structure Understanding

- `raganything/`: Main package directory
  - Core classes use mixin pattern for modular functionality
  - Each mixin handles a specific aspect (query, processing, batch operations)
- `examples/`: Comprehensive usage examples and test scripts
- `docs/`: Documentation for specific features
- `rag_storage/`: Default directory for RAG data storage (created at runtime)

### Development Notes

- **Configuration System**: Uses dataclass-based `RAGAnythingConfig` with comprehensive environment variable support (see `raganything/config.py`)
- **Extensible Architecture**: Modal processors can be extended by inheriting from `GenericModalProcessor`
- **Dual Parser Support**: Choose between MinerU (better for PDFs, OCR) and Docling (optimized for Office documents, HTML)
- **Async/Sync Operations**: Full support for both synchronous and asynchronous processing workflows
- **Intelligent Reranking**: Optional LLM-based reranking for improved search result relevance
- **Caching**: Multi-level caching for document parsing, query results, and LLM responses
- **Batch Processing**: Concurrent document processing with configurable worker limits
- **Direct Content Insertion**: Bypass document parsing by inserting pre-structured content lists
- **Context-Aware Processing**: Intelligent context extraction from surrounding document content

### Reranking System

The project includes an optional intelligent reranking system that uses LLMs to improve search result relevance:

```bash
# Enable reranking in any script
python quickstart_basic.py --file doc.pdf --api-key KEY --enable-reranker
python quickstart_basic.py --file doc.pdf --api-key KEY --enable-reranker --rerank-model gpt-4o
```

**Key Features**:
- LLM-powered relevance scoring (0-100 scale)
- Configurable rerank models (gpt-4o-mini, gpt-4o, etc.)
- Graceful fallback on failures
- Works with all query modes (hybrid, local, global, naive)