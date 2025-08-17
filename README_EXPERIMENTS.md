# 🧪 RAG-Anything Experiment Scripts

This directory contains practical, working scripts to help you experiment with RAG-Anything's capabilities. These scripts are based on the examples in the main README.md but are designed for immediate hands-on experimentation.

## 📋 Quick Start

1. **First-time setup:**
   ```bash
   python setup_and_config.py
   ```
   This will guide you through checking dependencies and creating configuration files.

2. **Basic document processing:**
   ```bash
   python quickstart_basic.py --file path/to/document.pdf --api-key YOUR_API_KEY
   ```

3. **Advanced multimodal queries:**
   ```bash
   python multimodal_query_demo.py --api-key YOUR_API_KEY
   ```

## 📚 Available Scripts

### 1. `setup_and_config.py` - Setup and Configuration Helper
**Purpose:** Check system requirements, create configuration files, test API connections

**Key Features:**
- ✅ Check Python packages and external dependencies
- 📝 Create sample .env configuration file  
- 🔗 Test OpenAI API connections
- 📄 Generate sample documents for testing
- 📖 Display quick start guide

**Usage:**
```bash
# Interactive setup
python setup_and_config.py

# Check all requirements
python setup_and_config.py --check-all

# Create .env file
python setup_and_config.py --create-env

# Test API connection
python setup_and_config.py --test-api --api-key YOUR_API_KEY
```

### 2. `quickstart_basic.py` - Basic Document Processing
**Purpose:** Simple end-to-end document processing and querying

**Key Features:**
- 📄 Process a single document (PDF, DOCX, images, etc.)
- 🤖 Interactive Q&A session
- 🔧 Configurable parser selection (MinerU/Docling)
- 💬 Real-time question answering

**Usage:**
```bash
python quickstart_basic.py --file document.pdf --api-key YOUR_API_KEY
python quickstart_basic.py --file presentation.pptx --api-key YOUR_API_KEY --parser docling

# With reranking for better search quality
python quickstart_basic.py --file document.pdf --api-key YOUR_API_KEY --enable-reranker
python quickstart_basic.py --file document.pdf --api-key YOUR_API_KEY --enable-reranker --rerank-model gpt-4o
```

**What it does:**
1. Processes your document using the specified parser
2. Extracts text, images, tables, and equations
3. Creates a knowledge graph
4. Optionally enables intelligent reranking for better search results
5. Starts an interactive session where you can ask questions

**Reranking Feature:**
- `--enable-reranker`: Activates LLM-based reranking of search results
- `--rerank-model`: Choose the model for reranking (default: gpt-4o-mini)
- Improves search quality by intelligently reordering results based on relevance
- Uses the same API as your main LLM, so no additional setup required

### 3. `multimodal_query_demo.py` - Advanced Query Capabilities
**Purpose:** Demonstrate different types of multimodal queries

**Key Features:**
- 🔍 VLM enhanced queries (automatic image analysis)
- 📊 Multimodal queries with tables and equations
- 🎯 Different query modes (hybrid, local, global, naive)
- 💬 Interactive multimodal session

**Usage:**
```bash
python multimodal_query_demo.py --api-key YOUR_API_KEY

# With reranking for better multimodal search quality
python multimodal_query_demo.py --api-key YOUR_API_KEY --enable-reranker
python multimodal_query_demo.py --api-key YOUR_API_KEY --enable-reranker --rerank-model gpt-4o
```

**Query Types Demonstrated:**
- **VLM Enhanced:** Automatically analyzes images in documents
- **Table Queries:** Include specific table data in queries
- **Equation Queries:** Include mathematical formulas
- **Mode Comparison:** Shows different retrieval strategies

### 4. `batch_processing_demo.py` - Process Multiple Documents
**Purpose:** Handle multiple documents and folders efficiently

**Key Features:**
- 📁 Process entire folders of documents
- 🔄 Recursive subfolder processing
- ⚡ Concurrent file processing
- 📊 File statistics and progress monitoring
- 🔍 Cross-document querying

**Usage:**
```bash
python batch_processing_demo.py --folder ./documents --api-key YOUR_API_KEY
python batch_processing_demo.py --folder ./research --api-key YOUR_API_KEY --max-workers 4 --recursive

# With reranking for better cross-document search quality
python batch_processing_demo.py --folder ./documents --api-key YOUR_API_KEY --enable-reranker
python batch_processing_demo.py --folder ./research --api-key YOUR_API_KEY --enable-reranker --rerank-model gpt-4o
```

**Features:**
- Automatically detects supported file types
- Shows processing statistics and estimated time
- Allows querying across all processed documents
- Handles different file formats simultaneously

### 5. `content_insertion_demo.py` - Direct Content Insertion
**Purpose:** Insert pre-parsed content without document parsing

**Key Features:**
- 📝 Bypass document parsing with pre-structured content
- 🛠️ Interactive content creation
- 📊 Support for tables, equations, images, and text
- 🔀 Multiple document simulation

**Usage:**
```bash
python content_insertion_demo.py --api-key YOUR_API_KEY
python content_insertion_demo.py --api-key YOUR_API_KEY --image-path ./sample_image.jpg

# With reranking for better content search quality
python content_insertion_demo.py --api-key YOUR_API_KEY --enable-reranker
python content_insertion_demo.py --api-key YOUR_API_KEY --enable-reranker --rerank-model gpt-4o
```

**Use Cases:**
- Content from external parsers
- Programmatically generated content
- Cached parsing results
- Custom content from multiple sources

## 🔄 Reranking Feature

All experiment scripts now support intelligent reranking to improve search result quality.

### **What is Reranking?**
Reranking uses an LLM to intelligently score and reorder search results based on relevance to your specific query, rather than relying only on vector similarity.

### **How to Enable:**
Add `--enable-reranker` to any script:
```bash
# Basic usage
python quickstart_basic.py --file doc.pdf --api-key YOUR_KEY --enable-reranker

# With custom rerank model  
python multimodal_query_demo.py --api-key YOUR_KEY --enable-reranker --rerank-model gpt-4o
```

### **Benefits:**
- 🎯 **Better relevance**: Results ordered by actual relevance to your question
- 🔄 **Smart scoring**: Each result gets a 0-100 relevance score  
- 📊 **Improved accuracy**: More precise answers from better result selection
- 🚀 **Easy integration**: Just add one flag to any script

### **Supported Models:**
- `gpt-4o-mini` (default - fast and cost-effective)
- `gpt-4o` (higher quality, more expensive)
- Any OpenAI-compatible model

### **Performance Notes:**
- Adds slight latency for scoring results
- Uses additional API calls (reranking is cost-effective with gpt-4o-mini)
- Falls back gracefully if reranking fails
- Most beneficial for complex queries or large document collections

## 🔧 Configuration

### Environment Variables
Create a `.env` file (use `setup_and_config.py --create-env`):
```bash
OPENAI_API_KEY=your_openai_api_key
PARSER=mineru
PARSE_METHOD=auto
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=true
ENABLE_EQUATION_PROCESSING=true
```

### API Keys
All scripts require an OpenAI API key. You can provide it via:
- Command line: `--api-key YOUR_API_KEY`
- Environment variable: `OPENAI_API_KEY=your_key`
- .env file: `OPENAI_API_KEY=your_key`

### Custom Endpoints
For OpenAI-compatible APIs:
```bash
python quickstart_basic.py --file doc.pdf --api-key KEY --base-url https://your-api.com/v1
```

## 🎯 Experiment Ideas

### For Beginners
1. **Start with text documents:** Use `quickstart_basic.py` with a simple PDF
2. **Try different parsers:** Compare MinerU vs Docling on the same document
3. **Basic queries:** Ask questions about document content, summaries, key points

### For Intermediate Users  
1. **Multimodal content:** Process documents with images and tables
2. **Batch processing:** Process a folder of mixed document types
3. **Query modes:** Compare different retrieval strategies (hybrid vs local vs global)

### For Advanced Users
1. **Custom content:** Use `content_insertion_demo.py` to insert programmatic content
2. **VLM analysis:** Test image analysis capabilities with visual documents
3. **Cross-document queries:** Process multiple related documents and find connections

## 🛠️ Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Install RAG-Anything
pip install raganything[all]

# Check installation
python setup_and_config.py --check-packages
```

**API Errors:**
```bash
# Test your API key
python setup_and_config.py --test-api --api-key YOUR_KEY
```

**Office Document Processing:**
```bash
# Install LibreOffice
# Ubuntu/Debian: sudo apt-get install libreoffice
# macOS: brew install --cask libreoffice
# Windows: Download from libreoffice.org

# Check installation
python setup_and_config.py --check-external
```

**Missing Dependencies:**
```bash
# Check what's missing
python setup_and_config.py --check-all

# Install optional features
pip install raganything[image,text]
```

### Performance Tips

1. **Start small:** Begin with a few small documents
2. **Use appropriate workers:** For batch processing, start with `--max-workers 2`
3. **Monitor API usage:** These scripts use OpenAI API which has usage costs
4. **Cache results:** Processed documents are cached in the working directory

## 📖 Next Steps

After experimenting with these scripts:

1. **Read the main README.md** for complete feature documentation
2. **Check CLAUDE.md** for architecture details and development guidance
3. **Explore the examples/ directory** for more specialized use cases
4. **Build your own scripts** using these as templates

## 💡 Tips for Experimentation

1. **Keep notes:** Document which documents work best with which parsers
2. **Try different content types:** Mix PDFs, Office docs, images, and text files
3. **Experiment with queries:** Test how different question styles affect results
4. **Compare modes:** See how hybrid vs local vs global retrieval differs
5. **Monitor performance:** Note processing times and API usage for optimization

Have fun experimenting with RAG-Anything! 🚀