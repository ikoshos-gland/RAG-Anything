#!/usr/bin/env python
"""
Setup and Configuration Helper for RAG-Anything

This script helps you:
1. Check system requirements and dependencies
2. Create and validate configuration files
3. Test API connections
4. Set up the environment for experimentation

Usage:
    python setup_and_config.py --check-all
    python setup_and_config.py --create-env
    python setup_and_config.py --test-api --api-key YOUR_API_KEY
"""

import os
import sys
import argparse
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Try importing required packages to check if they're installed
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    RAGANYTHING_AVAILABLE = True
except ImportError as e:
    RAGANYTHING_AVAILABLE = False
    IMPORT_ERROR = str(e)


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def check_package_installation(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed"""
    try:
        result = subprocess.run([sys.executable, "-c", f"import {package_name}"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Try to get version
            try:
                version_result = subprocess.run([sys.executable, "-c", 
                    f"import {package_name}; print(getattr({package_name}, '__version__', 'unknown'))"], 
                    capture_output=True, text=True)
                version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                return True, version
            except:
                return True, "unknown"
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def check_external_dependencies():
    """Check external system dependencies"""
    dependencies = {}
    
    # Check LibreOffice
    try:
        result = subprocess.run(["libreoffice", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            dependencies["LibreOffice"] = (True, version_line)
        else:
            dependencies["LibreOffice"] = (False, "Not found")
    except FileNotFoundError:
        dependencies["LibreOffice"] = (False, "Not installed")
    
    # Check MinerU
    try:
        result = subprocess.run(["mineru", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            dependencies["MinerU"] = (True, result.stdout.strip())
        else:
            dependencies["MinerU"] = (False, "Not working properly")
    except FileNotFoundError:
        dependencies["MinerU"] = (False, "Not installed")
    
    return dependencies


def check_python_packages():
    """Check required Python packages"""
    required_packages = [
        "raganything",
        "lightrag",
        "mineru", 
        "huggingface_hub",
        "tqdm",
        "dotenv"
    ]
    
    optional_packages = [
        "PIL",  # Pillow
        "reportlab",
        "markdown",
        "weasyprint",
        "pygments"
    ]
    
    results = {}
    
    print("🔍 Checking required packages...")
    for package in required_packages:
        installed, info = check_package_installation(package)
        results[package] = (installed, info, "required")
        status = "✅" if installed else "❌"
        print(f"   {status} {package}: {info}")
    
    print("\n🔍 Checking optional packages...")
    for package in optional_packages:
        installed, info = check_package_installation(package)
        results[package] = (installed, info, "optional")
        status = "✅" if installed else "⚠️"
        print(f"   {status} {package}: {info}")
    
    return results


def create_env_file():
    """Create a sample .env file"""
    env_content = """# RAG-Anything Configuration File
# Copy this to .env and update with your values

### API Configuration
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for custom endpoints

### RAG-Anything Basic Configuration
WORKING_DIR=./rag_storage
OUTPUT_DIR=./output
PARSER=mineru
PARSE_METHOD=auto
DISPLAY_CONTENT_STATS=true

### Multimodal Processing
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=true
ENABLE_EQUATION_PROCESSING=true

### Batch Processing
MAX_CONCURRENT_FILES=2
RECURSIVE_FOLDER_PROCESSING=true
SUPPORTED_FILE_EXTENSIONS=.pdf,.docx,.pptx,.xlsx,.jpg,.png,.txt,.md

### Context Configuration
CONTEXT_WINDOW=1
CONTEXT_MODE=page
MAX_CONTEXT_TOKENS=2000
INCLUDE_HEADERS=true
INCLUDE_CAPTIONS=true

### LLM Configuration
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0
MAX_ASYNC=4
MAX_TOKENS=32768
ENABLE_LLM_CACHE=true
TIMEOUT=240

### Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
EMBEDDING_BATCH_NUM=32
EMBEDDING_FUNC_MAX_ASYNC=16

### Logging
LOG_LEVEL=INFO
VERBOSE=false
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5

### Language Configuration
SUMMARY_LANGUAGE=English
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print(f"⚠️ .env file already exists at {env_path.absolute()}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Cancelled.")
            return False
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"✅ Created .env file at {env_path.absolute()}")
        print("💡 Please edit the .env file and add your API keys!")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


async def test_api_connection(api_key: str, base_url: str = None):
    """Test API connection with provided credentials"""
    print("🔗 Testing API connection...")
    
    try:
        # Test LLM
        print("   Testing LLM (gpt-4o-mini)...")
        llm_result = await openai_complete_if_cache(
            "gpt-4o-mini",
            "Say 'API connection successful' if you can read this.",
            api_key=api_key,
            base_url=base_url
        )
        print(f"   ✅ LLM Response: {llm_result}")
        
        # Test Embedding
        print("   Testing Embedding (text-embedding-3-large)...")
        embed_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
        embed_result = await embed_func(["Test embedding"])
        print(f"   ✅ Embedding: Generated {len(embed_result[0])}D vector")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return False


def create_sample_documents():
    """Create sample documents for testing"""
    docs_dir = Path("./sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Create sample markdown file
    sample_md = docs_dir / "sample_document.md"
    md_content = """# Sample Document for RAG-Anything Testing

## Introduction
This is a sample document to test RAG-Anything processing capabilities.

## Performance Data
| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | 95.2% | Excellent |
| Speed | 120ms | Good |
| Memory | 512MB | Acceptable |

## Mathematical Formula
The accuracy can be calculated using:
$$Accuracy = \\frac{Correct Predictions}{Total Predictions} \\times 100\\%$$

## Conclusion
This document demonstrates various content types that RAG-Anything can process.
"""
    
    with open(sample_md, 'w') as f:
        f.write(md_content)
    
    # Create sample text file
    sample_txt = docs_dir / "simple_text.txt"
    txt_content = """Simple Text Document

This is a plain text document for testing basic text processing.

Key Points:
- RAG-Anything supports multiple file formats
- Text processing is the foundation of RAG systems  
- Multimodal content enhances retrieval capabilities

End of document.
"""
    
    with open(sample_txt, 'w') as f:
        f.write(txt_content)
    
    print(f"✅ Created sample documents in {docs_dir.absolute()}")
    print(f"   - {sample_md.name}")
    print(f"   - {sample_txt.name}")
    
    return docs_dir


def display_quick_start_guide():
    """Display quick start instructions"""
    print("\n" + "="*60)
    print("🚀 QUICK START GUIDE")
    print("="*60)
    print("\n1. 📝 Edit your .env file:")
    print("   - Add your OpenAI API key")
    print("   - Adjust configuration as needed")
    
    print("\n2. 🧪 Test basic functionality:")
    print("   python quickstart_basic.py --file ./sample_documents/sample_document.md --api-key YOUR_API_KEY")
    
    print("\n3. 🔍 Try multimodal queries:")
    print("   python multimodal_query_demo.py --api-key YOUR_API_KEY")
    
    print("\n4. 📚 Process multiple documents:")
    print("   python batch_processing_demo.py --folder ./sample_documents --api-key YOUR_API_KEY")
    
    print("\n5. 🛠️ Insert custom content:")
    print("   python content_insertion_demo.py --api-key YOUR_API_KEY")
    
    print("\n💡 Tips:")
    print("   - Start with quickstart_basic.py for your first test")
    print("   - Use sample documents for initial experimentation")
    print("   - Check CLAUDE.md for detailed architecture information")
    print("   - All scripts support --help for detailed options")


def main():
    parser = argparse.ArgumentParser(description="RAG-Anything Setup and Configuration Helper")
    parser.add_argument("--check-all", action="store_true", 
                       help="Check all requirements and dependencies")
    parser.add_argument("--check-packages", action="store_true",
                       help="Check Python package installations")
    parser.add_argument("--check-external", action="store_true",
                       help="Check external dependencies (LibreOffice, MinerU)")
    parser.add_argument("--create-env", action="store_true",
                       help="Create a sample .env configuration file")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample documents for testing")
    parser.add_argument("--test-api", action="store_true",
                       help="Test API connection")
    parser.add_argument("--api-key", help="API key for testing")
    parser.add_argument("--base-url", help="Base URL for API testing")
    parser.add_argument("--quick-start", action="store_true",
                       help="Show quick start guide")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments provided, run interactive setup
        print("🔧 RAG-Anything Setup and Configuration Helper")
        print("=" * 50)
        
        # Check Python version
        py_ok, py_version = check_python_version()
        status = "✅" if py_ok else "❌"
        print(f"{status} Python Version: {py_version}")
        if not py_ok:
            print("❌ Python 3.9+ required!")
            return
        
        # Check if RAGAnything is available
        if not RAGANYTHING_AVAILABLE:
            print(f"❌ RAGAnything not available: {IMPORT_ERROR}")
            print("💡 Please install with: pip install raganything")
            return
        else:
            print("✅ RAGAnything imported successfully")
        
        print("\nChoose an option:")
        print("1. Check all requirements")
        print("2. Create .env configuration file") 
        print("3. Create sample documents")
        print("4. Test API connection")
        print("5. Show quick start guide")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            args.check_all = True
        elif choice == "2":
            args.create_env = True
        elif choice == "3":
            args.create_samples = True
        elif choice == "4":
            args.test_api = True
            args.api_key = input("Enter your API key: ").strip()
        elif choice == "5":
            args.quick_start = True
        elif choice == "6":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")
            return
    
    # Execute requested actions
    if args.check_all or args.check_packages:
        if not RAGANYTHING_AVAILABLE:
            print(f"❌ Cannot check packages: {IMPORT_ERROR}")
        else:
            check_python_packages()
    
    if args.check_all or args.check_external:
        print("\n🔍 Checking external dependencies...")
        external_deps = check_external_dependencies()
        for name, (installed, info) in external_deps.items():
            status = "✅" if installed else "❌"
            print(f"   {status} {name}: {info}")
    
    if args.create_env:
        create_env_file()
    
    if args.create_samples:
        create_sample_documents()
    
    if args.test_api:
        if not args.api_key:
            print("❌ --api-key required for API testing")
            return
        if not RAGANYTHING_AVAILABLE:
            print(f"❌ Cannot test API: {IMPORT_ERROR}")
            return
        asyncio.run(test_api_connection(args.api_key, args.base_url))
    
    if args.quick_start:
        display_quick_start_guide()


if __name__ == "__main__":
    main()