#!/usr/bin/env python
"""
Batch Processing Demo for RAG-Anything

This script demonstrates batch processing capabilities:
1. Process multiple documents in a folder
2. Handle different file types simultaneously
3. Monitor processing progress
4. Query across all processed documents

Usage:
    python batch_processing_demo.py --folder ./documents --api-key YOUR_API_KEY
    
Optional arguments:
    --base-url YOUR_BASE_URL (for custom OpenAI-compatible endpoints)
    --working-dir ./rag_storage (default: ./rag_storage)
    --max-workers 4 (number of concurrent files to process)
    --recursive (process subfolders recursively)
    --extensions .pdf,.docx,.pptx (comma-separated file extensions)
"""

import asyncio
import argparse
import os
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import json


async def create_rerank_function(api_key: str, base_url: str = None, model: str = "gpt-4o-mini"):
    """Create a reranking function that uses LLM to rerank search results"""
    
    async def rerank_function(query: str, documents: list, top_k: int = None, top_n: int = None, **kwargs) -> list:
        """
        Rerank documents based on relevance to query using LLM
        
        Args:
            query: The search query
            documents: List of document chunks to rerank
            top_k: Number of top documents to return (if None, return all)
            top_n: Alternative parameter name for top_k (for compatibility)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        # Handle top_n parameter (LightRAG uses top_n, we originally used top_k)
        limit = top_k or top_n
        
        # Prepare documents for reranking
        doc_texts = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = doc.get('content', str(doc))
            else:
                text = str(doc)
            doc_texts.append(f"Document {i+1}: {text[:500]}...")  # Limit length
        
        # Create reranking prompt
        prompt = f"""You are a relevance scoring system. Given a query and a list of documents, score each document's relevance to the query on a scale of 0-100 (100 being most relevant).

Query: {query}

Documents:
{chr(10).join(doc_texts)}

Respond with only a JSON array of scores in order, like: [85, 92, 45, 78, ...]
The array must have exactly {len(documents)} scores."""

        try:
            response = await openai_complete_if_cache(
                model,
                prompt,
                api_key=api_key,
                base_url=base_url,
                temperature=0.1
            )
            
            # Parse scores - handle empty or malformed responses
            response_clean = response.strip()
            if not response_clean:
                print(f"⚠️  Empty reranking response, returning original order")
                return documents
                
            # Try to parse JSON, extract array if wrapped in text
            try:
                scores = json.loads(response_clean)
            except json.JSONDecodeError:
                # Try to extract JSON array from response text
                import re
                json_match = re.search(r'\[[\d\s,]+\]', response_clean)
                if json_match:
                    scores = json.loads(json_match.group())
                else:
                    print(f"⚠️  Could not parse reranking scores: {response_clean[:100]}")
                    return documents
            
            # Combine documents with scores
            scored_docs = list(zip(documents, scores))
            
            # Sort by score (highest first)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return limit if specified
            if limit:
                scored_docs = scored_docs[:limit]
            
            # Return documents with score metadata
            reranked_docs = []
            for doc, score in scored_docs:
                if isinstance(doc, dict):
                    doc['rerank_score'] = score
                    reranked_docs.append(doc)
                else:
                    reranked_docs.append({'content': doc, 'rerank_score': score})
            
            return reranked_docs
            
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}, returning original order")
            return documents
    
    return rerank_function


def get_file_stats(folder_path, extensions=None, recursive=False):
    """Get statistics about files in the folder"""
    folder = Path(folder_path)
    
    if extensions:
        ext_list = [ext.strip() for ext in extensions.split(',')]
    else:
        ext_list = ['.pdf', '.docx', '.pptx', '.xlsx', '.jpg', '.png', '.txt', '.md']
    
    files = []
    if recursive:
        for ext in ext_list:
            files.extend(folder.rglob(f'*{ext}'))
    else:
        for ext in ext_list:
            files.extend(folder.glob(f'*{ext}'))
    
    # Group by extension
    by_extension = {}
    total_size = 0
    
    for file in files:
        ext = file.suffix.lower()
        if ext not in by_extension:
            by_extension[ext] = {'count': 0, 'size': 0}
        by_extension[ext]['count'] += 1
        by_extension[ext]['size'] += file.stat().st_size
        total_size += file.stat().st_size
    
    return {
        'total_files': len(files),
        'total_size': total_size,
        'by_extension': by_extension,
        'files': files
    }


def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


async def process_documents_with_progress(rag, folder_path, **kwargs):
    """Process documents with progress monitoring"""
    print("📊 Starting batch processing...")
    
    try:
        await rag.process_folder_complete(
            folder_path=folder_path,
            **kwargs
        )
        print("✅ Batch processing completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        return False


async def query_processed_documents(rag):
    """Demo queries across all processed documents"""
    print("\n" + "="*60)
    print("🔍 QUERYING PROCESSED DOCUMENTS")
    print("="*60)
    
    # Predefined queries for batch analysis
    batch_queries = [
        "What types of documents were processed?",
        "Summarize the main topics across all documents",
        "What are the key findings from all the documents?",
        "Are there any common themes or patterns?",
        "What images or tables are mentioned in the documents?"
    ]
    
    for i, query in enumerate(batch_queries, 1):
        print(f"🤖 Query {i}: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"📋 Result: {result[:300]}..." if len(result) > 300 else f"📋 Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)


async def interactive_batch_queries(rag):
    """Interactive session for querying batch processed documents"""
    print("\n" + "="*60)
    print("💬 INTERACTIVE BATCH QUERY SESSION")
    print("="*60)
    print("Ask questions about all your processed documents!")
    print("💡 Try questions like:")
    print("   - Compare findings across different documents")
    print("   - What are the similarities between the documents?")
    print("   - Summarize information from document type X")
    print("   - Find references to [specific topic] across all docs")
    print("\n💬 Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question about all documents: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            print("🔍 Searching across all documents...")
            
            # Query with VLM enhancement for multimodal content
            result = await rag.aquery(question, mode="hybrid")
            
            print(f"\n🎯 Answer:\n{result}\n")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during query: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Batch Processing Demo for RAG-Anything")
    parser.add_argument("--folder", required=True, help="Folder containing documents to process")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (optional)")
    parser.add_argument("--working-dir", default="./rag_storage", 
                       help="Working directory for RAG storage")
    parser.add_argument("--max-workers", type=int, default=2, 
                       help="Maximum number of files to process concurrently")
    parser.add_argument("--recursive", action="store_true", 
                       help="Process subfolders recursively")
    parser.add_argument("--extensions", default=".pdf,.docx,.pptx,.xlsx,.jpg,.png,.txt,.md",
                       help="Comma-separated file extensions to process")
    parser.add_argument("--output-dir", default="./output",
                       help="Output directory for parsed content")
    parser.add_argument("--enable-reranker", action="store_true", 
                       help="Enable reranking for better retrieval quality")
    parser.add_argument("--rerank-model", default="gpt-4o-mini",
                       help="Model to use for reranking (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Validate folder exists
    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder '{args.folder}' not found!")
        return
    
    # Get file statistics
    print("🚀 Starting RAG-Anything Batch Processing Demo")
    print(f"📁 Processing folder: {args.folder}")
    print(f"🔧 Max concurrent workers: {args.max_workers}")
    print(f"🔄 Recursive processing: {'Yes' if args.recursive else 'No'}")
    print(f"📄 File extensions: {args.extensions}")
    if args.enable_reranker:
        print(f"🔄 Reranker enabled: {args.rerank_model}")
    else:
        print("🔄 Reranker disabled")
    
    # Analyze folder contents
    print("\n📊 Analyzing folder contents...")
    stats = get_file_stats(args.folder, args.extensions, args.recursive)
    
    if stats['total_files'] == 0:
        print("❌ No matching files found in the specified folder!")
        print(f"💡 Looking for files with extensions: {args.extensions}")
        return
    
    print(f"📈 Found {stats['total_files']} files ({format_size(stats['total_size'])} total)")
    print("\n📋 File breakdown:")
    for ext, info in stats['by_extension'].items():
        print(f"   {ext}: {info['count']} files ({format_size(info['size'])})")
    
    # Ask for confirmation
    print(f"\n⚠️  About to process {stats['total_files']} files. This may take some time and use API credits.")
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Processing cancelled.")
        return
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        max_concurrent_files=args.max_workers,
        recursive_folder_processing=args.recursive,
        supported_file_extensions=args.extensions.split(','),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Define model functions
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=args.api_key,
            base_url=args.base_url,
            **kwargs,
        )
    
    def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                         image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o", "", system_prompt=None, history_messages=[], 
                messages=messages, api_key=args.api_key, base_url=args.base_url, **kwargs)
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o", "", system_prompt=None, history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]} if image_data else {"role": "user", "content": prompt}
                ],
                api_key=args.api_key, base_url=args.base_url, **kwargs)
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    embedding_func = EmbeddingFunc(
        embedding_dim=3072, max_token_size=8192,
        func=lambda texts: openai_embed(
            texts, model="text-embedding-3-large", 
            api_key=args.api_key, base_url=args.base_url))
    
    # Setup reranker if enabled
    rerank_func = None
    lightrag_kwargs = {}
    if args.enable_reranker:
        print("🔄 Setting up reranker...")
        rerank_func = await create_rerank_function(
            api_key=args.api_key, 
            base_url=args.base_url,
            model=args.rerank_model
        )
        lightrag_kwargs["rerank_model_func"] = rerank_func
        lightrag_kwargs["min_rerank_score"] = 0.3  # Minimum score threshold
    
    # Initialize RAGAnything
    print("🔧 Initializing RAGAnything...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )
    
    # Process documents
    success = await process_documents_with_progress(
        rag,
        folder_path=args.folder,
        output_dir=args.output_dir,
        file_extensions=args.extensions.split(','),
        recursive=args.recursive,
        max_workers=args.max_workers
    )
    
    if success:
        # Demo queries
        await query_processed_documents(rag)
        
        # Interactive session
        await interactive_batch_queries(rag)
    else:
        print("❌ Batch processing failed. Please check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())