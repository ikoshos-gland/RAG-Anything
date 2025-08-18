#!/usr/bin/env python
"""
Basic RAG-Anything Quickstart Script

This script demonstrates the simplest way to get started with RAG-Anything:
1. Process a single document OR multiple documents in a folder
2. Ask questions about the processed content
3. Get answers based on the document content

Usage:
    # Process a single file
    python quickstart_basic.py --file path/to/document.pdf --api-key YOUR_API_KEY
    
    # Process multiple files in a folder
    python quickstart_basic.py --folder path/to/documents --api-key YOUR_API_KEY
    
Optional arguments:
    --base-url YOUR_BASE_URL (for custom OpenAI-compatible endpoints)
    --parser mineru|docling (default: mineru)
    --working-dir ./rag_storage (default: ./rag_storage)
    --max-workers N (for folder processing, default: 2)
    --recursive (process subfolders when using --folder)
"""

import asyncio
import argparse
import os
import time
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


async def main():
    parser = argparse.ArgumentParser(description="Basic RAG-Anything Quickstart")
    
    # Create mutually exclusive group for file vs folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to single document to process")
    input_group.add_argument("--folder", help="Path to folder containing documents to process")
    
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (optional)")
    parser.add_argument("--parser", default="mineru", choices=["mineru", "docling"], 
                       help="Parser to use (default: mineru)")
    parser.add_argument("--working-dir", default="./rag_storage", 
                       help="Working directory for RAG storage")
    parser.add_argument("--enable-reranker", action="store_true", 
                       help="Enable reranking for better retrieval quality")
    parser.add_argument("--rerank-model", default="gpt-4o-mini",
                       help="Model to use for reranking (default: gpt-4o-mini)")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of concurrent files to process (default: 2)")
    parser.add_argument("--recursive", action="store_true",
                       help="Recursively process subfolders when using --folder")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output to see detailed processing steps")
    parser.add_argument("--show-progress", action="store_true",
                       help="Show detailed progress during document processing")
    
    args = parser.parse_args()
    
    # Validate input exists
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Error: File '{args.file}' not found!")
            return
        input_path = args.file
        processing_mode = "single"
    else:  # args.folder
        if not os.path.exists(args.folder):
            print(f"❌ Error: Folder '{args.folder}' not found!")
            return
        if not os.path.isdir(args.folder):
            print(f"❌ Error: '{args.folder}' is not a directory!")
            return
        input_path = args.folder
        processing_mode = "folder"
    
    print("🚀 Starting RAG-Anything Basic Example")
    if processing_mode == "single":
        print(f"📄 Processing file: {input_path}")
    else:
        print(f"📁 Processing folder: {input_path}")
        print(f"⚡ Max workers: {args.max_workers}")
        print(f"🔄 Recursive: {'Yes' if args.recursive else 'No'}")
    print(f"🔧 Using parser: {args.parser}")
    print(f"💾 Working directory: {args.working_dir}")
    if args.enable_reranker:
        print(f"🔄 Reranker enabled: {args.rerank_model}")
    else:
        print("🔄 Reranker disabled")
    if args.verbose:
        print("🔍 Verbose mode enabled - detailed processing info will be shown")
    if args.show_progress:
        print("📊 Progress tracking enabled")
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        parser=args.parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=args.max_workers,
        recursive_folder_processing=args.recursive,
        display_content_stats=args.verbose or args.show_progress,
    )
    
    # Define LLM model function
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
    
    # Define vision model function for image processing
    def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                         image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=args.api_key,
                base_url=args.base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=args.api_key,
                base_url=args.base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=args.api_key,
            base_url=args.base_url,
        ),
    )
    
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
    
    # Process the document(s)
    try:
        if processing_mode == "single":
            print("📝 Processing document...")
            if args.verbose:
                print(f"🔍 Starting document processing for: {input_path}")
                print(f"🔧 Parser: {args.parser}, Method: auto")
                print("🎯 Multimodal processing enabled: images, tables, equations")
            
            await rag.process_document_complete(
                file_path=input_path,
                output_dir="./output",
                parse_method="auto",
                display_stats=args.verbose or args.show_progress
            )
            print("✅ Document processed successfully!")
            
            if args.verbose:
                print("🔍 Processing completed - document content added to knowledge graph")
                
        else:  # folder processing
            print("📝 Processing documents in folder...")
            
            # Get list of supported files
            supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', 
                                  '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.md']
            
            if args.verbose:
                print(f"🔍 Scanning folder: {input_path}")
                print(f"🔍 Supported extensions: {', '.join(supported_extensions)}")
                print(f"🔍 Recursive scanning: {'enabled' if args.recursive else 'disabled'}")
            
            file_count = 0
            files_to_process = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        file_count += 1
                        file_path = os.path.join(root, file)
                        files_to_process.append(file_path)
                        if args.verbose:
                            print(f"  📄 Found: {file_path}")
                if not args.recursive:
                    break
            
            print(f"📊 Found {file_count} supported files to process")
            
            if args.verbose and file_count > 0:
                print(f"🔍 Processing with {args.max_workers} concurrent workers")
                print("🎯 Multimodal processing enabled for all files")
            
            await rag.process_folder_complete(
                folder_path=input_path,
                output_dir="./output",
                file_extensions=supported_extensions,
                recursive=args.recursive,
                max_workers=args.max_workers,
                display_stats=args.verbose or args.show_progress
            )
            print(f"✅ All {file_count} documents processed successfully!")
            
            if args.verbose:
                print("🔍 Batch processing completed - all documents added to unified knowledge graph")
                
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        if args.verbose:
            import traceback
            print("🔍 Full error traceback:")
            traceback.print_exc()
        return
    
    # Interactive query loop
    content_description = "document" if processing_mode == "single" else "documents"
    print(f"\n🤖 RAG-Anything is ready! Ask questions about your {content_description}.")
    if args.enable_reranker:
        print("🔄 Reranking enabled - search results will be intelligently reordered for relevance")
    print("💡 Try questions like:")
    if processing_mode == "single":
        print("   - What is this document about?")
        print("   - What are the main findings?")
        print("   - Summarize the key points")
        print("   - What do the images/tables show?")
    else:
        print("   - What are the common themes across all documents?")
        print("   - Compare the findings between documents")
        print("   - Summarize the key points from all documents")
        print("   - What patterns emerge from the data/images/tables?")
        print("   - Which document discusses [specific topic]?")
    print("\n💬 Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            if args.verbose:
                print(f"🔍 Searching for: '{question}'")
                search_start = time.time()
            else:
                print("🔍 Searching...")
            
            # Perform query
            result = await rag.aquery(question, mode="hybrid")
            
            if args.verbose:
                search_time = time.time() - search_start
                print(f"⏱️  Search completed in {search_time:.2f} seconds")
            
            print(f"\n🎯 Answer:\n{result}\n")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during query: {e}")


if __name__ == "__main__":
    asyncio.run(main())