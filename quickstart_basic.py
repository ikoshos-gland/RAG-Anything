#!/usr/bin/env python
"""
Basic RAG-Anything Quickstart Script

This script demonstrates the simplest way to get started with RAG-Anything:
1. Process a single document
2. Ask questions about it
3. Get answers based on the document content

Usage:
    python quickstart_basic.py --file path/to/document.pdf --api-key YOUR_API_KEY
    
Optional arguments:
    --base-url YOUR_BASE_URL (for custom OpenAI-compatible endpoints)
    --parser mineru|docling (default: mineru)
    --working-dir ./rag_storage (default: ./rag_storage)
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


async def main():
    parser = argparse.ArgumentParser(description="Basic RAG-Anything Quickstart")
    parser.add_argument("--file", required=True, help="Path to document to process")
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
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found!")
        return
    
    print("🚀 Starting RAG-Anything Basic Example")
    print(f"📄 Processing file: {args.file}")
    print(f"🔧 Using parser: {args.parser}")
    print(f"💾 Working directory: {args.working_dir}")
    if args.enable_reranker:
        print(f"🔄 Reranker enabled: {args.rerank_model}")
    else:
        print("🔄 Reranker disabled")
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        parser=args.parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
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
    
    # Process the document
    print("📝 Processing document...")
    try:
        await rag.process_document_complete(
            file_path=args.file,
            output_dir="./output",
            parse_method="auto"
        )
        print("✅ Document processed successfully!")
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return
    
    # Interactive query loop
    print("\n🤖 RAG-Anything is ready! Ask questions about your document.")
    if args.enable_reranker:
        print("🔄 Reranking enabled - search results will be intelligently reordered for relevance")
    print("💡 Try questions like:")
    print("   - What is this document about?")
    print("   - What are the main findings?")
    print("   - Summarize the key points")
    print("   - What do the images/tables show?")
    print("\n💬 Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            print("🔍 Searching...")
            
            # Perform query
            result = await rag.aquery(question, mode="hybrid")
            
            print(f"\n🎯 Answer:\n{result}\n")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during query: {e}")


if __name__ == "__main__":
    asyncio.run(main())