#!/usr/bin/env python
"""
Multimodal Query Demo for RAG-Anything

This script demonstrates advanced multimodal querying capabilities:
1. VLM Enhanced queries (automatically analyze images in retrieved context)
2. Multimodal queries with specific content (tables, equations, images)
3. Different query modes (hybrid, local, global, naive)

Usage:
    python multimodal_query_demo.py --api-key YOUR_API_KEY
    
Optional arguments:
    --base-url YOUR_BASE_URL (for custom OpenAI-compatible endpoints)
    --working-dir ./rag_storage (must contain processed documents)
"""

import asyncio
import argparse
import os

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


async def demo_vlm_enhanced_queries(rag):
    """Demonstrate VLM enhanced queries"""
    print("\n" + "="*60)
    print("🔍 VLM ENHANCED QUERIES DEMO")
    print("="*60)
    print("VLM enhanced queries automatically analyze images in retrieved context")
    print("when your documents contain images.")
    # Check if reranker is enabled
    if hasattr(rag, 'lightrag') and hasattr(rag.lightrag, 'rerank_model_func') and rag.lightrag.rerank_model_func:
        print("🔄 Reranking is enabled - results will be intelligently reordered")
    print()
    
    vlm_queries = [
        "What do the images in this document show?",
        "Analyze any charts or diagrams in the document",
        "Describe the visual content and relate it to the text",
        "What insights can be gained from the figures?"
    ]
    
    for i, query in enumerate(vlm_queries, 1):
        print(f"🤖 Query {i}: {query}")
        try:
            result = await rag.aquery(
                query,
                mode="hybrid",
                vlm_enhanced=True  # Force enable VLM enhancement
            )
            print(f"📋 Result: {result[:200]}..." if len(result) > 200 else f"📋 Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)


async def demo_multimodal_queries(rag):
    """Demonstrate multimodal queries with specific content"""
    print("\n" + "="*60)
    print("📊 MULTIMODAL QUERIES WITH SPECIFIC CONTENT DEMO")
    print("="*60)
    print("These queries include specific multimodal content for analysis.\n")
    
    # Table query example
    print("🔢 1. Table Query Example")
    table_result = await rag.aquery_with_multimodal(
        "Compare these performance metrics with the document content",
        multimodal_content=[{
            "type": "table",
            "table_data": """Method,Accuracy,Speed
RAGAnything,95.2%,120ms
Traditional,87.3%,180ms
Baseline,82.1%,250ms""",
            "table_caption": "Performance comparison table"
        }],
        mode="hybrid"
    )
    print(f"📋 Result: {table_result[:300]}..." if len(table_result) > 300 else f"📋 Result: {table_result}")
    print("-" * 40)
    
    # Equation query example
    print("🧮 2. Equation Query Example")
    equation_result = await rag.aquery_with_multimodal(
        "Explain this formula and its relevance to the document content",
        multimodal_content=[{
            "type": "equation",
            "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
            "equation_caption": "Document relevance probability formula"
        }],
        mode="hybrid"
    )
    print(f"📋 Result: {equation_result[:300]}..." if len(equation_result) > 300 else f"📋 Result: {equation_result}")
    print("-" * 40)
    
    # Complex multimodal query
    print("🔀 3. Complex Multimodal Query Example")
    complex_result = await rag.aquery_with_multimodal(
        "How do these metrics and formulas relate to the findings in the document?",
        multimodal_content=[
            {
                "type": "table",
                "table_data": """Metric,Value,Improvement
Precision,0.94,+12%
Recall,0.91,+8%
F1-Score,0.925,+10%""",
                "table_caption": "Evaluation metrics"
            },
            {
                "type": "equation",
                "latex": "F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}",
                "equation_caption": "F1-Score calculation"
            }
        ],
        mode="hybrid"
    )
    print(f"📋 Result: {complex_result[:300]}..." if len(complex_result) > 300 else f"📋 Result: {complex_result}")


async def demo_query_modes(rag):
    """Demonstrate different query modes"""
    print("\n" + "="*60)
    print("🎯 DIFFERENT QUERY MODES DEMO")
    print("="*60)
    print("RAG-Anything supports different query modes for different use cases.\n")
    
    sample_query = "What are the main findings in this document?"
    modes = [
        ("hybrid", "Combines vector similarity and graph traversal"),
        ("local", "Focuses on local graph neighborhood"),
        ("global", "Uses global graph structure"),
        ("naive", "Simple vector similarity search")
    ]
    
    for mode, description in modes:
        print(f"🔍 Mode: {mode} - {description}")
        try:
            result = await rag.aquery(sample_query, mode=mode)
            print(f"📋 Result: {result[:200]}..." if len(result) > 200 else f"📋 Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)


async def interactive_multimodal_session(rag):
    """Interactive session for custom multimodal queries"""
    print("\n" + "="*60)
    print("💬 INTERACTIVE MULTIMODAL SESSION")
    print("="*60)
    print("Create your own multimodal queries!")
    print("\nOptions:")
    print("1. Type 'vlm [question]' for VLM enhanced query")
    print("2. Type 'table [question]' to include sample table data")
    print("3. Type 'equation [question]' to include sample equation")
    print("4. Type 'normal [question]' for regular text query")
    print("5. Type 'quit' to exit")
    print("\nExample: vlm What do the images show?")
    print("Example: table How do these numbers compare to the document?")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n💭 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
            
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("❌ Please provide both type and question. Example: vlm What do the images show?")
                continue
                
            query_type, question = parts[0].lower(), parts[1]
            
            print("🔍 Processing...")
            
            if query_type == 'vlm':
                result = await rag.aquery(question, mode="hybrid", vlm_enhanced=True)
            elif query_type == 'table':
                result = await rag.aquery_with_multimodal(
                    question,
                    multimodal_content=[{
                        "type": "table",
                        "table_data": "Item,Value,Status\nAccuracy,95%,Good\nSpeed,Fast,Excellent\nCost,Low,Great",
                        "table_caption": "Sample metrics table"
                    }],
                    mode="hybrid"
                )
            elif query_type == 'equation':
                result = await rag.aquery_with_multimodal(
                    question,
                    multimodal_content=[{
                        "type": "equation",
                        "latex": "E = mc^2",
                        "equation_caption": "Einstein's mass-energy equivalence"
                    }],
                    mode="hybrid"
                )
            elif query_type == 'normal':
                result = await rag.aquery(question, mode="hybrid")
            else:
                print("❌ Unknown query type. Use: vlm, table, equation, or normal")
                continue
            
            print(f"\n🎯 Answer:\n{result}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Multimodal Query Demo for RAG-Anything")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (optional)")
    parser.add_argument("--working-dir", default="./rag_storage", 
                       help="Working directory with processed documents")
    parser.add_argument("--enable-reranker", action="store_true", 
                       help="Enable reranking for better retrieval quality")
    parser.add_argument("--rerank-model", default="gpt-4o-mini",
                       help="Model to use for reranking (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Check if working directory exists and has data
    if not os.path.exists(args.working_dir):
        print(f"❌ Error: Working directory '{args.working_dir}' not found!")
        print("💡 Please run quickstart_basic.py first to process some documents.")
        return
    
    print("🚀 Starting Multimodal Query Demo")
    print(f"💾 Using working directory: {args.working_dir}")
    if args.enable_reranker:
        print(f"🔄 Reranker enabled: {args.rerank_model}")
    else:
        print("🔄 Reranker disabled")
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Define model functions (same as basic example)
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
    
    # Ensure LightRAG storages are initialized
    print("🔄 Loading existing documents...")
    try:
        await rag._ensure_lightrag_initialized()
        print("✅ Successfully loaded existing documents!")
    except Exception as e:
        print(f"⚠️  Error initializing RAGAnything: {e}")
        print("💡 Please run quickstart_basic.py first to process some documents.")
        return
    
    # Run demos
    try:
        await demo_vlm_enhanced_queries(rag)
        await demo_multimodal_queries(rag)
        await demo_query_modes(rag)
        await interactive_multimodal_session(rag)
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    print("\n👋 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())