#!/usr/bin/env python
"""
Content List Insertion Demo for RAG-Anything

This script demonstrates how to bypass document parsing and directly insert
pre-parsed content lists into RAGAnything. Useful for:
1. Content from external parsers
2. Programmatically generated content
3. Cached parsing results
4. Custom content from multiple sources

Usage:
    python content_insertion_demo.py --api-key YOUR_API_KEY
    
Optional arguments:
    --base-url YOUR_BASE_URL (for custom OpenAI-compatible endpoints)
    --working-dir ./rag_storage (default: ./rag_storage)
    --image-path ./sample_image.jpg (path to sample image for demo)
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


def create_sample_content_list(image_path=None):
    """Create a sample content list for demonstration"""
    content_list = [
        {
            "type": "text",
            "text": """Introduction to Advanced RAG Systems
            
            Retrieval-Augmented Generation (RAG) systems have revolutionized how we process 
            and query large document collections. Modern RAG systems must handle not just 
            text, but also images, tables, equations, and other multimodal content.""",
            "page_idx": 0
        },
        {
            "type": "text", 
            "text": """Multimodal Content Processing Challenges
            
            Traditional RAG systems struggle with multimodal content because they were 
            designed primarily for text. Images contain visual information that cannot 
            be captured by text alone, tables have structured data relationships, and 
            mathematical equations require specialized processing.""",
            "page_idx": 1
        },
        {
            "type": "table",
            "table_body": """| System Type | Text Support | Image Support | Table Support | Equation Support |
|-------------|--------------|---------------|---------------|------------------|
| Traditional RAG | ✅ Excellent | ❌ None | ⚠️ Limited | ❌ None |
| RAG-Anything | ✅ Excellent | ✅ Full VLM | ✅ Structured | ✅ LaTeX |
| Competitor A | ✅ Good | ⚠️ Basic | ✅ Good | ❌ None |
| Competitor B | ✅ Good | ❌ None | ⚠️ Limited | ⚠️ Basic |""",
            "table_caption": ["Table 1: Comparison of RAG System Capabilities"],
            "table_footnote": ["Data as of 2024. VLM = Vision Language Model support."],
            "page_idx": 2
        },
        {
            "type": "equation",
            "latex": "\\text{RAG Score} = \\alpha \\cdot \\text{Retrieval}(q, D) + \\beta \\cdot \\text{Generation}(q, c)",
            "text": "This equation represents the RAG scoring function where q is the query, D is the document collection, and c is the retrieved context.",
            "page_idx": 3
        },
        {
            "type": "text",
            "text": """Performance Evaluation Results
            
            Our evaluation on a diverse set of multimodal documents shows significant 
            improvements in both retrieval accuracy and generation quality. The system 
            achieved a 95.2% accuracy rate on multimodal question-answering tasks.""",
            "page_idx": 4
        },
        {
            "type": "equation",
            "latex": "\\text{Accuracy} = \\frac{\\text{Correct Answers}}{\\text{Total Questions}} \\times 100\\%",
            "text": "Standard accuracy calculation formula used in our evaluation.",
            "page_idx": 4
        },
        {
            "type": "text",
            "text": """Conclusion and Future Work
            
            RAG-Anything represents a significant advancement in multimodal document 
            processing. Future work will focus on expanding support for additional 
            content types and improving processing efficiency for large-scale deployments.""",
            "page_idx": 5
        }
    ]
    
    # Add image content if image path is provided and exists
    if image_path and os.path.exists(image_path):
        image_content = {
            "type": "image",
            "img_path": str(Path(image_path).absolute()),  # Must be absolute path
            "img_caption": ["Figure 1: RAG-Anything System Architecture"],
            "img_footnote": ["This diagram shows the complete processing pipeline from document input to query response."],
            "page_idx": 1
        }
        # Insert after the first text block
        content_list.insert(2, image_content)
    
    return content_list


def create_research_paper_content():
    """Create content that simulates a research paper"""
    return [
        {
            "type": "text",
            "text": "Abstract: This paper presents a novel approach to multimodal retrieval-augmented generation...",
            "page_idx": 0
        },
        {
            "type": "text",
            "text": "1. Introduction\n\nThe rapid growth of multimodal data has created new challenges for information retrieval systems...",
            "page_idx": 1
        },
        {
            "type": "table",
            "table_body": """| Dataset | Size | Modalities | Benchmark Score |
|---------|------|------------|-----------------|
| MM-DocVQA | 5,000 | Text, Images | 87.3% |
| ScienceQA | 21,000 | Text, Diagrams | 92.1% |
| Our Dataset | 10,000 | Text, Images, Tables | 95.2% |""",
            "table_caption": ["Table 2: Dataset Comparison and Performance"],
            "page_idx": 2
        },
        {
            "type": "equation",
            "latex": "\\mathcal{L} = \\sum_{i=1}^{N} -\\log P(y_i | x_i, c_i)",
            "text": "Cross-entropy loss function for training the multimodal RAG system",
            "page_idx": 3
        }
    ]


def create_business_report_content():
    """Create content that simulates a business report"""
    return [
        {
            "type": "text",
            "text": "Executive Summary\n\nQ3 2024 has shown remarkable growth across all business units...",
            "page_idx": 0
        },
        {
            "type": "table",
            "table_body": """| Quarter | Revenue | Growth | Profit Margin |
|---------|---------|--------|---------------|
| Q1 2024 | $2.1M | +15% | 22% |
| Q2 2024 | $2.4M | +14% | 24% |
| Q3 2024 | $2.8M | +17% | 26% |""",
            "table_caption": ["Table 1: Quarterly Financial Performance"],
            "page_idx": 1
        },
        {
            "type": "text",
            "text": "Strategic Initiatives\n\nOur investment in AI-powered document processing has yielded significant returns...",
            "page_idx": 2
        }
    ]


async def demo_single_content_insertion(rag, image_path=None):
    """Demonstrate inserting a single content list"""
    print("=" * 60)
    print("📝 SINGLE CONTENT LIST INSERTION DEMO")
    print("=" * 60)
    
    content_list = create_sample_content_list(image_path)
    
    print(f"📋 Inserting content list with {len(content_list)} items:")
    for i, item in enumerate(content_list, 1):
        print(f"   {i}. {item['type']} content (page {item['page_idx']})")
    
    try:
        await rag.insert_content_list(
            content_list=content_list,
            file_path="multimodal_rag_guide.pdf",  # Reference file name
            display_stats=True
        )
        print("✅ Content list inserted successfully!")
    except Exception as e:
        print(f"❌ Error inserting content: {e}")
        return False
    
    return True


async def demo_multiple_content_insertion(rag):
    """Demonstrate inserting multiple content lists with different document IDs"""
    print("\n" + "=" * 60)
    print("📚 MULTIPLE CONTENT LISTS INSERTION DEMO")
    print("=" * 60)
    
    documents = [
        ("research_paper_2024.pdf", create_research_paper_content(), "research-paper-001"),
        ("business_report_q3.pdf", create_business_report_content(), "business-report-q3"),
    ]
    
    for file_name, content_list, doc_id in documents:
        print(f"\n📄 Inserting: {file_name} ({len(content_list)} items)")
        try:
            await rag.insert_content_list(
                content_list=content_list,
                file_path=file_name,
                doc_id=doc_id,
                display_stats=True
            )
            print(f"✅ {file_name} inserted successfully!")
        except Exception as e:
            print(f"❌ Error inserting {file_name}: {e}")


async def demo_content_queries(rag):
    """Demonstrate querying the inserted content"""
    print("\n" + "=" * 60)
    print("🔍 QUERYING INSERTED CONTENT")
    print("=" * 60)
    
    queries = [
        "What documents were inserted and what do they contain?",
        "Compare the performance metrics mentioned in the documents",
        "What mathematical formulas are discussed?",
        "What are the main findings about RAG systems?",
        "Summarize the business performance data",
        "What tables are mentioned and what data do they contain?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🤖 Query {i}: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"📋 Result: {result[:300]}..." if len(result) > 300 else f"📋 Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)


async def interactive_content_creation(rag):
    """Interactive session for creating and inserting custom content"""
    print("\n" + "=" * 60)
    print("🛠️ INTERACTIVE CONTENT CREATION")
    print("=" * 60)
    print("Create your own content list interactively!")
    print("\nCommands:")
    print("  add text [content] - Add text content")
    print("  add table [markdown_table] - Add table content") 
    print("  add equation [latex] [description] - Add equation content")
    print("  show - Show current content list")
    print("  insert [filename] - Insert content list with filename")
    print("  query [question] - Query the inserted content")
    print("  quit - Exit")
    print("-" * 60)
    
    custom_content = []
    page_idx = 0
    
    while True:
        try:
            user_input = input("\n💭 Command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            parts = user_input.split(' ', 2)
            command = parts[0].lower()
            
            if command == 'add' and len(parts) >= 3:
                content_type = parts[1].lower()
                content = parts[2]
                
                if content_type == 'text':
                    custom_content.append({
                        "type": "text",
                        "text": content,
                        "page_idx": page_idx
                    })
                    print(f"✅ Added text content (page {page_idx})")
                    
                elif content_type == 'table':
                    custom_content.append({
                        "type": "table", 
                        "table_body": content,
                        "table_caption": [f"User Table {len(custom_content) + 1}"],
                        "page_idx": page_idx
                    })
                    print(f"✅ Added table content (page {page_idx})")
                    
                elif content_type == 'equation':
                    equation_parts = content.split(' ', 1)
                    latex = equation_parts[0]
                    description = equation_parts[1] if len(equation_parts) > 1 else "User equation"
                    custom_content.append({
                        "type": "equation",
                        "latex": latex,
                        "text": description,
                        "page_idx": page_idx
                    })
                    print(f"✅ Added equation content (page {page_idx})")
                else:
                    print("❌ Unknown content type. Use: text, table, or equation")
                    continue
                    
                page_idx += 1
                
            elif command == 'show':
                if not custom_content:
                    print("📋 Content list is empty")
                else:
                    print(f"📋 Current content list ({len(custom_content)} items):")
                    for i, item in enumerate(custom_content, 1):
                        print(f"   {i}. {item['type']} (page {item['page_idx']})")
                        
            elif command == 'insert' and len(parts) >= 2:
                filename = parts[1]
                if not custom_content:
                    print("❌ No content to insert. Add some content first.")
                    continue
                    
                print(f"📤 Inserting {len(custom_content)} items as '{filename}'...")
                try:
                    await rag.insert_content_list(
                        content_list=custom_content.copy(),
                        file_path=filename,
                        display_stats=True
                    )
                    print("✅ Content inserted successfully!")
                except Exception as e:
                    print(f"❌ Error inserting content: {e}")
                    
            elif command == 'query' and len(parts) >= 2:
                question = ' '.join(parts[1:])
                print("🔍 Searching...")
                try:
                    result = await rag.aquery(question, mode="hybrid")
                    print(f"🎯 Answer: {result}")
                except Exception as e:
                    print(f"❌ Query error: {e}")
            else:
                print("❌ Unknown command or missing arguments")
                print("💡 Available commands: add, show, insert, query, quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Content List Insertion Demo for RAG-Anything")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (optional)")
    parser.add_argument("--working-dir", default="./rag_storage", 
                       help="Working directory for RAG storage")
    parser.add_argument("--image-path", default=None,
                       help="Path to sample image for demo (optional)")
    parser.add_argument("--enable-reranker", action="store_true", 
                       help="Enable reranking for better retrieval quality")
    parser.add_argument("--rerank-model", default="gpt-4o-mini",
                       help="Model to use for reranking (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Content List Insertion Demo")
    print(f"💾 Working directory: {args.working_dir}")
    if args.enable_reranker:
        print(f"🔄 Reranker enabled: {args.rerank_model}")
    else:
        print("🔄 Reranker disabled")
    if args.image_path:
        if os.path.exists(args.image_path):
            print(f"🖼️ Sample image: {args.image_path}")
        else:
            print(f"⚠️ Image not found: {args.image_path} (demo will continue without image)")
            args.image_path = None
    
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
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
    
    # Run demos
    try:
        # Demo single content insertion
        success = await demo_single_content_insertion(rag, args.image_path)
        if not success:
            return
        
        # Demo multiple content insertion
        await demo_multiple_content_insertion(rag)
        
        # Demo querying
        await demo_content_queries(rag)
        
        # Interactive content creation
        await interactive_content_creation(rag)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    print("\n👋 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())