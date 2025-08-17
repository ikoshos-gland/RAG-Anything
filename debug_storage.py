#!/usr/bin/env python
"""
Debug script to check RAG storage consistency
"""

import os
import json
from pathlib import Path

def check_storage_consistency(working_dir="./rag_storage"):
    """Check the consistency of RAG storage"""
    storage_path = Path(working_dir)
    
    if not storage_path.exists():
        print(f"❌ Storage directory {working_dir} doesn't exist")
        return
    
    print(f"🔍 Checking storage in: {storage_path.absolute()}")
    
    # Check for key files
    files_to_check = [
        "kv_store.json",
        "vdb_chunks.json", 
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
        "doc_status.json"
    ]
    
    print("\n📁 Storage files:")
    for file_name in files_to_check:
        file_path = storage_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_name} ({size} bytes)")
            
            # Check if JSON files are valid
            if file_name.endswith('.json') and size > 0:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"     📊 Contains {len(data) if isinstance(data, dict) else 'N/A'} entries")
                except json.JSONDecodeError as e:
                    print(f"     ❌ JSON decode error: {e}")
                except Exception as e:
                    print(f"     ⚠️  Read error: {e}")
        else:
            print(f"  ❌ {file_name} (missing)")
    
    # Check for any empty files
    print("\n🔍 Checking for empty or corrupted files:")
    empty_files = []
    for file_path in storage_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size == 0:
            empty_files.append(file_path.name)
    
    if empty_files:
        print(f"  ⚠️  Found {len(empty_files)} empty files: {empty_files}")
    else:
        print("  ✅ No empty files found")
    
    print(f"\n📊 Total storage size: {sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())} bytes")

if __name__ == "__main__":
    check_storage_consistency()