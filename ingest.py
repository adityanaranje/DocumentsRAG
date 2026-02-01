"""
Ingestion CLI for the Insurance RAG System.
Supports incremental and force re-ingestion.
"""

import os
import sys
import argparse

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import IngestionPipeline
from rag.vector_store import VectorStoreManager
from ingestion.state import IngestionState
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Insurance Document Ingestion")
    parser.add_argument("--force", action="store_true", 
                       help="Force full re-ingestion of all documents")
    args = parser.parse_args()

    DOCS_DIR = "docs"
    state = IngestionState()
    pipeline = IngestionPipeline(DOCS_DIR)
    vector_manager = VectorStoreManager()

    # 1. Identify files to process
    all_files = []
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx')):
                all_files.append(os.path.join(root, file))

    files_to_process = []
    if args.force:
        files_to_process = all_files
        print("FORCE mode: Re-indexing all files with enhanced metadata.")
    else:
        for f in all_files:
            if state.is_file_changed(f):
                files_to_process.append(f)
        
    if not files_to_process:
        print("Everything is up to date. No new or modified documents found.")
        return

    print(f"Found {len(files_to_process)} documents to process.")

    # 2. Process files using enhanced pipeline with progress percentage
    all_chunks = []
    total_files = len(files_to_process)
    for idx, file_path in enumerate(files_to_process, 1):
        try:
            # Show percentage progress
            percent = (idx / total_files) * 100
            print(f"\r[{percent:5.1f}%] Processing ({idx}/{total_files}): {os.path.basename(file_path)[:40]:<40}", end="", flush=True)
            
            chunks = pipeline.process_single_file(file_path)
            all_chunks.extend(chunks)
            state.update_file(file_path)
        except Exception as e:
            print(f"\nError on {file_path}: {e}")

    print(f"Total chunks created: {len(all_chunks)}")

    # 3. Update Vector Store
    if all_chunks:
        if args.force:
            print("Creating new vector store...")
            vector_manager.create_vector_store(all_chunks)
        else:
            print("Updating existing vector store...")
            vector_manager.update_vector_store(all_chunks)
        
        state.save_state()
        
        
        print("\nIngestion complete!")


if __name__ == "__main__":
    main()
