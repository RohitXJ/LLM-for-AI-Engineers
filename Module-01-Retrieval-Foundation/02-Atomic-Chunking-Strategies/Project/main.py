import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import tiktoken

from helper import *

DEFAULT_TOKENIZER = "o200k_base"
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx'}

def main(file_path=None, dir_path=None, strategy='recur', chunk_size=200, chunk_overlap=50, bp=95):
    # 0. Print Banner
    print_banner()

    # 1. Gather files to process
    files_to_process = []
    if file_path:
        files_to_process.append(Path(file_path))
        source_display = file_path
    else:
        dir_p = Path(dir_path)
        if not dir_p.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        files_to_process = [f for f in dir_p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        source_display = f"Directory: {dir_path} ({len(files_to_process)} files)"

    if not files_to_process:
        print_error_message("No supported files found to process.")
        return

    # 2. Extract documents
    all_documents = []
    with get_spinner("Extracting contents from files..."):
        for f_path in files_to_process:
            try:
                docs = extract_text(str(f_path))
                all_documents.extend(docs)
            except Exception as e:
                print_error_message(f"Skipping {f_path.name}: {e}")
        time.sleep(0.5)
    
    print_status_message(f"Successfully loaded [bold bright_white]{len(all_documents)}[/bold bright_white] semantic unit(s) from {source_display}.\n")
    
    # 3. Get splitter/model based on strategy
    with get_spinner(f"Initializing {strategy.upper()} chunking strategy..."):
        if strategy == 'recur':
            splitter = recurring_call(chunk_size, chunk_overlap)
        elif strategy == 'token':
            splitter = token_call(chunk_size, chunk_overlap)
        else:  # semantic
            splitter = semantic_call()
        time.sleep(0.5)
    print_status_message(f"Strategy [bold bright_white]{strategy.upper()}[/bold bright_white] loaded with Chunk Size: {chunk_size}, Overlap: {chunk_overlap}.\n")
    
    # 4. Process each document
    all_chunks = []
    
    console.print("[bold bright_yellow]Processing documents and generating chunks...[/bold bright_yellow]")
    for doc in tqdm(all_documents, desc="Processing Documents", unit="doc", bar_format=tqdm_bar_format, colour="green"):
        content = doc["content"]
        metadata = doc["metadata"]
        
        if strategy != 'semantic':
            chunk_texts = splitter.split_text(content)
        else:
            # Use the new adaptive percentile threshold
            chunk_texts = semantic_chunking(splitter=splitter, TEXT=content, percentile_threshold=bp)
        
        for i, text in enumerate(chunk_texts):
            chunk_obj = {
                "chunk_id": f"{metadata['source']}_p{metadata.get('page', 1)}_c{i+1}",
                "content": text,
                "metadata": {
                    **metadata,
                    "chunk_index": i + 1
                }
            }
            all_chunks.append(chunk_obj)

    console.print()

    # 5. Save to JSON
    output_file = "processed_chunks.json"
    with get_spinner("Exporting chunks to JSON...", spinner_style="bouncingBar"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4)
        time.sleep(0.5)
    print_success_message(f"Generation complete! Data saved to [bold bright_white]{output_file}[/bold bright_white]")

    # 6. Print Summary
    encoder = tiktoken.get_encoding(DEFAULT_TOKENIZER)
    total_tokens = sum(len(encoder.encode(c["content"])) for c in all_chunks)
    total_chars = sum(len(c["content"]) for c in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    avg_chars = total_chars / len(all_chunks) if all_chunks else 0
    
    print_summary_table(
        file_path=source_display, 
        strategy=strategy, 
        docs_len=len(all_documents), 
        chunks_len=len(all_chunks), 
        total_chars=total_chars, 
        total_tokens=total_tokens, 
        avg_chars=avg_chars, 
        avg_tokens=avg_tokens
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHUNKER-PRO CLI: Advanced Text Chunker")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a single input file")
    group.add_argument("-d", "--dir", type=str, help="Path to a directory for batch processing")
    
    parser.add_argument("-s", "--strategy", type=str, default="recur", 
                        choices=["recur", "token", "semantic"],
                        help="Chunking strategy: recur, token, or semantic")
    
    parser.add_argument("-cs", "--chunk_size", type=int, default=200, help="Chunk size (Default: 200)")
    parser.add_argument("-co", "--chunk_overlap", type=int, default=50, help="Overlap size (Default: 50)")
    parser.add_argument("-bp", "--breakpoint", type=int, default=95, help="Percentile for semantic splits (Default: 95)")
    
    args = parser.parse_args()
    
    try:
        main(
            file_path=args.file,
            dir_path=args.dir,
            strategy=args.strategy, 
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap,
            bp=args.breakpoint
        )
    except Exception as e:
        try:
            print_error_message(str(e))
        except:
            print(f"Error: {e}")
