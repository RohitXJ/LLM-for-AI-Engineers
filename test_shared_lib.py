import sys
from pathlib import Path

# Add the current directory to sys.path to test without installing first
sys.path.append(str(Path(__file__).parent))

try:
    from shared_llm import (
        DocumentMetadata, 
        SearchFilters, 
        Chunker, 
        DataLoader, 
        ChromaManager, 
        KeywordEngine, 
        HybridFusion, 
        LocalReranker, 
        ChatManager
    )
    print("✅ All modules imported successfully from shared_llm!")

    # Test initialization (where no heavy weights are loaded)
    chunker = Chunker(chunk_size=100)
    print("✅ Chunker initialized.")
    
    kw = KeywordEngine()
    print("✅ KeywordEngine initialized.")
    
    loader = DataLoader()
    print("✅ DataLoader initialized.")

    print("\n--- Library Check Complete ---")
    print("Next step: Run 'pip install -e .' in your terminal to enable global imports.")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
