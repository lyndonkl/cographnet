import json
from pathlib import Path
from typing import List, Dict, Any
import shutil
from dataclasses import dataclass
import logging
from tqdm import tqdm
import random

@dataclass
class OhsumedDocument:
    """Represents a parsed OHSUMED document with its content and category."""
    content: str
    category: str

class OhsumedDocumentProcessor:
    """Processes and organizes OHSUMED documents into train/test sets."""
    
    def __init__(self, output_base_dir: str = 'processed_data_ohsumed'):
        self.output_base_dir = Path(output_base_dir)
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self) -> None:
        """Creates necessary output directories."""
        for split in ['train', 'test']:
            split_dir = self.output_base_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            
    def parse_document(self, file_path: Path, category: str) -> OhsumedDocument:
        """Parses a text file containing an OHSUMED document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return OhsumedDocument(content=content, category=category)
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            return None

    def save_document(self, doc: OhsumedDocument, output_dir: Path, index: int) -> None:
        """Saves a processed document to the specified output directory."""
        output_file = output_dir / f"doc_{index}.json"
        
        processed_doc = {
            "text": doc.content,
            "category": doc.category
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, ensure_ascii=False, indent=2)

    def process_split(self, input_dir: Path, split: str) -> None:
        """Processes documents for a specific split (train or test)."""
        # Get all category directories
        category_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        doc_index = 0
        for category_dir in tqdm(category_dirs, desc=f"Processing {split} categories"):
            category = category_dir.name
            # Process all files in the category directory (no .txt extension filter)
            files = [f for f in category_dir.iterdir() if f.is_file()]
            
            for file_path in files:
                document = self.parse_document(file_path, category)
                if document:
                    self.save_document(document, self.output_base_dir / split, doc_index)
                    doc_index += 1
                
        self.logger.info(f"Processed {doc_index} {split} documents")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize processor
    processor = OhsumedDocumentProcessor()
    processor.setup_directories()
    
    # Process documents
    try:
        # Process train documents
        train_dir = Path('data/ohsumed/training')
        if not train_dir.exists():
            logger.error(f"Train directory not found: {train_dir}")
        else:
            logger.info(f"Processing train documents from {train_dir}")
            processor.process_split(train_dir, 'train')
        
        # Process test documents
        test_dir = Path('data/ohsumed/test')
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
        else:
            logger.info(f"Processing test documents from {test_dir}")
            processor.process_split(test_dir, 'test')
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 