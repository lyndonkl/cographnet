import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil
from dataclasses import dataclass
import logging
from tqdm import tqdm
import random

@dataclass
class Document:
    """Represents a parsed document with its content and category."""
    content: List[Dict[str, Any]]
    category: str
    
    def extract_text(self) -> str:
        """Extracts readable text content from the document structure."""
        text_parts = []
        
        def extract_recursive(content_item: Dict[str, Any]) -> None:
            if isinstance(content_item, dict):
                # Handle text nodes
                if content_item.get('_type') == '#text':
                    text = content_item.get('content', '').strip()
                    if text:
                        text_parts.append(text)
                # Handle nested content
                elif 'content' in content_item and isinstance(content_item['content'], list):
                    for item in content_item['content']:
                        extract_recursive(item)
            elif isinstance(content_item, list):
                for item in content_item:
                    extract_recursive(item)

        for section in self.content:
            extract_recursive(section)
            
        return ' '.join(text_parts)

class DocumentProcessor:
    """Processes and organizes documents into train/val/test sets."""
    
    def __init__(self, output_base_dir: str = 'processed_data', train_val_split: float = 0.9):
        self.output_base_dir = Path(output_base_dir)
        self.train_val_split = train_val_split
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self) -> None:
        """Creates necessary output directories."""
        for split in ['train', 'val', 'test']:
            split_dir = self.output_base_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            
    def parse_document(self, file_path: Path) -> List[Document]:
        """Parses a JSON file containing documents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Handle case where file contains array of documents
                    return [Document(doc['content'], doc['contentCategory']) 
                           for doc in data if 'content' in doc and 'contentCategory' in doc]
                elif isinstance(data, dict):
                    # Handle single document case
                    if 'content' in data and 'contentCategory' in data:
                        return [Document(data['content'], data['contentCategory'])]
                return []
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            return []

    def save_document(self, doc: Document, output_dir: Path, index: int) -> None:
        """Saves a processed document to the specified output directory."""
        output_file = output_dir / f"doc_{index}.json"
        
        processed_doc = {
            "text": doc.extract_text(),
            "category": doc.category
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, ensure_ascii=False, indent=2)

    def process_train_val_documents(self, input_dir: Path) -> None:
        """Processes documents for train/val split."""
        # Get all JSON files in the input directory
        json_files = list(input_dir.glob('**/*.json'))
        
        # Collect all documents first
        all_documents = []
        for file_path in tqdm(json_files, desc="Reading training/validation documents"):
            documents = self.parse_document(file_path)
            all_documents.extend(documents)
            
        # Shuffle documents
        random.shuffle(all_documents)
        
        # Split into train and val
        split_idx = int(len(all_documents) * self.train_val_split)
        train_docs = all_documents[:split_idx]
        val_docs = all_documents[split_idx:]
        
        # Save train documents
        for idx, doc in enumerate(tqdm(train_docs, desc="Saving training documents")):
            self.save_document(doc, self.output_base_dir / 'train', idx)
            
        # Save val documents
        for idx, doc in enumerate(tqdm(val_docs, desc="Saving validation documents")):
            self.save_document(doc, self.output_base_dir / 'val', idx)
            
        self.logger.info(f"Processed {len(train_docs)} training and {len(val_docs)} validation documents")

    def process_test_documents(self, input_dir: Path) -> None:
        """Processes test documents."""
        json_files = list(input_dir.glob('**/*.json'))
        
        doc_index = 0
        for file_path in tqdm(json_files, desc="Processing test documents"):
            documents = self.parse_document(file_path)
            for doc in documents:
                self.save_document(doc, self.output_base_dir / 'test', doc_index)
                doc_index += 1
                
        self.logger.info(f"Processed {doc_index} test documents")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    random.seed(42)

    # Initialize processor with 90/10 split
    processor = DocumentProcessor(train_val_split=0.9)
    processor.setup_directories()
    
    # Process documents
    try:
        # Process train/val documents
        train_val_dir = Path('data/train_val')
        if not train_val_dir.exists():
            logger.error(f"Train/val directory not found: {train_val_dir}")
        else:
            logger.info(f"Processing train/val documents from {train_val_dir}")
            processor.process_train_val_documents(train_val_dir)
        
        # Process test documents
        test_dir = Path('data/test')
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
        else:
            logger.info(f"Processing test documents from {test_dir}")
            processor.process_test_documents(test_dir)
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 