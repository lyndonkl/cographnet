from pathlib import Path
import json
from typing import List, Dict, Tuple
import torch
from torch_geometric.data import HeteroData
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

class GraphBuilder:
    def __init__(
        self, 
        window_size: int = 3,
        alpha: float = 0.5,
        bert_model: str = 'bert-base-uncased',
        min_word_freq: int = 5
    ):
        """Initialize the graph builder with configuration parameters.
        
        Args:
            window_size: Size of sliding window for word co-occurrence
            alpha: Regulatory factor for positional weighting
            bert_model: BERT model name for embeddings
            min_word_freq: Minimum frequency for a word to be included
        """
        self.window_size = window_size
        self.alpha = alpha
        self.min_word_freq = min_word_freq
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        # Initialize stopwords
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        
    def load_documents(self, json_files: List[Path]) -> List[Dict]:
        """Load documents from JSON files."""
        documents = []
        for file_path in json_files:
            with open(file_path, 'r') as f:
                doc = json.load(f)
                if 'content' in doc:
                    documents.append(doc['content'])
        return documents

    def _calculate_sentence_edge_weight(
        self, 
        emb1: torch.Tensor, 
        emb2: torch.Tensor, 
        pos1: int, 
        pos2: int
    ) -> float:
        """Calculate edge weight between sentences."""
        # Position bias
        wgp1 = torch.tanh(torch.tensor(1.0 / (pos1 + 1)))
        wgp2 = torch.tanh(torch.tensor(1.0 / (pos2 + 1)))
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        
        return cos_sim * wgp1 * wgp2

    def _preprocess_document(self, document: str) -> str:
        """Preprocess document by removing stopwords and rare words."""
        # Count word frequencies
        word_freq = {}
        words = document.split()
        for word in words:
            word = word.lower().strip()
            if word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter words based on frequency
        filtered_words = [
            word for word in words 
            if word.lower() not in self.stop_words 
            and word_freq.get(word.lower(), 0) >= self.min_word_freq
        ]
        
        return ' '.join(filtered_words)

    def build_graph(self, document: str) -> HeteroData:
        """Build heterogeneous graph from document."""
        # Preprocess document
        document = self._preprocess_document(document)
        
        # Create HeteroData object
        data = HeteroData()
        
        # Split document into words
        doc_words = document.split()
        
        # Create vocabulary for this document (unique words)
        doc_vocab = list(set(doc_words))
        doc_word_id_map = {word: idx for idx, word in enumerate(doc_vocab)}
        
        # Create word nodes using BERT
        with torch.no_grad():
            word_inputs = self.tokenizer(
                doc_vocab,
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            word_outputs = self.bert(**word_inputs)
            data['word'].x = word_outputs.last_hidden_state[:, 0, :]
        
        # Create sliding windows
        windows = []
        if len(doc_words) <= self.window_size:
            windows.append(doc_words)
        else:
            for j in range(len(doc_words) - self.window_size + 1):
                window = doc_words[j: j + self.window_size]
                windows.append(window)
        
        # Track word pair occurrences and weights
        word_pair_weights = {}
        
        # Process each window
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_q = window[q]
                    
                    # Get word indices in the document vocabulary
                    word_p_id = doc_word_id_map[word_p]
                    word_q_id = doc_word_id_map[word_q]
                    
                    if word_p_id == word_q_id:
                        continue
                    
                    # Calculate positional weight using our formula
                    # ρ is the position of the first word (q) in window
                    # L is window length
                    rho = q
                    L = len(window)
                    pos_weight = (self.alpha * (rho/L)) + ((1 - self.alpha) * ((L - rho)/L))
                    
                    # Create forward edge (q -> p)
                    edge_key = (word_q_id, word_p_id)  # q comes before p in the window
                    if edge_key in word_pair_weights:
                        word_pair_weights[edge_key].append(pos_weight)
                    else:
                        word_pair_weights[edge_key] = [pos_weight]
                    
                    # Create reverse edge (p -> q) with same weight
                    rev_edge_key = (word_p_id, word_q_id)
                    if rev_edge_key in word_pair_weights:
                        word_pair_weights[rev_edge_key].append(pos_weight)
                    else:
                        word_pair_weights[rev_edge_key] = [pos_weight]
                
                # We don't create reverse edges with the same weight
                # because the positional relationship is directional
        
        # Create final edges and weights
        word_word_edges = []
        edge_weights = []
        
        for (word1_idx, word2_idx), weights in word_pair_weights.items():
            word_word_edges.append([word1_idx, word2_idx])
            # Average the weights for multiple occurrences
            edge_weights.append(sum(weights) / len(weights))
        
        if word_word_edges:  # Check if we have any edges
            data['word', 'co_occurs', 'word'].edge_index = torch.tensor(
                word_word_edges
            ).t()
            data['word', 'co_occurs', 'word'].edge_attr = torch.tensor(
                edge_weights
            ).float()
        
        # Create sentence nodes
        sentence_embeddings = []
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            outputs = self.bert(**inputs)
            # Use [CLS] token embedding as sentence embedding
            sentence_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze())
        
        # Create sentence nodes
        data['sentence'].x = torch.stack(sentence_embeddings)
        
        # Create sentence-sentence edges
        sentence_edges = []
        sentence_edge_weights = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # Forward edge (i -> j)
                sentence_edges.append([i, j])
                weight = self._calculate_sentence_edge_weight(
                    sentence_embeddings[i],
                    sentence_embeddings[j],
                    i,
                    j
                )
                sentence_edge_weights.append(weight)
                
                # Reverse edge (j -> i) with same weight
                sentence_edges.append([j, i])
                sentence_edge_weights.append(weight)
        
        if sentence_edges:  # Check if we have any edges
            data['sentence', 'related_to', 'sentence'].edge_index = torch.tensor(
                sentence_edges
            ).t()
            data['sentence', 'related_to', 'sentence'].edge_attr = torch.tensor(
                sentence_edge_weights
            ).float()
        
        return data

    def process_documents(self, json_files: List[Path]) -> List[HeteroData]:
        """Process multiple documents into graphs."""
        documents = self.load_documents(json_files)
        graphs = []
        
        for doc in tqdm(documents, desc="Building graphs"):
            graph = self.build_graph(doc)
            graphs.append(graph)
            
        return graphs 