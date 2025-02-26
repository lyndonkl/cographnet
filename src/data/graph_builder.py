from pathlib import Path
import json
from typing import List, Dict
import torch
from torch_geometric.data import HeteroData
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import os
import ssl
from nltk.tokenize import sent_tokenize  # Better sentence tokenization

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
        # Configure SSL verification at all levels
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        os.environ['CURL_CA_BUNDLE'] = ''

        self.window_size = window_size
        self.alpha = alpha
        self.min_word_freq = min_word_freq
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Configure SSL verification at all levels
        os.environ['REQUESTS_CA_BUNDLE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'
        os.environ['SSL_CERT_FILE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'
        os.environ['CURL_CA_BUNDLE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'

        # Initialize stopwords with SSL verification disabled
        try:
            # Create unverified context for NLTK
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
            # Fallback to a basic set of stopwords if download fails
            self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                             'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                             'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        
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
        
        if not filtered_words:
            raise ValueError(f"Document has no words meeting minimum frequency threshold of {self.min_word_freq}")
        
        return ' '.join(filtered_words)

    def build_graph(self, text: str) -> HeteroData:
        """Build a heterogeneous graph from document text."""
        print(f"Building graph from text: {text[:100]}...")
        
        # Preprocess document
        doc_text = self._preprocess_document(text)
        print(f"Preprocessed text: {doc_text[:100]}...")
        
        if not doc_text.strip():
            raise ValueError("Document is empty after preprocessing")
        
        # Create HeteroData object
        data = HeteroData()
        
        # Get BERT embeddings for the whole text first
        text_inputs = self.tokenizer(
            doc_text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.bert(**text_inputs)
            # Get the full contextual embeddings
            full_embeddings = outputs.last_hidden_state[0]  # Shape: (num_tokens, hidden_dim)

        # Tokenized version of the full document (WordPiece tokens)
        full_tokens = self.tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])

        # Create word nodes with contextual embeddings
        doc_words = doc_text.split()
        doc_vocab = list(set(doc_words))  # Unique words in document
        doc_word_id_map = {word: idx for idx, word in enumerate(doc_vocab)}  # Preserve mapping

        # Initialize word embeddings tensor
        word_embeddings = torch.zeros(len(doc_vocab), full_embeddings.size(-1))

        # Process each unique word
        for word in doc_vocab:
            # Tokenize the word using BERT's tokenizer (WordPiece tokenization)
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
            
            # Find matching token positions in the full document's tokenized output
            token_indices = [i for i, token in enumerate(full_tokens) if token in word_tokens]

            if not token_indices:  # Skip if word tokens were not found
                continue

            # Extract and average embeddings of matched subwords
            word_emb = full_embeddings[token_indices].mean(dim=0)  # Mean pooling

            # Store the embedding in the tensor using doc_word_id_map
            word_embeddings[doc_word_id_map[word]] = word_emb

        # Store word embeddings
        data['word'].x = word_embeddings
        
        # Tokenize text into sentences
        sentences = sent_tokenize(doc_text)
        sentence_embeddings = []

        for sentence in sentences:
            # Tokenize the sentence and convert to input tensor
            sent_inputs = self.tokenizer(
                sentence,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            with torch.no_grad():
                sent_outputs = self.bert(**sent_inputs)
                token_embeddings = sent_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                ## Use [CLS] Token for Sentence Embedding
                sent_emb = token_embeddings[:, 0, :]  # Extract [CLS] token embedding

                # Store the sentence embedding
                sentence_embeddings.append(sent_emb.squeeze(0))

        # Stack sentence embeddings into a tensor
        data['sentence'].x = torch.stack(sentence_embeddings)
        
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
                    
                    # Get word indices from the unique word mapping
                    word_p_id = doc_word_id_map[word_p]
                    word_q_id = doc_word_id_map[word_q]
                    
                    if word_p_id == word_q_id:
                        continue
                    
                    # Calculate positional weight
                    rho = q
                    L = len(window)
                    pos_weight = (self.alpha * (rho/L)) + ((1 - self.alpha) * ((L - rho)/L))
                    
                    # Create forward edge (q -> p)
                    edge_key = (word_q_id, word_p_id)
                    if edge_key in word_pair_weights:
                        word_pair_weights[edge_key].append(pos_weight)
                    else:
                        word_pair_weights[edge_key] = [pos_weight]
                    
                    # Create reverse edge (p -> q)
                    rev_edge_key = (word_p_id, word_q_id)
                    if rev_edge_key in word_pair_weights:
                        word_pair_weights[rev_edge_key].append(pos_weight)
                    else:
                        word_pair_weights[rev_edge_key] = [pos_weight]
        
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
        
        # Initialize empty sentence edges and weights
        data['sentence', 'related_to', 'sentence'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['sentence', 'related_to', 'sentence'].edge_attr = torch.empty(0, dtype=torch.float)
        
        # Create sentence-sentence edges if we have multiple sentences
        if len(sentences) > 1:
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