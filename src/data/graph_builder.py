from pathlib import Path
import json
from typing import List, Dict
import torch
from torch_geometric.data import HeteroData
from transformers import BertTokenizerFast, BertModel, pipeline
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import os
import ssl
from collections import defaultdict
import torch.nn.functional as F

class GraphBuilder:
    def __init__(
        self, 
        window_size: int = 5,
        alpha: float = 0.5,
        bert_model: str = 'dmis-lab/biobert-base-cased-v1.1',
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
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.window_size = window_size
        self.alpha = alpha
        self.min_word_freq = min_word_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically use GPU if available
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model).to(self.device)
        
        # Configure SSL verification at all levels
        os.environ['REQUESTS_CA_BUNDLE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'
        os.environ['SSL_CERT_FILE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'
        os.environ['CURL_CA_BUNDLE'] = '/Users/kdsouza/Documents/Projects/cacerts/ZscalerRoot.pem'

        # Medical Stopword List (Better than NLTK's)
        self.stop_words = {
            "a", "about", "after", "again", "against", "all", "almost", "also", "although",
            "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
            "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
            "done", "due", "during", "each", "either", "few", "for", "from", "further", "had",
            "has", "have", "having", "here", "how", "however", "if", "in", "into", "is", "it",
            "its", "itself", "just", "may", "might", "more", "most", "must", "my", "nor", "not",
            "now", "of", "on", "once", "only", "or", "other", "ought", "our", "out", "over",
            "same", "should", "some", "such", "than", "that", "the", "their", "them", "then",
            "there", "these", "they", "this", "those", "through", "thus", "to", "too", "under",
            "until", "very", "was", "were", "what", "when", "where", "whether", "which", "while",
            "who", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet"
        }
        
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
        words = document.lower().split()
        
        # Remove only frequent stopwords, but keep rare words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words
        ]

        if not filtered_words:
            raise ValueError("Document is empty after preprocessing!")

        return ' '.join(filtered_words)

    def get_word_embeddings(self, doc_text, tokenizer, model, max_length=512, stride=256):
        """Computes word embeddings using a sliding window approach while preserving full-word mappings.

        Args:
            doc_text (str): Full text of the document.
            tokenizer (BertTokenizerFast): Tokenizer instance.
            model (BertModel): Pretrained BERT model.
            max_length (int): Maximum number of tokens per chunk.
            stride (int): Overlapping stride between chunks.

        Returns:
            torch.Tensor: Word embeddings.
            dict: doc_word_id_map - Unique words mapped to indices.
            list: ordered_words - List of words in original order (including duplicates).
        """
        # Tokenize document with offset mapping
        encoded = tokenizer(
            doc_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=False,
            truncation=False  # We manually handle chunking
        )

        input_ids = encoded["input_ids"][0]  
        attention_mask = encoded["attention_mask"][0]
        offset_mapping = encoded["offset_mapping"][0].tolist()

        # Convert token IDs to words using offset mapping
        word_to_token_indices = defaultdict(list)  # Map full words to their token indices
        token_idx_to_word = {}  # Map each token index to its corresponding full word
        full_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        ordered_words = []  # Stores words in original order, including duplicates

        # Build a correct mapping of words to token indices
        reconstructed_word = ""
        current_indices = []

        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Skip special tokens like CLS, SEP
                continue
            
            token = full_tokens[i]
            
            # If it's a subword (starts with ##), append to the previous part
            if token.startswith("##"):
                reconstructed_word += token[2:]  # Remove '##' and append
            else:
                # Store the last reconstructed word before starting a new one
                if reconstructed_word:
                    word_to_token_indices[reconstructed_word].extend(current_indices)
                    ordered_words.append(reconstructed_word)

                # Start a new word
                reconstructed_word = token
                current_indices = []

            current_indices.append(i)  # Store token index
            
            # Map each token position back to the full word
            token_idx_to_word[i] = reconstructed_word

        # Store the last reconstructed word
        if reconstructed_word:
            word_to_token_indices[reconstructed_word].extend(current_indices)
            ordered_words.append(reconstructed_word)

        # Track word embeddings across chunks
        word_embeddings = defaultdict(list)  # Collect embeddings for each word

        # Create sliding windows
        with torch.no_grad():
            for start in range(0, len(input_ids), stride):
                end = min(start + max_length, len(input_ids))

                chunk_input_ids = input_ids[start:end].unsqueeze(0)
                chunk_attention_mask = attention_mask[start:end].unsqueeze(0)

                # Run through BERT
                outputs = model(
                    input_ids=chunk_input_ids.to(model.device),
                    attention_mask=chunk_attention_mask.to(model.device)
                )
                chunk_embeddings = outputs.last_hidden_state[0]  # Shape: (chunk_length, hidden_dim)

                # Assign embeddings to full words
                for token_idx in range(start, end):
                    word = token_idx_to_word.get(token_idx, None)
                    if word and word in word_to_token_indices:
                        word_embeddings[word].append(chunk_embeddings[token_idx - start])

                if end == len(input_ids):  # Stop if we've processed the full text
                    break

        # Aggregate multiple embeddings per word (Mean Pooling)
        final_word_embeddings = {}
        for word, embeddings in word_embeddings.items():
            final_word_embeddings[word] = torch.mean(torch.stack(embeddings), dim=0)

        # Convert to tensor format for PyG
        doc_vocab = list(final_word_embeddings.keys())
        doc_word_id_map = {word: idx for idx, word in enumerate(doc_vocab)}

        word_embedding_tensor = torch.zeros(len(doc_vocab), next(iter(final_word_embeddings.values())).size(-1))
        for word, emb in final_word_embeddings.items():
            word_embedding_tensor[doc_word_id_map[word]] = emb

        # Normalize embeddings for stability
        word_embedding_tensor = F.normalize(word_embedding_tensor, p=2, dim=1)

        return word_embedding_tensor, doc_word_id_map, ordered_words

    def get_sentence_embeddings(self, doc_text, tokenizer, model, max_length=512, stride=256):
        """Computes sentence embeddings using BERT-based sentence splitting and a sliding window approach.

        Args:
            doc_text (str): Full document text.
            tokenizer (BertTokenizerFast): Tokenizer instance.
            model (BertModel): Pretrained BERT model.
            max_length (int): Maximum number of tokens per chunk.
            stride (int): Overlap between chunks.

        Returns:
            torch.Tensor: Sentence embeddings.
            List[str]: Extracted sentences.
        """
        # Use BERT tokenizer to split sentences
        encoded = tokenizer(doc_text, return_offsets_mapping=True, truncation=False)
        offsets = encoded['offset_mapping']
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        sentences = []
        current_sentence = []
        last_end = 0
        
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue  # Skip special tokens
            
            token_text = doc_text[start:end]
            current_sentence.append(token_text)
            
            # Use punctuation as a heuristic for sentence boundaries
            if token_text in {'.', '!', '?'}:
                sentence_text = " ".join(current_sentence).strip()
                sentences.append(sentence_text)
                current_sentence = []
            
            last_end = end
        
        if current_sentence:
            sentences.append(" ".join(current_sentence).strip())  # Append the last sentence
        
        sentence_embeddings = []
        
        # Compute embeddings for each sentence
        with torch.no_grad():
            for sentence in sentences:
                encoded = tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding=False,
                    truncation=False  # Manually handle chunking
                )
                
                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                
                chunk_embeddings = []
                for start in range(0, len(input_ids), stride):
                    end = min(start + max_length, len(input_ids))
                    
                    chunk_input_ids = input_ids[start:end].unsqueeze(0)
                    chunk_attention_mask = attention_mask[start:end].unsqueeze(0)
                    
                    # Run through BERT
                    outputs = model(
                        input_ids=chunk_input_ids.to(model.device),
                        attention_mask=chunk_attention_mask.to(model.device)
                    )
                    chunk_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
                    
                    chunk_embeddings.append(chunk_embedding)
                    
                    if end == len(input_ids):  # Stop if we've reached the full sentence
                        break
                
                # Aggregate chunks (Mean Pooling)
                sentence_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
                sentence_embeddings.append(sentence_embedding.squeeze(0))
        
        return torch.stack(sentence_embeddings), sentences


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
        
        word_embeddings, doc_word_id_map, doc_words = self.get_word_embeddings(doc_text, self.tokenizer, self.bert)

        # Store word embeddings
        data['word'].x = word_embeddings
        
        sentence_embeddings, sentences = self.get_sentence_embeddings(doc_text, self.tokenizer, self.bert)

        # Store sentence embeddings in PyTorch Geometric format
        data['sentence'].x = sentence_embeddings
        
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
            L = len(window)  # Window length
            for p in range(1, L):
                for q in range(0, p):
                    word_p, word_q = window[p], window[q]
                    word_p_id, word_q_id = doc_word_id_map[word_p], doc_word_id_map[word_q]

                    if word_p_id == word_q_id:  # Avoid self-loops
                        continue

                    # Compute positional weight
                    rho = q
                    pos_weight = (self.alpha * (rho / L)) + ((1 - self.alpha) * ((L - rho) / L))

                    # Store weight in a directional manner
                    edge_key = (word_q_id, word_p_id)  # Maintain directionality
                    if edge_key in word_pair_weights:
                        word_pair_weights[edge_key] += pos_weight  # Sum occurrences
                    else:
                        word_pair_weights[edge_key] = pos_weight

        # Convert word pair occurrences into final edge tensors
        word_word_edges, edge_weights = [], []

        for (word1_idx, word2_idx), weight in word_pair_weights.items():
            word_word_edges.append([word1_idx, word2_idx])
            edge_weights.append(weight)  # Use summed weights

        # Store edges in PyTorch Geometric HeteroData format
        if word_word_edges:  # Check if we have any edges
            data['word', 'co_occurs', 'word'].edge_index = torch.tensor(
                word_word_edges, dtype=torch.long
            ).t()
            data['word', 'co_occurs', 'word'].edge_attr = torch.tensor(
                edge_weights, dtype=torch.float
            )
        
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
            
            if sentence_edges:  # Check if we have any edges
                data['sentence', 'related_to', 'sentence'].edge_index = torch.tensor(
                    sentence_edges, dtype=torch.long
                ).t()
                data['sentence', 'related_to', 'sentence'].edge_attr = torch.tensor(
                    sentence_edge_weights, dtype=torch.float
                )
        
        return data

    def process_documents(self, json_files: List[Path]) -> List[HeteroData]:
        """Process multiple documents into graphs."""
        documents = self.load_documents(json_files)
        graphs = []
        
        for doc in tqdm(documents, desc="Building graphs"):
            graph = self.build_graph(doc)
            graphs.append(graph)
            
        return graphs 