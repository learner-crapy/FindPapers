import argparse
import os
import numpy as np
import pandas as pd
import torch
import json
from typing import List, Union, Dict, Any, Optional
from sklearn.preprocessing import normalize
from transformers import BertModel, BertTokenizer
from pymilvus import model
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextEmbedder:
    """
    A unified text embedding system supporting multiple embedding models.
    Takes text input and outputs embeddings.
    """

    def __init__(self,
                 model_type: str = 'sentence-transformer',
                 model_name: str = 'all-distilroberta-v1',
                 embedding_dim: int = 512,
                 device: str = 'cuda',
                 normalize_embeddings: bool = True,
                 api_key: Optional[str] = None):
        """
        Initialize the text embedder with specified model configuration.

        Args:
            model_type: Type of embedding model ('bert', 'sentence-transformer', 'splade', 'bge-m3', 'mgte', 'mistral')
            model_name: Specific model name/identifier
            embedding_dim: Target embedding dimension
            device: Device to run the model on ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings
            api_key: API key for cloud-based models (Mistral, OpenAI)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.api_key = api_key

        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.embedding_function = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the specified embedding model."""
        if self.model_type == 'bert':
            self._initialize_bert()
        elif self.model_type == 'sentence-transformer':
            self._initialize_sentence_transformer()
        elif self.model_type == 'splade':
            self._initialize_splade()
        elif self.model_type == 'bge-m3':
            self._initialize_bge_m3()
        elif self.model_type == 'mgte':
            self._initialize_mgte()
        elif self.model_type == 'mistral':
            self._initialize_mistral()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_bert(self):
        """Initialize BERT model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

    def _initialize_sentence_transformer(self):
        """Initialize Sentence Transformer embedding function."""
        self.embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings
        )
        print(f"device {self.device}")

    def _initialize_splade(self):
        """Initialize SPLADE embedding function."""
        self.embedding_function = model.sparse.SpladeEmbeddingFunction(
            model_name=self.model_name,
            device=self.device
        )

    def _initialize_bge_m3(self):
        """Initialize BGE-M3 embedding function."""
        self.embedding_function = model.hybrid.BGEM3EmbeddingFunction(
            model_name=self.model_name,
            device=self.device,
            use_fp16=False,
            normalize_embeddings=self.normalize_embeddings
        )

    def _initialize_mgte(self):
        """Initialize MGTE embedding function."""
        self.embedding_function = model.hybrid.MGTEEmbeddingFunction(
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings
        )

    def _initialize_mistral(self):
        """Initialize Mistral AI embedding function."""
        if not self.api_key:
            raise ValueError("API key is required for Mistral AI models")
        self.embedding_function = model.dense.MistralAIEmbeddingFunction(
            model_name=self.model_name,
            api_key=self.api_key
        )

    def _pad_or_truncate_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Pad or truncate vector to target embedding dimension.

        Args:
            vector: Input vector

        Returns:
            Vector with target dimension
        """
        vector = np.array(vector)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] < self.embedding_dim:
            padding = np.zeros((vector.shape[0], self.embedding_dim - vector.shape[1]))
            vector = np.concatenate([vector, padding], axis=1)
        elif vector.shape[1] > self.embedding_dim:
            vector = vector[:, :self.embedding_dim]

        if self.normalize_embeddings:
            vector = normalize(vector)

        return vector.squeeze()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            Embedding vector as numpy array
        """
        if self.model_type == 'bert':
            return self._embed_bert(text)
        else:
            return self._embed_with_function([text])

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input text strings

        Returns:
            Embedding vectors as numpy array
        """
        if self.model_type == 'bert':
            embeddings = []
            for text in texts:
                embeddings.append(self._embed_bert(text))
            return np.array(embeddings)
        else:
            return self._embed_with_function(texts)

    def _embed_bert(self, text: str) -> np.ndarray:
        """Generate BERT embedding for text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return self._pad_or_truncate_vector(embedding)

    def _embed_with_function(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Milvus embedding functions."""
        if self.model_type in ['bge-m3', 'mgte']:
            vectors = self.embedding_function.encode_documents(texts)
            dense_vectors = vectors['dense']
        else:
            dense_vectors = self.embedding_function.encode_documents(texts)

        # Handle single text case
        if len(texts) == 1:
            return self._pad_or_truncate_vector(dense_vectors)

        # Handle multiple texts
        processed_vectors = []
        for vector in dense_vectors:
            processed_vectors.append(self._pad_or_truncate_vector(vector))

        return np.array(processed_vectors)

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the current embedding configuration."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Text Embedding System with Multiple Model Options')

    parser.add_argument('--model_type', type=str, default='sentence-transformer',
                        choices=['bert', 'sentence-transformer', 'splade', 'bge-m3', 'mgte', 'mistral'],
                        help='Type of embedding model to use')

    parser.add_argument('--model_name', type=str, default='all-distilroberta-v1',
                        help='Specific model name/identifier')

    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Target embedding dimension')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to run the model on')

    parser.add_argument('--input_text', type=str, default=None,
                        help='Single text to embed')

    parser.add_argument('--input_file', type=str, default=None,
                        help='File containing texts to embed (one per line)')

    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save embeddings (JSON format)')

    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for cloud-based models')

    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize embeddings')

    return parser.parse_args()


def main():
    """Main function to demonstrate text embedding usage."""
    args = parse_args()

    # Initialize embedder
    embedder = TextEmbedder(
        model_type=args.model_type,
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        device=args.device,
        normalize_embeddings=args.normalize,
        api_key=args.api_key
    )

    # Get input texts
    texts = []
    if args.input_text:
        texts = [args.input_text]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default example texts
        texts = [
            "This is a sample text for embedding.",
            "Another example text to demonstrate the system.",
            "The quick brown fox jumps over the lazy dog."
        ]

    print(f"Using model: {args.model_type} - {args.model_name}")
    print(f"Embedding {len(texts)} texts...")

    # Generate embeddings
    if len(texts) == 1:
        embedding = embedder.embed_text(texts[0])
        print(f"Single text embedding shape: {embedding.shape}")
        print(f"Embedding: {embedding[:10]}...")  # Show first 10 values
    else:
        embeddings = embedder.embed_texts(texts)
        print(f"Multiple texts embedding shape: {embeddings.shape}")
        print(f"First embedding: {embeddings[0][:10]}...")  # Show first 10 values of first embedding

    # Save embeddings if output file specified
    if args.output_file:
        result = {
            'model_info': embedder.get_embedding_info(),
            'texts': texts,
            'embeddings': embeddings.tolist() if len(texts) > 1 else embedding.tolist()
        }

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Embeddings saved to: {args.output_file}")


if __name__ == "__main__":
    main()

