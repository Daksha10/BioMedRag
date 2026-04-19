"""
DPR (Dense Passage Retrieval) Encoders
---------------------------------------
Facebook's DPR framework utilizes two distinct transformer models trained to work in pairs:
  - DPRQuestionEncoder : Designed to understand search queries (used by DPRRetriever).
  - DPRContextEncoder  : Designed to summarize long document passages (used by the indexing scripts).

Both models convert text into 768-dimensional float32 vectors. When these vectors 
are close together in space (measured via Cosine Similarity or Dot Product), 
it indicates that the question and document are semantically related.
"""

import torch  # Import PyTorch for deep learning operations
from transformers import (
    DPRQuestionEncoder,         # Transformer model for questions
    DPRQuestionEncoderTokenizer, # Tokenizer for splitting question text into IDs
    DPRContextEncoder,          # Transformer model for document passages
    DPRContextEncoderTokenizer,  # Tokenizer for splitting passage text into IDs
)
import numpy as np  # Import numpy for numeric vector manipulation


class DPRQueryEncoder:
    """
    Encoder for search queries. Used 'online' during the RAG retrieval phase.
    """

    def __init__(
        self,
        model_name: str = "facebook/dpr-question_encoder-single-nq-base",
        max_length: int = 512,  # Maximum tokens the model can process at once
    ):
        # Detect if a GPU (CUDA) is available for faster inference, else use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        # Load the pre-trained tokenizer from HuggingFace
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        # Load the pre-trained weights and move the model to the target device (GPU/CPU)
        self.model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        # Set the model to evaluation mode (disables dropout, etc.)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """Processes a single text string into a 768-d numpy vector."""
        # Convert raw text into numerical tensors that the model can understand
        inputs = self.tokenizer(
            text,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,     # Cut text if it exceeds max_length
            padding=True,        # Pad text if it is shorter than the max_length
            max_length=self.max_length,
        ).to(self.device)

        # Disable gradient calculations to save memory and improve speed
        with torch.no_grad():
            outputs = self.model(**inputs)

        # pooler_output extracts the [CLS] token representation, 
        # which is the optimized summary of the entire sentence for retrieval.
        vector = outputs.pooler_output[0].cpu().numpy().astype("float32")
        return vector


class DPRPassageEncoder:
    """
    Encoder for document passages. Used 'offline' to pre-calculate embeddings for the index.
    """

    def __init__(
        self,
        model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        max_length: int = 512,
    ):
        # Initialize device and max processing length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        # Load tokenizer and model for context/passage encoding
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRContextEncoder.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """Processes a single document passage into a 768-d numpy vector."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the pooled vector representation
        vector = outputs.pooler_output[0].cpu().numpy().astype("float32")
        return vector

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Processes multiple text passages simultaneously for high-throughput encoding.
        Returns a stacked 2D numpy array (num_docs, 768).
        """
        all_vectors = []
        # Process the input list in chunks of batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Tokenize the entire batch at once
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Move results to CPU and convert to numpy
            vectors = outputs.pooler_output.cpu().numpy().astype("float32")
            all_vectors.append(vectors)

        # Jointly stack all batches into one large 2D matrix
        return np.vstack(all_vectors)


if __name__ == "__main__":
    # Integration smoke test to verify model loading and vector shapes
    print("Testing DPRQueryEncoder...")
    qenc = DPRQueryEncoder()
    vec = qenc.encode("What are the structural proteins of a coronavirus?")
    print(f"  Query vector shape : {vec.shape}, dtype: {vec.dtype}")

    print("Testing DPRPassageEncoder...")
    penc = DPRPassageEncoder()
    bvecs = penc.encode_batch(
        [
            "Coronaviruses have four major structural proteins: spike (S), envelope (E), membrane (M), and nucleocapsid (N).",
            "The spike protein mediates viral entry into host cells.",
        ]
    )
    print(f"  Passage batch shape: {bvecs.shape}, dtype: {bvecs.dtype}")
    print("DPR encoders OK ✓")
