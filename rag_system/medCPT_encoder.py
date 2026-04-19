from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification  # Import HuggingFace models
import torch  # Import PyTorch for tensor operations


class MedCPTQueryEncoder:
    """
    Encoder for medical search queries based on the MedCPT framework.
    This model is specifically pre-trained on medical query-document pairs.
    """
    def __init__(self, model_name='ncbi/MedCPT-Query-Encoder', max_length=512):
        # Choose GPU (cuda) if available for performance, otherwise fallback to CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Define the maximum number of tokens allowed per query
        self.max_length = max_length

        # Load the pre-trained MedCPT Query model from the NCBI repository
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # Load the corresponding tokenizer to process text into IDs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        """Processes a query string into a vector representation."""
        # Disable gradient tracking as this is an inference-only operation
        with torch.no_grad():
            # Convert text to tensor and move to device (CPU/GPU)
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(self.device)
            # Forward pass through the transformer
            outputs = self.model(**inputs)
            # The [CLS] token (first index) typically represents the summary vector
            return outputs.last_hidden_state[:, 0, :]


class MedCPTCrossEncoder:
    """
    A Cross-Encoder model that evaluates pairs of (Query, Document) simultaneously.
    Unlike standard retrievers, this provides a highly accurate semantic relevance score.
    Used in the HybridRetriever's 'Reranking' phase.
    """
    def __init__(self, model_name='ncbi/MedCPT-Cross-Encoder'):
        # Detect the best available computing device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load a model specifically architecture for Sequence Classification (cross-attention scoring)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # Load the tokenizer for processing paired text
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def score(self, articles, query):
        """
        Computes a relevance score for multiple articles against a single query.
        Returns a tensor of scores where higher values indicate stronger relevance.
        """
        # Create pairs of [Query, Document1], [Query, Document2], etc.
        pairs = [[query, article] for article in articles]

        with torch.no_grad():
            # Tokenize all pairs simultaneously in a single batch
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",  # Use PyTorch tensors
                max_length=512,
            ).to(self.device)

            # Extract the classification logits (raw output scores) from the model
            logits = self.model(**encoded).logits.squeeze(dim=1)
        
        # Return the resulting relevance scores
        return logits


if __name__ == "__main__":
    # Integration smoke test for the CrossEncoder functionality
    cross_encoder = MedCPTCrossEncoder()

    # Define a sample medical query
    query = "What is the treatment for diabetes?"

    # Define sample medical documents
    articles = [
        "Diabetes is a chronic disease that occurs when the body is unable to produce enough insulin or use it effectively. Treatment for diabetes includes lifestyle changes, such as diet and exercise, as well as medications like insulin and oral hypoglycemic drugs.",
        "The treatment for diabetes involves managing blood sugar levels through diet, exercise, and medication. Insulin therapy, oral medications, and lifestyle changes are common approaches to managing diabetes.",
        "Diabetes treatment typically involves a combination of diet, exercise, and medication. Insulin therapy, oral medications, and lifestyle changes are key components of managing diabetes.",
    ]

    # Calculate and display scores
    scores = cross_encoder.score(articles, query)

    # Print results to console
    for i, (article, score) in enumerate(zip(articles, scores)):
        print(f"Article {i+1}: {article}")
        print(f"Score: {score:.4f}\n")
