# ragtree/vendor/byokg/bedrock_llms.py
from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class BedrockLLM(LLM):
    """
    Placeholder adapter for AWS Bedrock usage.
    In RAGTree you will use your existing LLM client abstraction instead.
    """
    def __init__(self, client=None, model_id=None, temperature=0.0):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        # Placeholder: integrate with boto3 bedrock runtime in a real deployment.
        raise NotImplementedError("BedrockLLM is a placeholder in this vendored file.")
