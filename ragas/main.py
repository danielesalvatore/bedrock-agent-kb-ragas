from langchain_community.document_loaders import DirectoryLoader
from langchain_aws import BedrockChat, BedrockEmbeddings

from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


# ------------------------- Generate a Synthetic Test Set

# Load sample dataset
loader = DirectoryLoader("../assets/kb-docs")
documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source']

# Configure Bedrock model
config = {
    "credentials_profile_name": "fao-sandbox",  # E.g "default"
    "region_name": "us-east-1",  # E.g. "us-east-1"
    "generator_llm": "amazon.titan-text-lite-v1",  # E.g "anthropic.claude-v2"
    "critic_llm": "anthropic.claude-3-haiku-20240307-v1:0",  # E.g "anthropic.claude-v2"
    "model_kwargs": {"temperature": 0},
}

generator_llm = BedrockChat(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model_id=config["generator_llm"],
    model_kwargs=config["model_kwargs"],
)

critic_llm = BedrockChat(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model_id=config["critic_llm"],
    model_kwargs=config["model_kwargs"],
)

# Initialize embeddings
bedrock_embeddings = BedrockEmbeddings(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
)

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, bedrock_embeddings)

# generate testset
testset = generator.generate_with_langchain_docs(
    documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

# testset.to_pandas().to_csv("ragas/testset.csv", index=False)

# ------------------------- Evaluating Using Your Test Set

# Define metrics
metrics = [
    faithfulness,
    context_recall,
    context_precision,
    harmfulness,
]

# Evaluate the model
result = evaluate(
    testset,
    metrics=metrics,
    llm=critic_llm,
    embeddings=bedrock_embeddings,
)

# Convert results to pandas DataFrame
df = result.to_pandas()
df.head()
