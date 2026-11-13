"""Wire concrete components together from Settings (composition root)."""

from __future__ import annotations

from .agent import RetailAssistant, load_prompt
from .clients import make_openai_client, search_credential
from .config import Settings
from .documents import load_corpus
from .embeddings import AzureOpenAIEmbedder, HashingEmbedder
from .retriever import Retriever
from .search_backend import AzureSearchBackend, InMemoryBackend


def build_retriever(settings: Settings, openai_client=None) -> Retriever:
    if settings.search_backend == "memory":
        embedder = (
            AzureOpenAIEmbedder(
                openai_client,
                settings.azure_openai_embedding_deployment,
                settings.embedding_dimensions,
            )
            if openai_client is not None
            else HashingEmbedder()
        )
        retriever = Retriever(embedder, InMemoryBackend(), settings.top_k)
        retriever.index(load_corpus(settings.data_dir))  # in-memory index is built at startup
        return retriever

    openai_client = openai_client or make_openai_client(settings)
    backend = AzureSearchBackend(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index,
        credential=search_credential(settings),
        dimensions=settings.embedding_dimensions,
        use_semantic_ranker=settings.use_semantic_ranker,
    )
    embedder = AzureOpenAIEmbedder(
        openai_client, settings.azure_openai_embedding_deployment, settings.embedding_dimensions
    )
    return Retriever(embedder, backend, settings.top_k)


def build_shield(settings: Settings):
    if not settings.content_safety_endpoint:
        return None
    from .security import PromptShield

    return PromptShield(settings.content_safety_endpoint, settings.content_safety_api_key or "")


def build_assistant(settings: Settings) -> RetailAssistant:
    client = make_openai_client(settings)
    system, version = load_prompt(settings.prompt_path)
    return RetailAssistant(
        chat_client=client,
        deployment=settings.azure_openai_chat_deployment,
        retriever=build_retriever(settings, client),
        system_prompt=system,
        prompt_version=version,
        max_steps=settings.max_agent_steps,
        top_k=settings.top_k,
        answer_guard=settings.answer_guard,
        shield=build_shield(settings),
    )
