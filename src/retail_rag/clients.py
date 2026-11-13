"""Azure client factories. API key if provided, otherwise Entra ID (keyless) auth."""

from __future__ import annotations

from urllib.parse import urlsplit

from openai import AzureOpenAI

from .config import Settings


def base_endpoint(url: str) -> str:
    """Foundry shows URLs like https://x.services.ai.azure.com/openai/v1/responses;
    the AzureOpenAI client needs only scheme + host."""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"invalid Azure OpenAI endpoint: {url!r}")
    return f"{parts.scheme}://{parts.netloc}/"


def make_openai_client(settings: Settings) -> AzureOpenAI:
    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    endpoint = base_endpoint(settings.azure_openai_endpoint)
    if settings.azure_openai_api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
    )


def search_credential(settings: Settings):
    if settings.azure_search_api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(settings.azure_search_api_key)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()
