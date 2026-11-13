"""Azure client factories. API key if provided, otherwise Entra ID (keyless) auth."""

from __future__ import annotations

from openai import AzureOpenAI

from .config import Settings


def make_openai_client(settings: Settings) -> AzureOpenAI:
    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    if settings.azure_openai_api_key:
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
    )


def search_credential(settings: Settings):
    if settings.azure_search_api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(settings.azure_search_api_key)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()
