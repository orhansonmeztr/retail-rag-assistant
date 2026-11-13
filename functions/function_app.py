"""Azure Functions app: re-index the knowledge base when a catalog file changes.

Upload `products.jsonl` or `faq.md` to the `catalog` blob container. An Event Grid based
blob trigger fires, and only documents whose content changed are re-embedded and upserted;
documents that disappeared from the file are deleted from the index.
"""

import logging

import azure.functions as func

from retail_rag.config import get_settings
from retail_rag.sync import build_sync_target, handle_catalog_blob

app = func.FunctionApp()


@app.blob_trigger(
    arg_name="blob",
    path="catalog/{name}",
    connection="CatalogStorage",
    source=func.BlobSource.EVENT_GRID,
)
def catalog_changed(blob: func.InputStream) -> None:
    name = (blob.name or "").rsplit("/", 1)[-1]
    retriever = build_sync_target(get_settings())
    report = handle_catalog_blob(name, blob.read().decode("utf-8"), retriever)
    logging.info("catalog sync %s: %s", name, report)
