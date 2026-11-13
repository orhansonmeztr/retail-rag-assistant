# Deployment

Everything below uses the Azure CLI (`az login` first). Replace `<...>` placeholders.
Resource names must be globally unique where noted.

```bash
RG=rg-retail-rag
LOC=swedencentral
```

## 1. API on Azure Container Apps

The `Publish image` workflow pushes `ghcr.io/<owner>/retail-rag-assistant:<sha>` and `:latest`
on every push to `main`. Make the package public once (GitHub → Packages → package settings →
Change visibility), or pass `--registry-*` credentials below.

```bash
az containerapp env create -n retail-rag-env -g $RG -l $LOC

az containerapp create -n retail-rag-api -g $RG --environment retail-rag-env \
  --image ghcr.io/<owner>/retail-rag-assistant:latest \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 \
  --secrets aoai-key=<azure-openai-key> search-key=<search-key> api-keys=<comma-separated-client-keys> \
  --env-vars \
    AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/ \
    AZURE_OPENAI_API_KEY=secretref:aoai-key \
    AZURE_OPENAI_API_VERSION=2024-10-21 \
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini \
    AZURE_SEARCH_ENDPOINT=https://<search>.search.windows.net \
    AZURE_SEARCH_API_KEY=secretref:search-key \
    AZURE_SEARCH_INDEX=retail-knowledge-v1 \
    API_KEYS=secretref:api-keys

az containerapp show -n retail-rag-api -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

`--min-replicas 0` scales to zero when idle, so a demo costs almost nothing; the first
request after a pause takes a few seconds longer (cold start).

**Continuous deployment (optional).** Create a federated credential for GitHub Actions
(OIDC, no stored passwords), then set the repository secrets `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` and the variables `DEPLOY_CONTAINER_APP=true`,
`CONTAINER_APP_NAME=retail-rag-api`, `AZURE_RESOURCE_GROUP=rg-retail-rag`. Each push to
`main` then rolls the new image out.

## 2. Event-driven re-indexing with Azure Functions

```bash
ST=<uniquestorage>
APP=<unique-function-app>

az storage account create -n $ST -g $RG -l $LOC --sku Standard_LRS
az storage container create -n catalog --account-name $ST --auth-mode login
az functionapp create -g $RG -n $APP --storage-account $ST \
  --flexconsumption-location $LOC --runtime python --runtime-version 3.12

az functionapp config appsettings set -g $RG -n $APP --settings \
  CatalogStorage="$(az storage account show-connection-string -n $ST -g $RG -o tsv)" \
  AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/ \
  AZURE_OPENAI_API_KEY=<key> AZURE_OPENAI_API_VERSION=2024-10-21 \
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small \
  AZURE_SEARCH_ENDPOINT=https://<search>.search.windows.net \
  AZURE_SEARCH_API_KEY=<key> AZURE_SEARCH_INDEX=retail-knowledge-v1

cd functions && ./build.sh && func azure functionapp publish $APP --python
```

The trigger uses the Event Grid source, so subscribe the function to blob-created events:

```bash
KEY=$(az functionapp keys list -g $RG -n $APP --query "systemKeys.blobs_extension" -o tsv)
az eventgrid system-topic create -g $RG -n catalog-events -l $LOC \
  --topic-type Microsoft.Storage.StorageAccounts \
  --source $(az storage account show -n $ST -g $RG --query id -o tsv)
az eventgrid system-topic event-subscription create -g $RG --system-topic-name catalog-events \
  -n catalog-changed --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with /blobServices/default/containers/catalog/ \
  --endpoint "https://$APP.azurewebsites.net/runtime/webhooks/blobs?functionName=Host.Functions.catalog_changed&code=$KEY"
```

Test it: upload an edited catalog and watch the log line `catalog sync ...`.

```bash
az storage blob upload --account-name $ST -c catalog -f data/products.jsonl -n products.jsonl \
  --overwrite --auth-mode login
```

Locally, `retail-rag sync data/products.jsonl` runs the same incremental sync without
Functions.
