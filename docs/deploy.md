# Deployment

Everything below uses the Azure CLI (`az login` first). Region, resource group and the
name prefix come from the *Deployment* section of `.env` (see `.env.example`), so they are
written down once; the Azure OpenAI and Search settings come from the same file.

```bash
set -a; source .env; set +a          # export every .env value into this shell
RG=$AZURE_RESOURCE_GROUP
LOC=$AZURE_LOCATION                  # region for NEW resources; existing ones keep theirs
P=$AZURE_PREFIX                      # names: $P-api, $P-env, ... (keep it short and unique)
```

The simplest route is the Bicep template, which creates the same resources as the commands
below and wires their keys together:

```bash
az deployment group create -g $RG -f infra/app.bicep -p prefix=$P location=$LOC \
  openAiName=<openai-or-foundry-resource> openAiEndpoint=$AZURE_OPENAI_ENDPOINT \
  searchName=<search-service> searchIndex=$AZURE_SEARCH_INDEX
```

Add `deployApi=true apiImage=ghcr.io/<owner>/retail-rag-assistant:latest apiKeys="$API_KEYS"`
to also run the API. If Content Safety, Log Analytics or Application Insights were already
created in the portal, reuse them instead of creating duplicates (a subscription gets only
one free F0 Content Safety resource):

```bash
  createContentSafety=false contentSafetyName=<content-safety-resource> \
  createMonitoring=false logAnalyticsName=<workspace> appInsightsName=<app-insights>
```

Run `az deployment group what-if` with the same parameters first to see exactly what will be
created or changed.

The step-by-step CLI commands are kept for when you want to create pieces individually.

## 1. API on Azure Container Apps

The `Publish image` workflow pushes `ghcr.io/<owner>/retail-rag-assistant:<sha>` and `:latest`
on every push to `main`. Make the package public once (GitHub → Packages → package settings →
Change visibility), or pass `--registry-*` credentials below.

```bash
az containerapp env create -n $P-env -g $RG -l $LOC

az containerapp create -n $P-api -g $RG --environment $P-env \
  --image ghcr.io/<owner>/retail-rag-assistant:latest \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 \
  --secrets aoai-key="$AZURE_OPENAI_API_KEY" search-key="$AZURE_SEARCH_API_KEY" api-keys="$API_KEYS" \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
    AZURE_OPENAI_API_KEY=secretref:aoai-key \
    AZURE_OPENAI_API_VERSION="$AZURE_OPENAI_API_VERSION" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="$AZURE_OPENAI_CHAT_DEPLOYMENT" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="$AZURE_OPENAI_EMBEDDING_DEPLOYMENT" \
    AZURE_SEARCH_ENDPOINT="$AZURE_SEARCH_ENDPOINT" \
    AZURE_SEARCH_API_KEY=secretref:search-key \
    AZURE_SEARCH_INDEX="$AZURE_SEARCH_INDEX" \
    API_KEYS=secretref:api-keys

az containerapp show -n $P-api -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

`--min-replicas 0` scales to zero when idle, so a demo costs almost nothing; the first
request after a pause takes a few seconds longer (cold start).

**Continuous deployment (optional).** Create a federated credential for GitHub Actions
(OIDC, no stored passwords), then set the repository secrets `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` and the variables `DEPLOY_CONTAINER_APP=true`,
`CONTAINER_APP_NAME=<prefix>-api` and `AZURE_RESOURCE_GROUP` (same values as in `.env`). Each push to
`main` then rolls the new image out.

## 2. Event-driven re-indexing with Azure Functions

```bash
ST=${P//-/}catalog                    # storage names: lowercase letters and digits only
APP=$P-sync

az storage account create -n $ST -g $RG -l $LOC --sku Standard_LRS
az storage container create -n catalog --account-name $ST --auth-mode login
az functionapp create -g $RG -n $APP --storage-account $ST \
  --flexconsumption-location $LOC --runtime python --runtime-version 3.12

az functionapp config appsettings set -g $RG -n $APP --settings \
  CatalogStorage="$(az storage account show-connection-string -n $ST -g $RG -o tsv)" \
  AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY" \
  AZURE_OPENAI_API_VERSION="$AZURE_OPENAI_API_VERSION" \
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT="$AZURE_OPENAI_EMBEDDING_DEPLOYMENT" \
  AZURE_SEARCH_ENDPOINT="$AZURE_SEARCH_ENDPOINT" AZURE_SEARCH_API_KEY="$AZURE_SEARCH_API_KEY" \
  AZURE_SEARCH_INDEX="$AZURE_SEARCH_INDEX"

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
