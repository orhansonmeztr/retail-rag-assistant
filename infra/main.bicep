// Full environment from scratch: Azure AI Search and an Azure AI Foundry (AIServices)
// resource with chat + embedding deployments and an optional eval judge, plus the application layer in app.bicep (Content Safety, Application
// Insights, optional Container Apps). If Search / OpenAI already exist, deploy app.bicep
// alone instead (only one free-tier Search service is allowed per subscription).
// Deploy:  az deployment group create -g <rg> -f infra/main.bicep -p prefix=<unique>
//          add  -p deployApi=true apiImage=ghcr.io/<owner>/retail-rag-assistant:latest apiKeys=<k>
//          to also run the API.
// The AIServices resource can then be connected to an Azure AI Foundry project
// (Foundry portal > Management center > Connected resources) for playground, tracing and evals.

@description('Short unique prefix for resource names')
param prefix string
@description('Region for the AIServices resource and all application resources')
param location string = resourceGroup().location
@description('Region for AI Search; can differ when the free tier or capacity is not offered in location')
param searchLocation string = location

@allowed(['free', 'basic'])
param searchSku string = 'free'

param chatModel string = 'gpt-4.1-mini'
param chatModelVersion string = '2025-04-14'
param embeddingModel string = 'text-embedding-3-small'

@description('Optional eval judge from another model family (avoids self-preference bias); empty = none. Used only by `retail-rag eval`, never by the API.')
param judgeModel string = ''
@description('Model publisher format of the judge, e.g. DeepSeek, OpenAI, Microsoft')
param judgeModelFormat string = 'DeepSeek'
param judgeModelVersion string = '1'

@allowed(['F0', 'S0'])
@description('F0 is free but limited to one Content Safety resource per subscription')
param contentSafetySku string = 'F0'

param deployApi bool = false
param apiImage string = ''
@secure()
@description('Comma-separated API keys clients must send in X-API-Key')
param apiKeys string = ''

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${prefix}-search'
  location: searchLocation
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: searchSku == 'free' ? 'disabled' : 'free'
    authOptions: { aadOrApiKey: { aadAuthFailureMode: 'http401WithBearerChallenge' } }
  }
}

// AIServices (not kind OpenAI) so non-OpenAI models such as the judge can be deployed too;
// it still serves the Azure OpenAI API the app uses.
resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-ai'
  location: location
  kind: 'AIServices'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: '${prefix}-ai'
    publicNetworkAccess: 'Enabled'
  }
}

resource chat 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: chatModel, version: chatModelVersion }
  }
}

resource embedding 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: embeddingModel
  dependsOn: [chat] // deployments on one account must be created sequentially
  sku: { name: 'Standard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModel, version: '1' }
  }
}

resource judge 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (!empty(judgeModel)) {
  parent: openai
  name: empty(judgeModel) ? 'none' : judgeModel
  dependsOn: [embedding]
  sku: { name: 'GlobalStandard', capacity: 10 }
  properties: {
    model: { format: judgeModelFormat, name: judgeModel, version: judgeModelVersion }
  }
}

module app 'app.bicep' = {
  name: 'app'
  params: {
    prefix: prefix
    location: location
    openAiName: openai.name
    openAiEndpoint: openai.properties.endpoint
    chatDeployment: chat.name
    embeddingDeployment: embedding.name
    searchName: search.name
    contentSafetySku: contentSafetySku
    deployApi: deployApi
    apiImage: apiImage
    apiKeys: apiKeys
  }
}

output AZURE_OPENAI_ENDPOINT string = openai.properties.endpoint
output AZURE_SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output AZURE_OPENAI_CHAT_DEPLOYMENT string = chat.name
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT string = embedding.name
output JUDGE_DEPLOYMENT string = empty(judgeModel) ? '' : judge.name
output CONTENT_SAFETY_ENDPOINT string = app.outputs.CONTENT_SAFETY_ENDPOINT
output API_URL string = app.outputs.API_URL
