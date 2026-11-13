// Minimal infrastructure: Azure AI Search + Azure OpenAI with chat and embedding deployments.
// Deploy:  az deployment group create -g <rg> -f infra/main.bicep -p prefix=<unique>
// The Azure OpenAI resource can then be connected to an Azure AI Foundry project
// (Foundry portal > Management center > Connected resources) for playground, tracing and evals.

@description('Short unique prefix for resource names')
param prefix string
param location string = resourceGroup().location

@allowed(['free', 'basic'])
param searchSku string = 'free'

param chatModel string = 'gpt-4.1-mini'
param chatModelVersion string = '2025-04-14'
param embeddingModel string = 'text-embedding-3-small'

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${prefix}-search'
  location: location
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: searchSku == 'free' ? 'disabled' : 'free'
    authOptions: { aadOrApiKey: { aadAuthFailureMode: 'http401WithBearerChallenge' } }
  }
}

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-aoai'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: '${prefix}-aoai'
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

output AZURE_OPENAI_ENDPOINT string = openai.properties.endpoint
output AZURE_SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output AZURE_OPENAI_CHAT_DEPLOYMENT string = chat.name
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT string = embedding.name
