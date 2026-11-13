// Application layer on top of EXISTING Azure OpenAI and Azure AI Search resources:
// Azure AI Content Safety (Prompt Shields), Log Analytics + Application Insights, and
// optionally the API on Azure Container Apps.
//
//   az deployment group create -g <rg> -f infra/app.bicep \
//     -p prefix=<unique> openAiName=<aoai-or-foundry-resource> searchName=<search-service> \
//        openAiEndpoint=https://<name>.services.ai.azure.com/
//   add  -p deployApi=true apiImage=ghcr.io/<owner>/retail-rag-assistant:latest apiKeys=<k>

@description('Short unique prefix for new resource names')
param prefix string
param location string = resourceGroup().location

@description('Existing Azure OpenAI / Azure AI Foundry (AIServices) resource')
param openAiName string
param openAiResourceGroup string = resourceGroup().name
@description('Endpoint the API should call, e.g. https://<name>.services.ai.azure.com/')
param openAiEndpoint string
param chatDeployment string = 'gpt-4.1-mini'
param embeddingDeployment string = 'text-embedding-3-small'

@description('Existing Azure AI Search service')
param searchName string
param searchResourceGroup string = resourceGroup().name
param searchIndex string = 'retail-knowledge-v1'

@allowed(['F0', 'S0'])
@description('F0 is free but limited to one Content Safety resource per subscription')
param contentSafetySku string = 'F0'

param deployApi bool = false
param apiImage string = ''
@secure()
@description('Comma-separated API keys clients must send in X-API-Key')
param apiKeys string = ''

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: openAiName
  scope: resourceGroup(openAiResourceGroup)
}

resource search 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: searchName
  scope: resourceGroup(searchResourceGroup)
}

resource contentSafety 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-safety'
  location: location
  kind: 'ContentSafety'
  sku: { name: contentSafetySku }
  properties: {
    customSubDomainName: '${prefix}-safety'
    publicNetworkAccess: 'Enabled'
  }
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-appi'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = if (deployApi) {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = if (deployApi) {
  name: '${prefix}-api'
  location: location
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: { external: true, targetPort: 8000 }
      secrets: [
        { name: 'aoai-key', value: openai.listKeys().key1 }
        { name: 'search-key', value: search.listAdminKeys().primaryKey }
        { name: 'safety-key', value: contentSafety.listKeys().key1 }
        { name: 'api-keys', value: apiKeys }
        { name: 'appi-conn', value: appInsights.properties.ConnectionString }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAiEndpoint }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'aoai-key' }
            { name: 'AZURE_OPENAI_API_VERSION', value: '2024-10-21' }
            { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatDeployment }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: embeddingDeployment }
            { name: 'AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
            { name: 'AZURE_SEARCH_API_KEY', secretRef: 'search-key' }
            { name: 'AZURE_SEARCH_INDEX', value: searchIndex }
            { name: 'CONTENT_SAFETY_ENDPOINT', value: contentSafety.properties.endpoint }
            { name: 'CONTENT_SAFETY_API_KEY', secretRef: 'safety-key' }
            { name: 'API_KEYS', secretRef: 'api-keys' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appi-conn' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/health', port: 8000 } }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}

output CONTENT_SAFETY_ENDPOINT string = contentSafety.properties.endpoint
output CONTENT_SAFETY_NAME string = contentSafety.name
output APPLICATIONINSIGHTS_NAME string = appInsights.name
output API_URL string = deployApi ? 'https://${api!.properties.configuration.ingress.fqdn}' : ''
