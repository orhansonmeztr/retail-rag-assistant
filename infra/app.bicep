// Application layer on top of EXISTING Azure OpenAI and Azure AI Search resources:
// Azure AI Content Safety (Prompt Shields), Log Analytics + Application Insights, and
// optionally the API on Azure Container Apps. Content Safety and the monitoring pair are
// created by default; pass createContentSafety=false / createMonitoring=false with their
// names to reuse ones made in the portal (only one free F0 Content Safety per subscription).
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

param createContentSafety bool = true
param contentSafetyName string = '${prefix}-safety'
@allowed(['F0', 'S0'])
@description('F0 is free but limited to one Content Safety resource per subscription')
param contentSafetySku string = 'F0'

param createMonitoring bool = true
param logAnalyticsName string = '${prefix}-logs'
param appInsightsName string = '${prefix}-appi'

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

resource newContentSafety 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (createContentSafety) {
  name: contentSafetyName
  location: location
  kind: 'ContentSafety'
  sku: { name: contentSafetySku }
  properties: {
    customSubDomainName: contentSafetyName
    publicNetworkAccess: 'Enabled'
  }
}

resource oldContentSafety 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = if (!createContentSafety) {
  name: contentSafetyName
}

resource newLogs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = if (createMonitoring) {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource oldLogs 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = if (!createMonitoring) {
  name: logAnalyticsName
}

resource newAppInsights 'Microsoft.Insights/components@2020-02-02' = if (createMonitoring) {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: newLogs.id
  }
}

resource oldAppInsights 'Microsoft.Insights/components@2020-02-02' existing = if (!createMonitoring) {
  name: appInsightsName
}

// the same values whether the resource was created here or reused
var safetyEndpoint = createContentSafety
  ? newContentSafety!.properties.endpoint
  : oldContentSafety!.properties.endpoint
var safetyKey = createContentSafety ? newContentSafety!.listKeys().key1 : oldContentSafety!.listKeys().key1
var logsCustomerId = createMonitoring ? newLogs!.properties.customerId : oldLogs!.properties.customerId
var logsKey = createMonitoring
  ? newLogs!.listKeys().primarySharedKey
  : oldLogs!.listKeys().primarySharedKey
var appiConnection = createMonitoring
  ? newAppInsights!.properties.ConnectionString
  : oldAppInsights!.properties.ConnectionString

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = if (deployApi) {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logsCustomerId
        sharedKey: logsKey
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
        { name: 'safety-key', value: safetyKey }
        { name: 'api-keys', value: apiKeys }
        { name: 'appi-conn', value: appiConnection }
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
            { name: 'CONTENT_SAFETY_ENDPOINT', value: safetyEndpoint }
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

output CONTENT_SAFETY_ENDPOINT string = safetyEndpoint
output CONTENT_SAFETY_NAME string = contentSafetyName
output APPLICATIONINSIGHTS_NAME string = appInsightsName
output API_URL string = deployApi ? 'https://${api!.properties.configuration.ingress.fqdn}' : ''
