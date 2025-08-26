# Kubernetes MCP Server Deployment (Unauthenticated)

This directory contains the deployment manifests for the [Kubernetes MCP Server](https://github.com/containers/kubernetes-mcp-server) that enables your workshop chatbot to interact with Kubernetes/OpenShift clusters.

**Note: This deployment is configured for unauthenticated access for development/workshop purposes.**

## 🚀 Quick Deployment

### Prerequisites

- OpenShift CLI (`oc`) installed and configured
- Logged into your OpenShift cluster
- Cluster admin permissions (for RBAC setup)

### Deploy

```bash
# Make sure you're in the k8s-mcp-server directory
cd k8s-mcp-server

# Run the deployment script
./deploy.sh
```

The script will:
1. ✅ Check your OpenShift login status
2. 🚀 Deploy all components to the `workshop-mcp` namespace
3. 📊 Wait for the deployment to be ready
4. 🌐 Provide you with the MCP server URL (no authentication required)

### Manual Deployment

If you prefer to deploy manually:

```bash
# Apply all manifests (no secret needed for unauthenticated setup)
oc apply -f namespace.yaml
oc apply -f rbac.yaml
oc apply -f configmap.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml

# Check deployment status
oc get pods -n workshop-mcp
```

## 🔧 Configuration

### Unauthenticated Access

The MCP server is configured for unauthenticated access, making it simple to use for development and workshop scenarios. No API keys or authentication headers are required.

### RBAC Permissions

The MCP server runs with a service account that has cluster-wide read permissions and limited write permissions for workshop demonstrations. The RBAC configuration includes:

- **Read access**: All core resources (pods, services, deployments, etc.)
- **Write access**: Deployments, pods (for demo purposes)
- **Special access**: Pod exec and logs for troubleshooting
- **OpenShift**: Routes and projects

## 🧪 Testing

### Health Check

```bash
# Get the route URL
ROUTE_URL=$(oc get route mcp-server -n workshop-mcp -o jsonpath='{.spec.host}')

# Test health endpoint (no authentication required)
curl "https://${ROUTE_URL}/health"
```

### MCP Protocol Test

```bash
# Test MCP server capabilities (no authentication required)
curl -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' \
     "https://${ROUTE_URL}/mcp"
```

## 📋 Available MCP Tools

The server provides these tools for your chatbot:

| Tool | Description |
|------|-------------|
| `pods_list` | List all pods in the cluster |
| `pods_get` | Get details of a specific pod |
| `pods_log` | Get pod logs |
| `pods_exec` | Execute commands in pods |
| `deployments_list` | List deployments |
| `services_list` | List services |
| `namespaces_list` | List namespaces |
| `events_list` | List cluster events |
| `resources_create_or_update` | Create/update resources |
| `resources_delete` | Delete resources |
| `helm_install` | Install Helm charts |
| `helm_list` | List Helm releases |

## 🔍 Monitoring

### View Logs

```bash
# Follow deployment logs
oc logs -f deployment/mcp-server -n workshop-mcp

# Get pod status
oc get pods -n workshop-mcp -w
```

### Scaling

```bash
# Scale the deployment
oc scale deployment mcp-server --replicas=2 -n workshop-mcp
```

## 🗑️ Cleanup

```bash
# Delete the entire namespace and all resources
oc delete namespace workshop-mcp

# Or delete individual components
oc delete -f .
```

## 🔐 Security Considerations

**⚠️ Important: This deployment is configured for unauthenticated access**

1. **Network**: The route is exposed with TLS termination
2. **RBAC**: Minimal required permissions are granted to the service account
3. **CORS**: Configured to allow browser-based chatbot access
4. **Scope**: Suitable for development, workshops, and trusted environments
5. **Production**: For production use, consider adding authentication layers

## 🐛 Troubleshooting

### Common Issues

**Pod not starting?**
```bash
oc describe pod -l app.kubernetes.io/name=workshop-mcp -n workshop-mcp
```

**RBAC issues?**
```bash
oc auth can-i --list --as=system:serviceaccount:workshop-mcp:mcp-server
```

**Route not accessible?**
```bash
oc get route mcp-server -n workshop-mcp -o yaml
```

### Debug Mode

Enable debug logging by updating the ConfigMap:

```bash
oc patch configmap mcp-server-config -n workshop-mcp --type='merge' -p='{"data":{"server-config.json":"{\"logging\":{\"level\":\"debug\"}}"}}'
oc rollout restart deployment/mcp-server -n workshop-mcp
```
