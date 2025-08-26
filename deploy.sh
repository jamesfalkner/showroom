#!/bin/bash

# Workshop Deployment Script
# Deploys both the backend service and MCP server to OpenShift

set -e

# Configuration
NAMESPACE="showroom-mcp"
BACKEND_IMAGE="workshop-backend"
MCP_IMAGE="mcp-server"
GITLAB_MCP_IMAGE="gitlab-mcp-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if user is logged into OpenShift
check_oc_login() {
    log_info "Checking OpenShift login status..."

    if ! command -v oc &> /dev/null; then
        log_error "OpenShift CLI (oc) is not installed or not in PATH"
        exit 1
    fi

    if ! oc whoami &> /dev/null; then
        log_error "You are not logged into OpenShift. Please run 'oc login' first."
        exit 1
    fi

    local current_user=$(oc whoami)
    local current_server=$(oc whoami --show-server)
    log_success "Logged in as: $current_user"
    log_info "Connected to: $current_server"
}

# Check if namespace exists, create if it doesn't
setup_namespace() {
    log_info "Setting up namespace: $NAMESPACE"

    if oc get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        log_info "Creating namespace: $NAMESPACE"
        cat <<EOF | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
  labels:
    name: $NAMESPACE
    app.kubernetes.io/name: showroom-workshop
    app.kubernetes.io/component: namespace
EOF
        log_success "Created namespace: $NAMESPACE"
    fi

    # Switch to the namespace
    oc project "$NAMESPACE"
    log_success "Switched to namespace: $NAMESPACE"
}

# Create BuildConfig for backend
create_backend_buildconfig() {
    log_info "Creating BuildConfig for backend service..."

    cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: $BACKEND_IMAGE
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: $BACKEND_IMAGE
    app.kubernetes.io/component: backend
spec:
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Dockerfile.showroom
  output:
    to:
      kind: ImageStreamTag
      name: $BACKEND_IMAGE:latest
EOF

    log_success "Created BuildConfig for backend"
}

# Create BuildConfig for MCP server
create_mcp_buildconfig() {
    log_info "Creating BuildConfig for MCP server..."

    # Read Dockerfile.mcp contents and indent for YAML
    local dockerfile_content
    dockerfile_content=$(sed 's/^/      /' "$SCRIPT_DIR/Dockerfile.mcp")

    cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: $MCP_IMAGE
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: $MCP_IMAGE
    app.kubernetes.io/component: mcp-server
spec:
  source:
    type: Dockerfile
    dockerfile: |
$dockerfile_content
  strategy:
    type: Docker
  output:
    to:
      kind: ImageStreamTag
      name: $MCP_IMAGE:latest
EOF

    log_success "Created BuildConfig for MCP server"
}

# Create BuildConfig for GitLab MCP server
create_gitlab_mcp_buildconfig() {
    log_info "Creating BuildConfig for GitLab MCP server..."

    # Read Dockerfile.gitlab-mcp contents and indent for YAML
    local dockerfile_content
    dockerfile_content=$(sed 's/^/      /' "$SCRIPT_DIR/Dockerfile.gitlab-mcp")

    cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: $GITLAB_MCP_IMAGE
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: $GITLAB_MCP_IMAGE
    app.kubernetes.io/component: gitlab-mcp-server
spec:
  source:
    type: Dockerfile
    dockerfile: |
$dockerfile_content
  strategy:
    type: Docker
  output:
    to:
      kind: ImageStreamTag
      name: $GITLAB_MCP_IMAGE:latest
EOF

    log_success "Created BuildConfig for GitLab MCP server"
}



# Create ImageStreams
create_imagestreams() {
    log_info "Creating ImageStreams..."

    for image in $BACKEND_IMAGE $MCP_IMAGE $GITLAB_MCP_IMAGE; do
        if ! oc get imagestream "$image" &> /dev/null; then
            log_info "Creating ImageStream: $image"
            oc create imagestream "$image"
        else
            log_warning "ImageStream $image already exists"
        fi
    done

    log_success "ImageStreams ready"
}

# Build backend image
build_backend() {
    log_info "Building backend application..."

    # Build static content first
    log_info "Building static content with Antora..."
    if command -v npx &> /dev/null; then
        # Clean up any existing static content
        log_info "Cleaning existing www/ directory..."
        rm -rf "$SCRIPT_DIR/www"

        # Build fresh static content
        npx antora default-site.yml
        log_success "Static content built"
    else
        log_warning "npx not found, assuming static content is already built"
    fi

    # Start binary build
    log_info "Starting binary build for backend..."
    oc start-build "$BACKEND_IMAGE" --from-dir="$SCRIPT_DIR" --follow --wait

    log_success "Backend image built successfully"
}

# Build MCP server image
build_mcp() {
    log_info "Building MCP server..."

    # Start build (no source needed since Dockerfile is inline)
    log_info "Starting build for MCP server..."
    oc start-build "$MCP_IMAGE" --follow --wait

    log_success "MCP server image built successfully"
}

# Build GitLab MCP server image
build_gitlab_mcp() {
    log_info "Building GitLab MCP server..."

    # Start build (no source needed since Dockerfile is inline)
    log_info "Starting build for GitLab MCP server..."
    oc start-build "$GITLAB_MCP_IMAGE" --follow --wait

    log_success "GitLab MCP server image built successfully"
}


# Deploy backend resources
deploy_backend() {
    log_info "Deploying backend resources..."

    # Update namespace in backend manifests and apply
    local temp_dir=$(mktemp -d)

    # Copy and modify backend manifests
    cp -r k8s-backend/* "$temp_dir/"

    # Update image reference with actual namespace
    sed -i '' "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "$temp_dir/deployment.yaml"

    # Apply all backend manifests except namespace (we created our own)
    for file in "$temp_dir"/*.yaml; do
        filename="$(basename "$file")"
        if [[ "$filename" != "namespace.yaml" ]]; then
            log_info "Applying: $filename"
            oc apply -f "$file"
        fi
    done

    # Update secret with API key from .env.yaml
    local env_file="$SCRIPT_DIR/.env.yaml"
    if [[ -f "$env_file" ]]; then
        log_info "Reading API key from .env.yaml..."
        local api_key
        api_key=$(grep "^llm_api_key:" "$env_file" | cut -d'"' -f2)

        if [[ -n "$api_key" && "$api_key" != "your-openai-api-key-here" ]]; then
            log_info "Updating secret with API key..."
            oc set data secret/workshop-backend-secrets llm-api-key="$api_key"
            log_success "API key updated in secret"
        else
            log_warning "API key not found or still using dummy value in .env.yaml"
            log_warning "Please update .env.yaml with your actual OpenAI API key"
        fi
    else
        log_warning ".env.yaml file not found - secret will use empty API key"
        log_warning "Create .env.yaml with your OpenAI API key"
    fi

    # Clean up
    rm -rf "$temp_dir"

    log_success "Backend resources deployed"
}

# Deploy MCP server resources
deploy_mcp() {
    log_info "Deploying MCP server resources..."

    # Update namespace in MCP manifests and apply
    local temp_dir=$(mktemp -d)

    # Copy and modify MCP manifests
    cp -r k8s-mcp-server/* "$temp_dir/"

    # Update image reference with actual namespace
    sed -i '' "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "$temp_dir/deployment.yaml"

    # Update ClusterRoleBinding to include namespace for ServiceAccount
    sed -i '' "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "$temp_dir/rbac.yaml"

    # Apply all MCP manifests except namespace (we created our own)
    for file in "$temp_dir"/*.yaml; do
        filename="$(basename "$file")"
        if [[ "$filename" != "namespace.yaml" ]]; then
            log_info "Applying: $filename"
            oc apply -f "$file"
        fi
    done

    # Clean up
    rm -rf "$temp_dir"

    log_success "MCP server resources deployed"
}

# Deploy GitLab MCP server resources
deploy_gitlab_mcp() {
    log_info "Deploying GitLab MCP server resources..."

    # Update namespace in GitLab MCP manifests and apply
    local temp_dir=$(mktemp -d)

    # Copy and modify GitLab MCP manifests
    cp -r k8s-gitlab-mcp-server/* "$temp_dir/"

    # Update image reference with actual namespace
    sed -i '' "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "$temp_dir/deployment.yaml"

    # Apply all GitLab MCP manifests
    for file in "$temp_dir"/*.yaml; do
        filename="$(basename "$file")"
        log_info "Applying: $filename"
        oc apply -f "$file"
    done

    # Update GitLab secret with token from .env.yaml
    local env_file="$SCRIPT_DIR/.env.yaml"
    if [[ -f "$env_file" ]]; then
        log_info "Reading GitLab token from .env.yaml..."
        local gitlab_token
        gitlab_token=$(grep "^gitlab_personal_access_token:" "$env_file" | cut -d'"' -f2)

        if [[ -n "$gitlab_token" && "$gitlab_token" != "your-gitlab-token-here" ]]; then
            log_info "Updating GitLab secret with access token..."
            oc set data secret/gitlab-mcp-secrets gitlab-personal-access-token="$gitlab_token"
            log_success "GitLab token updated in secret"
        else
            log_warning "GitLab token not found or still using dummy value in .env.yaml"
            log_warning "Please update .env.yaml with your actual GitLab personal access token"
        fi
    else
        log_warning ".env.yaml file not found - GitLab secret will use empty token"
        log_warning "Create .env.yaml with your GitLab personal access token"
    fi

    # Clean up
    rm -rf "$temp_dir"

    log_success "GitLab MCP server resources deployed"
}


# Trigger rollout restart for all deployments
trigger_rollout_restart() {
    log_info "Triggering rollout restart for all deployments..."

    # Restart backend deployment
    log_info "Restarting backend deployment..."
    oc rollout restart deployment/workshop-backend

    # Restart MCP server deployment
    log_info "Restarting MCP server deployment..."
    oc rollout restart deployment/mcp-server

    # Restart GitLab MCP server deployment
    log_info "Restarting GitLab MCP server deployment..."
    oc rollout restart deployment/gitlab-mcp-server

    log_success "All deployment restarts triggered"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."

    # Wait for backend deployment
    log_info "Waiting for backend deployment..."
    oc rollout status deployment/workshop-backend --timeout=300s

    # Wait for MCP server deployment
    log_info "Waiting for MCP server deployment..."
    oc rollout status deployment/mcp-server --timeout=300s

    # Wait for GitLab MCP server deployment
    log_info "Waiting for GitLab MCP server deployment..."
    oc rollout status deployment/gitlab-mcp-server --timeout=300s

    log_success "All deployments are ready"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo

    log_info "Pods:"
    oc get pods -o wide
    echo

    log_info "Services:"
    oc get services
    echo

    log_info "Routes:"
    oc get routes
    echo

    # Get route URLs
    local backend_route=$(oc get route workshop-backend -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")
    local mcp_route=$(oc get route mcp-server -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")
    local gitlab_mcp_route=$(oc get route gitlab-mcp-server -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")

    log_success "Deployment completed successfully!"
    echo
    log_info "Access URLs:"
    echo "  Backend: https://$backend_route"
    echo "  Kubernetes MCP Server: https://$mcp_route"
    echo "  GitLab MCP Server: https://$gitlab_mcp_route"
}

# Main deployment function
main() {
    log_info "Starting Workshop deployment to OpenShift..."
    echo

    # Check prerequisites
    check_oc_login

    # Setup
    setup_namespace

    # Create build resources
    create_imagestreams
    create_backend_buildconfig
    create_mcp_buildconfig
    create_gitlab_mcp_buildconfig

    # Build images
    build_backend
    build_mcp
    build_gitlab_mcp

    # Deploy applications
    deploy_backend
    deploy_mcp
    deploy_gitlab_mcp

    # Trigger rollout restart to ensure fresh deployments
    trigger_rollout_restart

    # Wait for readiness
    wait_for_deployments

    # Show final status
    show_status
}

# Backend-only deployment function
deploy_backend_only() {
    log_info "Starting backend-only deployment to OpenShift..."
    echo

    # Check prerequisites
    check_oc_login

    # Switch to namespace (must already exist)
    if ! oc get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist. Run full deployment first."
        exit 1
    fi
    oc project "$NAMESPACE"
    log_success "Switched to existing namespace: $NAMESPACE"

    # Create backend build resources if needed
    if ! oc get imagestream "$BACKEND_IMAGE" &> /dev/null; then
        log_info "Creating backend ImageStream..."
        oc create imagestream "$BACKEND_IMAGE"
    fi

    if ! oc get buildconfig "$BACKEND_IMAGE" &> /dev/null; then
        log_info "Creating backend BuildConfig..."
        create_backend_buildconfig
    fi

    # Build backend image
    build_backend

    # Deploy backend resources
    deploy_backend

    # Restart backend deployment only
    log_info "Restarting backend deployment..."
    oc rollout restart deployment/workshop-backend

    # Wait for backend deployment
    log_info "Waiting for backend deployment..."
    oc rollout status deployment/workshop-backend --timeout=300s

    # Show status
    log_success "Backend-only deployment completed successfully!"
    echo
    log_info "Backend Status:"
    oc get pods -l app.kubernetes.io/name=workshop-backend -o wide
    echo
    local backend_route=$(oc get route workshop-backend -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")
    log_info "Backend URL: https://$backend_route"
}

# MCP-only deployment function (all MCP servers)
deploy_mcp_only() {
    log_info "Starting MCP-only deployment to OpenShift (all MCP servers)..."
    echo

    # Check prerequisites
    check_oc_login

    # Switch to namespace (must already exist)
    if ! oc get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist. Run full deployment first."
        exit 1
    fi
    oc project "$NAMESPACE"
    log_success "Switched to existing namespace: $NAMESPACE"

    # Create MCP build resources if needed
    for image in $MCP_IMAGE $GITLAB_MCP_IMAGE; do
        if ! oc get imagestream "$image" &> /dev/null; then
            log_info "Creating ImageStream: $image"
            oc create imagestream "$image"
        fi
    done

    # Always recreate MCP BuildConfigs to ensure they use latest Dockerfiles
    if oc get buildconfig "$MCP_IMAGE" &> /dev/null; then
        log_info "Deleting existing Kubernetes MCP BuildConfig to use latest Dockerfile..."
        oc delete buildconfig "$MCP_IMAGE"
    fi
    log_info "Creating Kubernetes MCP BuildConfig with current Dockerfile..."
    create_mcp_buildconfig

    if oc get buildconfig "$GITLAB_MCP_IMAGE" &> /dev/null; then
        log_info "Deleting existing GitLab MCP BuildConfig to use latest Dockerfile..."
        oc delete buildconfig "$GITLAB_MCP_IMAGE"
    fi
    log_info "Creating GitLab MCP BuildConfig with current Dockerfile..."
    create_gitlab_mcp_buildconfig

    # Build MCP server images
    build_mcp
    build_gitlab_mcp

    # Deploy MCP server resources
    deploy_mcp
    deploy_gitlab_mcp

    # Restart MCP deployments
    log_info "Restarting Kubernetes MCP server deployment..."
    oc rollout restart deployment/mcp-server

    log_info "Restarting GitLab MCP server deployment..."
    oc rollout restart deployment/gitlab-mcp-server

    # Wait for MCP deployments
    log_info "Waiting for Kubernetes MCP server deployment..."
    oc rollout status deployment/mcp-server --timeout=300s

    log_info "Waiting for GitLab MCP server deployment..."
    oc rollout status deployment/gitlab-mcp-server --timeout=300s

    # Show status
    log_success "MCP-only deployment completed successfully!"
    echo
    log_info "MCP Servers Status:"
    oc get pods -l app.kubernetes.io/component=mcp-server -o wide
    echo
    local mcp_route=$(oc get route mcp-server -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")
    local gitlab_mcp_route=$(oc get route gitlab-mcp-server -o jsonpath='{.spec.host}' 2>/dev/null || echo "No route found")
    log_info "Kubernetes MCP Server URL: https://$mcp_route"
    log_info "GitLab MCP Server URL: https://$gitlab_mcp_route"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "backend")
        deploy_backend_only
        ;;
    "mcp")
        deploy_mcp_only
        ;;
    "clean")
        log_warning "Cleaning up deployment..."
        oc delete namespace "$NAMESPACE" --ignore-not-found=true
        log_success "Cleanup completed"
        ;;
    "status")
        oc project "$NAMESPACE" 2>/dev/null || { log_error "Namespace $NAMESPACE not found"; exit 1; }
        show_status
        ;;
    *)
        echo "Usage: $0 [deploy|backend|mcp|clean|status]"
        echo "  deploy:  Deploy the workshop (default)"
        echo "  backend: Rebuild and redeploy only the backend service"
        echo "  mcp:     Rebuild and redeploy all MCP servers (Kubernetes + GitLab)"
        echo "  clean:   Delete the namespace and all resources"
        echo "  status:  Show current deployment status"
        exit 1
        ;;
esac