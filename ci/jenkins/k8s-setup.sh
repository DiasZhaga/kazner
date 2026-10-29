#!/usr/bin/env bash
# Create the kazner-test namespace and the namespace-scoped deployer account, then write a
# kubeconfig for Jenkins to ci/jenkins/secrets/kubeconfig (ignored by git).
#
#   bash ci/jenkins/k8s-setup.sh            # Docker Desktop Kubernetes (context docker-desktop)
#   KUBE_CONTEXT=minikube KUBE_SERVER=https://<ip>:8443 bash ci/jenkins/k8s-setup.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ctx="${KUBE_CONTEXT:-docker-desktop}"
# The API server as seen from the Jenkins container; its certificate is issued for this name.
server="${KUBE_SERVER:-https://kubernetes.docker.internal:6443}"

kubectl --context "$ctx" apply -f "$repo/deploy/k8s/rbac.yaml"

token=""
for _ in $(seq 30); do
  token="$(kubectl --context "$ctx" -n kazner-test get secret jenkins-deployer-token \
    -o jsonpath='{.data.token}' 2>/dev/null | base64 -d || true)"
  [ -n "$token" ] && break
  sleep 1
done
[ -n "$token" ] || { echo "the service-account token was not issued" >&2; exit 1; }
ca="$(kubectl --context "$ctx" -n kazner-test get secret jenkins-deployer-token \
  -o jsonpath='{.data.ca\.crt}')"

mkdir -p "$here/secrets"
umask 077
cat > "$here/secrets/kubeconfig" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: kazner-test-cluster
    cluster:
      server: $server
      certificate-authority-data: $ca
users:
  - name: jenkins-deployer
    user:
      token: $token
contexts:
  - name: kazner-test
    context:
      cluster: kazner-test-cluster
      user: jenkins-deployer
      namespace: kazner-test
current-context: kazner-test
EOF
echo "wrote $here/secrets/kubeconfig (namespace kazner-test, server $server)"
