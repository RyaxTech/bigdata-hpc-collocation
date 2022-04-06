#!/usr/bin/env bash

set -e
set -u

USER_NAME=workflow-manager-1
GROUP_NAME=workflow-manager

TMP_DIR=$(mktemp -d)

echo Check dependencies...
which openssl > /dev/null  || (echo Please install openssl && exit 1)
which kubectl > /dev/null  || (echo Please install kubectl && exit 1)
which jq > /dev/null || (echo Please install jq && exit 1)

openssl genrsa -out $USER_NAME.key 4096

cat > $TMP_DIR/csr_$USER_NAME.cnf <<EOF
[ req ]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn

[ dn ]
CN = $USER_NAME
O = $GROUP_NAME

[ v3_ext ]
authorityKeyIdentifier=keyid,issuer:always
basicConstraints=CA:FALSE
keyUsage=keyEncipherment,dataEncipherment
extendedKeyUsage=serverAuth,clientAuth
EOF

openssl req -config $TMP_DIR/csr_$USER_NAME.cnf -new -key $USER_NAME.key -nodes -out $TMP_DIR/$USER_NAME.csr

BASE64_CSR=$(cat $TMP_DIR/$USER_NAME.csr | base64 | tr -d '\n')

cat > $TMP_DIR/csr_$USER_NAME.yaml <<EOF
apiVersion: certificates.k8s.io/v1beta1
kind: CertificateSigningRequest
metadata:
  name: ${USER_NAME}-csr
spec:
  groups:
  - system:authenticated
  request: ${BASE64_CSR}
  usages:
  - digital signature
  - key encipherment
  - server auth
  - client auth
EOF

kubectl delete csr ${USER_NAME}-csr || true
kubectl apply -f $TMP_DIR/csr_$USER_NAME.yaml
kubectl get csr ${USER_NAME}-csr
kubectl certificate approve ${USER_NAME}-csr

cat > $TMP_DIR/node-deleter.yaml <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-deleter
rules:
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "delete", "patch"]
# Needed for drain
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["list"]
- apiGroups: [""]
  resources: ["pods/eviction"]
  verbs: ["create"]
- apiGroups: ["apps"]
  resources: ["daemonsets"]
  verbs: ["get", "list", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: delete-nodes
subjects:
- kind: User
  name: $USER_NAME
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: node-deleter
  apiGroup: rbac.authorization.k8s.io
EOF

kubectl apply -f $TMP_DIR/node-deleter.yaml

# User identifier
CLUSTER_NAME=$(kubectl config view --minify -o jsonpath={.current-context} | cut -d@ -f 2)
# Client certificate
CLIENT_CERTIFICATE_DATA=$(kubectl get csr ${USER_NAME}-csr -o jsonpath='{.status.certificate}')
# Cluster Certificate Authority
CLUSTER_CA=$(kubectl config view --raw -o json | jq -r '.clusters[] | select(.name == "'$CLUSTER_NAME'") | .cluster."certificate-authority-data"')
# API Server endpoint
CLUSTER_ENDPOINT=$(kubectl config view --raw -o json | jq -r '.clusters[] | select(.name == "'$CLUSTER_NAME'") | .cluster."server"')

cat > $USER_NAME-kubeconfig.yaml <<EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: ${CLUSTER_CA}
    server: ${CLUSTER_ENDPOINT}
  name: ${CLUSTER_NAME}
users:
- name: ${USER_NAME}
  user:
    client-certificate-data: ${CLIENT_CERTIFICATE_DATA}
contexts:
- context:
    cluster: ${CLUSTER_NAME}
    user: ${USER_NAME}
  name: ${USER_NAME}@${CLUSTER_NAME}
current-context: ${USER_NAME}@${CLUSTER_NAME}
EOF

echo The User configuration file:
echo   $USER_NAME-kubeconfig.yaml
echo The user secrete key:
echo   $USER_NAME.key
echo To activate the user credentials use:
echo   kubectl config set-credentials $USER_NAME --kubeconfig $USER_NAME-kubeconfig.yaml --client-key $USER_NAME.key --embed-certs=true

