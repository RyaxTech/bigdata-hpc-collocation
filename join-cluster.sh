#!/usr/bin/env bash
# Usage:
# sudo MASTER_NODE_IP=51.89.119.183 K8S_DISCOVERY_TOKEN=337971d51d4cdb65edbf7f2a5da78546d947f4e016c705e63ecb717ad1b2159d K8S_JOIN_TOKEN=huoqrt.vtbglb7cejba2k2y ./join-cluster.sh
set -e
set -u
set -x

if [ -z "$K8S_JOIN_TOKEN" ] || [ -z "$MASTER_NODE_IP" ] || [ -z "$NODE_IP" ] 
then
echo K8S_JOIN_TOKEN, NODE_IP and MASTER_NODE_IP must be set.
echo For more information see:
echo https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/#join-nodes
exit 1
fi

set -e
set -u

echo Cleaning previous configuration
yes | kubeadm reset || true

echo Check that services are installed properly
systemctl is-active --quiet sycri && echo Syngularity CRI is running || (echo ERROR: Singularity CRI is not running! && exit 1)

echo Generate config

TMP_DIR=$(mktemp -d)
cat <<EOF > $TMP_DIR/kubeadm-config.yaml
apiVersion: kubeadm.k8s.io/v1beta2
kind: JoinConfiguration
caCertPath: /etc/kubernetes/pki/ca.crt
discovery:
  bootstrapToken:
    apiServerEndpoint: "$MASTER_NODE_IP:6443"
    token: "$K8S_JOIN_TOKEN"
    unsafeSkipCAVerification: true
  timeout: 5m0s
nodeRegistration:
  criSocket: /var/run/singularity.sock
  name: $(hostname)
  taints: null
  # Add extra kubelet CLI arguments here
  kubeletExtraArgs:
    node-labels: runtime=singularity,type=hpc
    node-ip: "$NODE_IP"            

EOF

kubeadm join --config "$TMP_DIR/kubeadm-config.yaml"

