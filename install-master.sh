#!/usr/bin/env bash

echo WARNING: This script need to be executed with sudo rights


curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update

apt install -y kubectl=1.19.16-00
