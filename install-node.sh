#!/usr/bin/env bash

echo WARNING: This script need to be executed with sudo rights

TMP_DIR=$(mktemp -d)
set -e
set -u
set -x

export PREFIX="/usr/local"
mkdir -p $PREFIX

SINGULARITY_RELEASE="3.9.4"
K8S_RELEASE="v1.19.16"
CNI_PLUGINS_VERSION="0.8.5"
GO_VERSION="1.16.8"

NODE_TYPE=${NODE_TYPE:-hpc} # or bigdata

# Set the local IP address before running the script
MAIN_DEV=$(ip -4 route ls | grep default | head -1 | grep -Po '(?<=dev )(\S+)')
NODE_IP_ADDRESS=$(ip -f inet -o addr show dev "$MAIN_DEV" | cut -d\  -f 7 | cut -d/ -f 1)
echo Selected IP address: "$NODE_IP_ADDRESS"

### Install dependencies ###

# install dependencies
INSTALL_DEPS="curl wget build-essential apt-transport-https ca-certificates gnupg2 software-properties-common"
# Singularity dependencies
SINGULARITY_DEPS="libseccomp-dev pkg-config squashfs-tools cryptsetup"

### WARNING: Dabian based System only
# Clean previous installation
apt remove --purge kubelet kubectl kubeadm kubernetes-cni cri-tools -y || true
apt-get update
apt-get install -y $INSTALL_DEPS $SINGULARITY_DEPS

# Chose one compatible version from
# apt list -a docker-ce
#curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
#add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update

#apt install docker-ce=5:20.10.12~3-0~ubuntu-focal
apt install -y kubeadm=1.19.16-00 kubelet=1.19.16-00 kubectl=1.19.16-00

# Go installation
# install go
export GO_VERSION=$GO_VERSION OS=linux ARCH=amd64
#if [ ! -d "$PREFIX/go" ] || [ -z $(cat "$PREFIX/go/VERSION" | grep "$GO_VERSION") ]
#then
wget -q -O "$TMP_DIR/go${GO_VERSION}.${OS}-${ARCH}.tar.gz" "https://dl.google.com/go/go${GO_VERSION}.${OS}-${ARCH}.tar.gz"

rm -rf "$PREFIX/go"
tar -C "$PREFIX" -xzf "$TMP_DIR/go${GO_VERSION}.${OS}-${ARCH}.tar.gz"

# configure environment
export GOPATH=${HOME}/go
export PATH=${PATH}:/usr/local/go/bin:${GOPATH}/bin

if ! grep "ADDED BY INSTALL NODE" ~/.bashrc
then
cat >> ~/.bashrc <<EOF
# ADDED BY INSTALL NODE
export GOPATH=${GOPATH}
export PATH=${PATH}
EOF
fi
#fi

### For centos use:

# yum groupinstall -y 'Development Tools' && \
# yum install -y epel-release && \
# yum install -y golang libseccomp-devel \
# squashfs-tools cryptsetup

### Install Singularity ###
## Install Singularity and Singularity CRI
#curl -L "https://github.com/sylabs/singularity/releases/download/v${SINGULARITY_RELEASE}/singularity-${SINGULARITY_RELEASE}.tar.gz" | tar -C "$TMP_DIR" -xz
#cd ${TMP_DIR}/singularity && ./mconfig --prefix=${PREFIX} && cd ./builddir &&  make && make install
wget -q -O $TMP_DIR/singurarity.deb https://github.com/sylabs/singularity/releases/download/v${SINGULARITY_RELEASE}/singularity-ce_${SINGULARITY_RELEASE}-focal_amd64.deb 
dpkg -i $TMP_DIR/singurarity.deb

# install singularity-cri
SINGULARITY_CRI_REPO="https://github.com/RyaxTech/singularity-cri"
git clone ${SINGULARITY_CRI_REPO} ${TMP_DIR}/singularity-cri
rm -f $PREFIX/bin/sycri || true
cd ${TMP_DIR}/singularity-cri && make && INSTALL_DIR=$PREFIX/bin make install

# set up CNI config
mkdir -p /etc/cni/net.d
cat > /etc/cni/net.d/11_bridge.conflist <<EOF
{
    "cniVersion": "0.3.1",
    "name": "bridge",
    "plugins": [
        {
            "type": "loopback"
        },
        {
            "type": "bridge",
            "bridge": "cbr0",
            "isGateway": true,
            "isDefaultGateway": true,
            "ipMasq": true,
            "capabilities": {"ipRanges": true},
            "ipam": {
                "type": "host-local",
                "routes": [
                    { "dst": "0.0.0.0/0" }
                ]
            }
        },
        {
            "type": "portmap",
            "capabilities": {"portMappings": true},
            "snat": true
        }
    ]
}
EOF

cat > /usr/local/bin/clean-cni.sh <<EOF
#!/usr/bin/env bash
ip addr flush cni0 || true
ip addr flush cbr0 || true
EOF
chmod +x /usr/local/bin/clean-cni.sh

# set up sycri service
cat > /etc/systemd/system/sycri.service <<EOF
[Unit]
Description=Singularity-CRI
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
ExecStart=/usr/local/bin/sycri
ExecStartPre=/usr/local/bin/clean-cni.sh
Environment="PATH=/usr/local/libexec/singularity/bin:/bin:/sbin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

[Install]
WantedBy=multi-user.target
EOF

# configure crictl
touch /etc/crictl.yaml
chown $USER:$USER /etc/crictl.yaml
cat > /etc/crictl.yaml << EOF
runtime-endpoint: unix:///var/run/singularity.sock
image-endpoint: unix:///var/run/singularity.sock
timeout: 10
debug: false
EOF

# configure system network config
cat > /etc/modules-load.d/k8s.conf <<EOF
br_netfilter
EOF
cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
EOF
modprobe br_netfilter
sysctl -w net.bridge.bridge-nf-call-ip6tables=1
sysctl -w net.bridge.bridge-nf-call-iptables=1
sysctl -w net.ipv4.ip_forward=1

# Install kubeadm systemd override file for kubeadm join to work
cat <<EOF >/etc/systemd/system/kubelet.service.d/10-kubeadm.conf
[Service]
Environment="KUBELET_KUBECONFIG_ARGS=--bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf --kubeconfig=/etc/kubernetes/kubelet.conf"
Environment="KUBELET_CONFIG_ARGS=--config=/var/lib/kubelet/config.yaml"
# This is a file that "kubeadm init" and "kubeadm join" generate at runtime, populating the KUBELET_KUBEADM_ARGS variable dynamically
EnvironmentFile=-/var/lib/kubelet/kubeadm-flags.env
# This is a file that the user can use for overrides of the kubelet args as a last resort. Preferably,
# the user should use the .NodeRegistration.KubeletExtraArgs object in the configuration files instead.
# KUBELET_EXTRA_ARGS should be sourced from this file.
EnvironmentFile=-/etc/default/kubelet
ExecStart=
ExecStart=/usr/bin/kubelet \$KUBELET_KUBECONFIG_ARGS \$KUBELET_CONFIG_ARGS \$KUBELET_KUBEADM_ARGS \$KUBELET_EXTRA_ARGS
EOF

systemctl enable --now sycri.service
systemctl enable --now kubelet
systemctl disable --now ufw.service
