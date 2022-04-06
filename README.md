# Big Data and HPC Collocation with BeBiDa

This document explains how to install and configure the cluster in
order to achieve a dynamic resource sharing between two resource management
systems, one for the traditional HPC workload, the other for a HPDA workload.

We consider that the computing environment is composed of two separate
partitions: one for High Performance Big Data (HPDA) and one HPC partition. In
addition, the HPC partition has either Torque or Slurm as resource management
system installed.  Also, the HPDA partition is considered already managed by Kubernetes
which is setup and configured as described in the
[official installation documentation](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/).

> **WARNING**: Before starting this process, be aware that Slurm is considered
> already installed and configured on the HPC nodes and Kubernetes is
> considered already installed and configured on the HPDA nodes.

The resource sharing between the HPC and the HPDA partitions is implemented by
attaching the idle resources of the HPC partitions to the Kubernetes
cluster. This mechanism is implemented using the HPC resource manager (here
Slurm) prolog and epilog scripts: each HPC worker node is a Kubernetes worker
which is decommissioned if an HPC job requires the node and re-attached to the
pool of Kubernetes workers when the job finishes.


We first present an overview of the installation steps followed by more
detailed explanations:

- Install Singularity, Singularity-CRI and Kubernetes on the HPC nodes
- Attach the worker nodes to Kubernetes
- Configure Slurm to add prolog/epilog scripts
- Run an example Spark application

## Install HPC worker nodes

This script installs Singularity, Singularity-CRI and Kubernetes on the HPC nodes.

__WARNING__: Read this scripts carefully before executing it in order to adapt
it to your needs and your platform. *It was only made and tested for Ubuntu 20.04*

```sh
sudo ./install-node.sh
```

## Attach the worker node to Kubernetes

Now, we can attach our node to the Kubernetes cluster. In order to join the node to the cluster you need two token. To obtain these
token connect to your kubernetes master and run:
```sh
# Get the K8S_JOIN_TOKEN
kubeadm token create
```

Further details in the [Kubernetes
documentation](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/#join-nodes).

Then, return to the HPC node and run:
```sh
# WARNING Change these values!!!
export MASTER_NODE_IP=51.89.119.183
export K8S_JOIN_TOKEN=ctju7r.it7u0o2wmr7uh88x
export NODE_IP=192.168.100.1

sudo ./join-cluster.sh
```

Check that the services are up and running with:
```sh
systemctl status sycri.service
systemctl status kubelet.service
```

You can also check that your node is now present in the Kubernetes cluster:
```sh
kubectl get nodes
```

### Troubleshooting

If something went wrong during this step, you can retry the installation
process of the node but first consider reset the kubernetes configuration and
state with:

```sh
sudo kubeadm reset --cri-socket='unix:///var/run/singularity.sock'
sudo bash -c "iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X"
```

## Configure the HPC resource manager

The next step is to configure Slurm to enable the prolog and epilog scripts execution for each job.

Note that while this process is focused on SLurm, it can be adapted to any
resource manager that supports prolog/epilog.


### Configure master

We will now configure the master node of Slurm, the host of the `slumctld`
daemon so the master prolog can tell Kubernetes that the nodes that are part
the HPC job allocation must be decommissioned.

To do so, we need the access rights to the Kubernetes cluster to be able to
drain and delete nodes. To generate the proper access key with the associated
user, you can run this script on the kubernetes master (or on any place where
you have admin access it with `kubectl`):
```sh
create-delete-node-config.sh
```

Now you can transfer the config `workflow-manager-1-kubeconfig.yaml` and the
key file `workflow-manager-1.key`  along with the master prolog and epilog
`master-*log.sh` into your Slurm master.


Then, connect to the master and enable the user with:
```sh
kubectl config set-credentials workflow-manager-1 --kubeconfig workflow-manager-1-kubeconfig.yaml --client-key workflow-manager-1.key --embed-certs=true
```

Then, put them in a place where Slurm can access them (as root): 
```
cp master-prolog.sh /usr/local/bin/
cp master-epilog.sh /usr/local/bin/
cp workflow-manager-1* /usr/local/etc/
chown slurm:slurm /usr/local/etc/workflow-manager-1*
mkdir -p /usr/local/logs
chown slurm:slurm /usr/local/logs
```

Finally, append these lines to the Slurm configuration `/etc/slurm-llnl/slurm.conf` in order to
enable the prolog/epilog.
```
PrologSlurmctld=/usr/local/bin/master-prolog.sh
EpilogSlurmctld=/usr/local/bin/master-epilog.sh
Epilog=/usr/local/bin/epilog.sh
Prolog=/usr/local/bin/prolog.sh
PrologFlags=Alloc
```

The master prolog and epilog have some dependencies so install them:
> WARNING: Adapt this script to your infrastructure: *This was only tested on Ubuntu 20.04*
```sh
./install-master.sh
```

Restart Slurm controler and node daemons:
```
systemctl restart slurmctld.service
systemctl restart slurmd.service
```

### Configure workers

First, copy this new confguration from the master to all nodes.

Then, copy the `prolog.sh` and `epilog.sh` scripts to the nodes in `/usr/local/bin`,
 make it accessible by Slurm and keep them executable. These scripts are very simple, and only starts and stops the
Kubernetes daemons:

```sh
cat > /usr/local/bin/epilog.sh <<EOF
#!/usr/bin/env bash

systemctl restart sycri.service
systemctl restart kubelet.service
EOF
chmod +x /usr/local/bin/epilog.sh

cat > /usr/local/bin/prolog.sh <<EOF
#!/usr/bin/env bash

systemctl stop kubelet.service
systemctl stop sycri.service
EOF
chmod +x /usr/local/bin/prolog.sh
```

Restart Slurm on all nodes to the new configuration.
```
systemctl restart slurmd.service
```

## Running Spark Example Application

### Configure Kubernetes to be the Spark Cluster Manager

First, Spark needs to be configured to use Kubernetes as a cluster manager.
Official documentation can be found [here](https://spark.apache.org/docs/latest/running-on-kubernetes.html).

As explained in the official documentation, on Kubernetes with RBAC enabled, it
is required to create service account for Spark drivers.
```sh
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```

### Run the Spark example

Now that system is configured, we will run an example application to test our
setup.

We wiil run a Spark application on the cluster using K8s has an external Scheduler for it. 
The `spark-submit-pod.yaml` provide an application definition but it has to
customize regarding the Kubernetes master IP and the node that will be used to
deploy the Spark driver. 

> WARNING: It if use this file untouch it won't work!

One configured run the Spark application on the cluster simply with:
```sh
kubectl apply -f spark-submit-pod.yaml
```

In order to see the dynamic resource allocation in action, you can create an
HPC job in parallel with:
```sh
srun -N2 sleep 20
```

If any executors run on a node allocated to your HPC job, it will be deleted
and replace by another allocation on another available resource and both jobs
should run successfully.

### (OPTIONAL) Build the Spark application container image

> This is only required to update the Spark engine. See the current version in
> the `Dockerfile`.

Edit the `Dockerfile` to do your updates.

Run the build with (be sure to update the version)
```sh
docker build . -t spark-on-k8s
```

Tag it with an updated version and push it to your repository (here the ryax
one):
```sh
docker tag spark-on-k8s ryaxtech/spark-on-k8s:v0.1.0
docker push ryaxtech/spark-on-k8s:v0.1.0
```
