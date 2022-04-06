#!/usr/bin/env bash

set -x
set -e

export SCRIPT_PATH=$(cd $(dirname $0) >/dev/null 2>&1 && pwd )
(
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export KUBECONFIG=$SCRIPT_PATH/../etc/workflow-manager-1-kubeconfig.yaml

printenv
export SLURM_NODELIST=$SLURM_JOB_NODELIST
for node in $(scontrol show hostnames)
do
        kubectl drain --force --grace-period=5 --ignore-daemonsets --delete-local-data $node
done
) > $SCRIPT_PATH/../logs/${SLURM_JOB_ID}-prolog-logs.out 2> $SCRIPT_PATH/../logs/${SLURM_JOB_ID}-prolog-logs.err
