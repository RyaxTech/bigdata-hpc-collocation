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
        kubectl uncordon $node
done
) > $SCRIPT_PATH/../logs/${SLURM_JOB_ID}-epilog-logs.out 2> $SCRIPT_PATH/../logs/${SLURM_JOB_ID}-epilog-logs.err
