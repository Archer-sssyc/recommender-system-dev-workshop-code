#!/usr/bin/env bash
set -e

export SECRET_NAME=gcr-rs-dev-workshop-github
export GITHUB_USER=Archer-sssyc
export ACCESS_TOKEN=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
export APP_CONF_REPO=recommender-system-dev-workshop-code

input=$1

if [ $input = "online-codebuild" ]
then
    echo "start create online codebuild project!"
    ./online-code-build-setup.sh
elif [ $input = "deploy-offline" ]
then
    echo "start create offline codebuild project!"
    ./create-offline.sh    
elif [ $input = "infra" ]
then
    echo "start create online infrastructure!"
    ./create-online-infra.sh
elif [ $input = "config" ]
then
    echo "start update online config!"
    ./update-online-config.sh
elif [ $input = "argo-server" ]
then
    echo "start setup argocd server!"
    ./setup-argocd-server.sh
elif [ $input = "application" ]
then
    echo "start create application!"
    ./create-argocd-application.sh
else
    echo "Please enter correct parameter"
fi  