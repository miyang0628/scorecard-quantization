#!/usr/bin/env bash
# Downloads the four public credit data sets into ./data.
# The data are not redistributed with this repository; check each source's terms.
set -euo pipefail
mkdir -p data
# Taiwan default of credit card clients (Yeh and Lien, 2009); UCI copy hosted by scikit-learn on Hugging Face
curl -L -o data/taiwan.csv "https://huggingface.co/datasets/scikit-learn/credit-card-clients/resolve/main/UCI_Credit_Card.csv"
# Give Me Some Credit (Kaggle, 2011): training file cs-training.csv.
# Preferred: download from https://www.kaggle.com/c/GiveMeSomeCredit and save as data/gmsc.csv.
# A public mirror is used here for convenience:
curl -L -o data/gmsc.csv "https://raw.githubusercontent.com/jalajthanaki/credit-risk-modelling/master/data/cs-training.csv"
# FICO HELOC (FICO, 2018), original heloc_dataset_v1.csv
curl -L -o data/heloc_v1.csv "https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv"
# Statlog German credit (Hofmann, 1994), the labelled copy shipped with the scorecardpy package
pip download --no-deps -d /tmp/scpy scorecardpy==0.1.9.7 >/dev/null
tar -xzf /tmp/scpy/scorecardpy-0.1.9.7.tar.gz -C /tmp/scpy scorecardpy-0.1.9.7/scorecardpy/data/germancredit.csv
cp /tmp/scpy/scorecardpy-0.1.9.7/scorecardpy/data/germancredit.csv data/german.csv
echo "data ready"
