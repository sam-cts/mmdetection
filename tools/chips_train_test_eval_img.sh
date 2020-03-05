#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1

$PYTHON -u tools/train.py $CONFIG 