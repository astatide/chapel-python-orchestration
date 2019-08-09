#! /bin/bash -l
R=$(sed 's|C02XG1XXJGH5|0.0.0.0|g' <<< $1)
S=$(sed 's|C02XG1XXJGH5|0.0.0.0|g' <<< $2)

export PYTHONPATH=../lib/:$PYTHONPATH

python3 v.py --zmqPort $R #&> /dev/null
