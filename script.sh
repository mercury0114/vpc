#!/bin/bash

filename='./../data/gIDs.txt'
while read line; do
python2 createMask.py -g $line
python2 getStatistics.py -g $line
done < $filename
