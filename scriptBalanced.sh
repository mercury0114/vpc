#!/bin/bash
fibersDir='./../data/fibersBalancedMindaugo/'
python createMaskBalanced.py './../data/modelMindaugo.h5' $fibersDir
python getStatisticsBalanced.py './../data/statisticsBalancedMindaugo.txt' $fibersDir
