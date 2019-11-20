#!/bin/bash
fibersDir='./../data/fibersMariaus/'
python createMask1500.py './../data/modelMariaus.h5' $fibersDir
python getStatistics1500.py './../data/statisticsMariaus.txt' $fibersDir
