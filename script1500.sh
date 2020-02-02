#!/bin/bash
fibersDir='./../data/fibers1500Mariaus/'
python createMask1500.py './../data/modelMariaus.h5' $fibersDir
python getStatistics1500.py './../data/survival/statistics1500Mariaus.txt' $fibersDir
