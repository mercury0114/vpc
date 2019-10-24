import csv

f = open("./../data/statistics/merged_fiber.txt", "r")
next(f)
data = [[float(x) for x in line.split(",")] for line in f]
dictionary = {}
for row in data:
    if (int(row[1]) == 0):
        dictionary[int(row[0])] = row[2:]

f = open("./../data/survival.txt")
survival = [line.split(",") for line in f]
output = {}
for entry in survival:
    output[int(entry[0])] = int(entry[2])

X = []
y = []
for key in output.keys():
    X.append(dictionary[key])
    y.append(output[key])

