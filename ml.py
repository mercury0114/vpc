import csv

f = open("./../data/statistics/merged_fiber.txt", "r")
next(f)
data = [[float(x) for x in line.split(",")] for line in f]

dictionary = {}
for row in data:
    if (row[1] == 0):
        dictionary[row[0]] = row[2:]

print(len(dictionary))
