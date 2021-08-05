import csv


symptomList = []
file = open('dataset.csv')
reader = csv.reader(file, delimiter=",")
for line in reader:
    symptomList.append(line[1:])
symptomSorted = {"null": 1}
for line in symptomList:
    for elem in line:
        if elem not in symptomSorted:
            symptomSorted[elem] = 1
        else:
            symptomSorted.get(elem) + 1
