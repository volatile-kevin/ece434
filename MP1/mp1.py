import csv
import numpy

holdstatic_20 = []
inpocket_26 = []
inpocket_36 = []
swing_32 = []
swing_38 = []

with open("sample_data/holdstatic_20steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        holdstatic_20.append(row)
with open("sample_data/inpocket_26steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inpocket_26.append(row)
with open("sample_data/inpocket_36steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inpocket_36.append(row)
with open("sample_data/swing_32steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        swing_32.append(row)
with open("sample_data/swing_38steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        swing_38.append(row)