import csv
import numpy
import math

# import csv here
holdstatic_20 = []
inpocket_26 = []
inpocket_36 = []
swing_32 = []
swing_38 = []

# these will hold magnitudes of acceleration
m_holdstatic_20  = []
m_inpocket_26 = []
m_inpocket_36 = []
m_swing_32 = []
m_swing_38 = []

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

for x in range(len(holdstatic_20)):
    m_holdstatic_20.append(math.sqrt((holdstatic_20[x][0])**2 + (holdstatic_20[x][1])**2 + (holdstatic_20[x][2])**2))

for x in range(len(inpocket_26)):
    m_inpocket_26.append(math.sqrt((inpocket_26[x][0])**2 + (inpocket_26[x][1])**2 + (inpocket_26[x][2])**2))

for x in range(len(inpocket_36)):
    m_inpocket_36.append(math.sqrt((inpocket_36[x][0])**2 + (inpocket_36[x][1])**2 + (inpocket_36[x][2])**2))

for x in range(len(swing_32)):
    m_swing_32.append(math.sqrt((swing_32[x][0])**2 + (swing_32[x][1])**2 + (swing_32[x][2])**2))

for x in range(len(swing_38)):
    m_swing_38.append(math.sqrt((swing_38[x][0])**2 + (swing_38[x][1])**2 + (swing_38[x][2])**2))

