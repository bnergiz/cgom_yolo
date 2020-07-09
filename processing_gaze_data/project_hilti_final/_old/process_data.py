"""
ST: 'cGOM'
Berk Nergiz
pdz, ETH Zürich
2020

This file transforms eye tracking data from Tobii Pro Recordings.
"""

# Global imports
import os
import sys
import cv2
import numpy as np
import json

# Local imports
# import utils

path = 'Users/berkn/Desktop/ETH/Master/Semester_2/Semester_Project/cgom/gaze'
  
  
# the file to be converted to  
# json format 
filename = 'new_file.txt'

# intermediate and resultant dictionaries 
# intermediate 
dict2 = {} 
  
# resultant 
dict1 = {}

output = []
  
# fields in the sample file  
fields =['Start', 'End', 'Duration', 'x', 'y'] 
  
with open(filename) as fh: 
      
    # loop variable 
    i = 0
      
    # count variable for employee id creation 
    l = 1
      
    for line in fh:
          
        # reading line by line from the text file 
        description = list( line.strip().split(None, 4))
        output.append(list( line.strip().split(None, 4)))
          
        # for output see below 
        print(description)
          
        # for automatic creation of id for each employee 
        sno ='line'+str(l) 
      
        while i<len(fields):
              
                # creating dictionary for each employee 
                dict2[fields[i]]= description[i] 
                i = i + 1
                  
        # appending the record of each employee to 
        # the main dictionary 
        dict1[sno]= dict2 
        l = l + 1
    

# creating json file         
out_file = open("raw.json", "w")
json.dump(output[1:], out_file, indent = 4) 
out_file.close()

print(output[1][0])

output = output[1:]

output_2 = []

o = 0

fixation_counter = 0
fixation = []

for x in range(len(output)-1):
    if output[o+1][4] != output[o][4]:
        fixation_counter = fixation_counter + 1
        fixation.append(o)
    o = o+1

print(fixation_counter)

output_2.append(output[0])

k = 0

for i in range(len(fixation)-1):
    # update the end time
    output_2[k][1] = output[fixation[k]][1]
    # calculate duration
    output_2[k][2] = float(output_2[k][1]) - float(output_2[k][0])
    # initialize output next line
    output_2.append(output[fixation[k]+1])
    k = k+1

k = 0

start = []
end = []
duration = []
x = []
y = []

for i in range(len(fixation)-1):
    start.append(output_2[k][0])
    end.append(output_2[k][1])
    duration.append(output_2[k][2])
    x.append(output_2[k][3])
    y.append(output_2[k][4])
    k = k+1

out_file = open("first.json", "w")
json.dump(output_2, out_file, indent = 4) 
out_file.close()

out_file = open("start.txt", "w")
json.dump(start, out_file, indent = 4) 
out_file.close()

out_file = open("end.txt", "w")
json.dump(end, out_file, indent = 4) 
out_file.close()

out_file = open("duration.txt", "w")
json.dump(duration, out_file, indent = 4) 
out_file.close()

out_file = open("x.txt", "w")
json.dump(x, out_file, indent = 4) 
out_file.close()

out_file = open("y.txt", "w")
json.dump(y, out_file, indent = 4) 
out_file.close()
