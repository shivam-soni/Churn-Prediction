# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:09:58 2018

@author: Shivam Soni
"""

import numpy as np
import math 
print("l1")
l1=[]
for i in range(10):
     l1.append(float(input()))
    
print (l1)

print("l2")  

l2=[]
for i in range(10):
     l2.append(float(input()))    

   
mean1=np.mean(l1)
mean2=np.mean(l2)

std1=np.std(l1)    
std2=np.std(l2)

var1=std1 **2
var2=std2 **2

temp1= var1/10
temp2= var2/10

t= abs(mean1 - mean2)/(math.sqrt(temp1 + temp2))
print("t=",t)