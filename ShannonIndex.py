#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:02:55 2019

@author: pepijn
"""
import numpy as np

def ShannonIndex(neighborhood):
    '''
    Calculates the Shannon index of a neighborhood
    '''
    total = len(neighborhood)
    states, counts = np.unique(neighborhood, return_counts=True)
    statusDict = dict(zip(states, counts))
    entropy = 0
    
    for state in statusDict:
        pi = statusDict[state]/total
        entropy =+ pi * np.log2(pi)   
    return -entropy

        
        
        
    