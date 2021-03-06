# -*- coding: utf-8 -*-
"""
Load Data for ppLFERMUM Model
Created on Fri Jun 15 14:51:15 2018

@author: Tim Rodgers
"""
#Import packages
import pandas as pd

#For ppLFER-MUM
chemsumm = pd.read_excel('OPECHEMSUMM.xlsx') #Import excel files as csv tends to truncate
#Location summary for the modelled area. Descriptors should be in the first column (0)
locsumm = pd.read_excel('locsumm.xlsx',index_col = 0) 
#parameters must be loaded with the descriptor in the first column (0) and the values in a "Value" column
params = pd.read_excel('params.xlsx',index_col = 0) 