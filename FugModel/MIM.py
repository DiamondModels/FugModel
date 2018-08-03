# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:22:09 2018

@author: Tim Rodgers
"""
import numpy as np
import pandas as pd
from HelperFuncs import vant_conv, arr_conv
from FugModel import FugModel

class MIM(FugModel):
    
    """Multimedia Indoor Model fugacity model object. Implementation of the model by
    ?? as updated by Adjei-Kyereme (2018) and Kvasnicka (in prep)
        
    Attributes:
    ----------
            ic input_calc (df): Dataframe describing the system up to the point 
            of matrix solution, which includes D values DTi and D_IJ
    """
    def __init__(self,locsumm,chemsumm,params,num_compartments = 6,name = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params)
        
    def input_calc(self,locsumm,chemsumm,params,pp):
        """ Perform the initial calulations to set up the fugacity matrix. A steady state
        MIM object is an n compartment fugacity model solved at steady
        state using the compartment parameters from locsumm and the chemical
        parameters from chemsumm, other parameters from params
        """
        #Declare constants
        R = 8.314 #Ideal gas constant, J/mol/K
        #Initialize results by copying the chemsumm dataframe
        res = pd.DataFrame.copy(chemsumm,deep=True)
        
        return res
    