# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:32:21 2018

@author: Tim Rodgers
"""
def forward_calc(self,ic,num_compartments):
        """ Perform forward calculations to determine model concentrations
        based on input emissions. initial_calcs (ic) are calculated at the initialization
        of the model and include the matrix values DTi, and D_ij for each compartment 
        num_compartments (numc) defines the size of the matrix
        """
    #Determine number of chemicals
    numchems = 0
    for chems in ic.Compound:
        numchems = numchems + 1
            
        #Initialize dataframe of n x n D values, D_mat
        #D_mat = pd.DataFrame(index = range(num_compartments),columns = range(num_compartments))
        #Initialize 3d DataArray with numchem data varaiables and coordinates of 7 x 7 (figure out a better way (no panels) later)
        #D_array = pd.Panel(items = ic.Compound,major_axis = range(num_compartments),minor_axis = range(num_compartments)).to_xarray()
        #Initialize output
    fw_out = pd.DataFrame(ic['Compound'].copy(deep=True))
        #Initialize a blank matrix of D values. We will iterate across this to solve for each compound
    D_mat = pd.DataFrame(index = range(num_compartments),columns = range(num_compartments))
    for chem in ic.Compound:
            #generate matrix. Names of D values in ic must conform to these labels
            #DTj for output and D_jk for transfer between compartments j and k
        for j in D_mat.index: #compartment j, index of D_mat
            for k in D_mat.columns: #compartment k, column of D_mat
                if j == k:
                    DT = 'DT' + str(j + 1)
                    D_mat.iloc[j,k] = -ic.loc[chem,DT]
                else:
                    D_val = 'D_' +str(j+1)+str(col+1) #label compartments from 1
                    if D_val in ic.columns: #Check if there is transfer between the two compartments
                        D_mat.iloc[j,col] = ic.loc[chem,D_val]
                    else:
                        D_mat.iloc[j,col] = 0 #If no transfer, set to 0
    