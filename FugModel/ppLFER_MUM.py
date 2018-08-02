# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:48:58 2018

@author: Tim Rodgers
"""
import numpy as np
import pandas as pd
from HelperFuncs import ppLFER, vant_conv, arr_conv, make_ppLFER
from FugModel import FugModel


class ppLFERMUM(FugModel):
    
    """ ppLFER based Multimedia Urban Model fugacity model object. Implementation of the model by
    Diamond et al (2001) as updated by Rodgers et al. (2018)
        
    Attributes:
    ----------
            pplfer_system (df): (optional) ppLFER system parameters, with 
            columns as systems(Kij or dUij) and the row index as l,s,a,b,v,c 
            e.g pp = pd.DataFrame(index = ['l','s','a','b','v','c']) by default
            the system will define the ppLFER system as per Rodgers et al. (2018)
            ic input_calc (df): Dataframe describing the system up to the point 
            of matrix solution. 
    """
    
    def __init__(self,locsumm,chemsumm,params,num_compartments = 7,name = None,pplfer_system = None):
        FugModel. __init__(self,locsumm,chemsumm,params,num_compartments,name)
        self.pp = pplfer_system
        self.ic = self.input_calc(self.locsumm,self.chemsumm,self.params,self.pp)

        
    def input_calc(self,locsumm,chemsumm,params,pp):
        """ Perform the initial calulations to set up the fugacity matrix. A steady state
        ppLFERMUM is an n compartment fugacity model solved at steady
        state using the compartment parameters from locsumm and the chemical
        parameters from chemsumm, other parameters from params, and a ppLFER system
        from pp, use pp = None to use the defaults. 
        """
        #pdb.set_trace()
        #Initialize used inputs dataframe with input properties
        ic_inp = pd.DataFrame.copy(chemsumm,deep=True)        
        #Declare constants and calculate non-chemical dependent parameters
        #Should I make if statements here too? Many of the params.Value items could be here instead.
        R = 8.314 #Ideal gas constant, J/mol/K
        locsumm.loc[:,'V']= locsumm.Area*locsumm.Depth #Add volumes  m³
        params.loc['TempK','Value'] = params.Value['Temp'] +273.15 #°C to K
        #Calculate air density kg/m^3
        locsumm.loc[['Lower_Air','Upper_Air'],'Density'] = 0.029 * 101325 / (R * params.Value.TempK)
        Y4 = locsumm.Depth.Soil/2 #Soil diffusion path length (m)
        Y5 = locsumm.Depth.Sediment/2 #Sediment diffusion path length (m)
        #Boundary layer depth - leaves & film (m) Nobel (1991)
        delta_blv = 0.004 * ((0.07 / params.Value.WindSpeed) ** 0.5)
        delta_blf = 0.006 * ((0.07 / params.Value.WindSpeed) ** 0.5) 
        #Film to water MTC (m/h)
        kfw = params.Value.FilmThickness * params.Value.W
        #Dry deposition interception fraction (Diamond, Premiere, & Law 2001)
        Ifd = 1 - np.exp(-2.8 * params.Value.Beta)
        #Soil to groundwater leaching rate from Mackay & Paterson (1991)
        Usg = 0.4 * params.Value.RainRate
        #Fraction soil volume occupied by interstitial air and water
        ic_inp.loc[:,'Bea'] = ic_inp.AirDiffCoeff*locsumm.VFAir.Soil**(10/3) \
            /(locsumm.VFAir.Soil +locsumm.VFWat.Soil)**2
        ic_inp.loc[:,'Bew'] = ic_inp.WatDiffCoeff*locsumm.VFWat.Soil**(10/3) \
            /(locsumm.VFAir.Soil +locsumm.VFWat.Soil)**2
        #Fraction sediment volume occupied by water
        ic_inp.loc[:,'Bwx'] = ic_inp.WatDiffCoeff*locsumm.VFWat.Sediment**(4/3) 
        #Airside MTCs for veg and film (m/h)
        ic_inp.loc[:,'k_av'] = ic_inp.AirDiffCoeff / delta_blv
        ic_inp.loc[:,'k_af'] = ic_inp.AirDiffCoeff / delta_blf
        
        #ppLFER system parameters - initialize defaults if not there already
        if pp is None:
            pp = pd.DataFrame(index = ['l','s','a','b','v','c'])
            pp = make_ppLFER(pp)
        
        #Check if partition coefficients & dU values have been provided, or only solute descriptors
        #add based on ppLFER if not, then adjust partition coefficients for temperature of system
        #Aerosol-Air (Kqa), use octanol-air enthalpy
        if 'LogKqa' not in ic_inp.columns:
            ic_inp.loc[:,'LogKqa'] = ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.logKqa.l,pp.logKqa.s,pp.logKqa.a,pp.logKqa.b,pp.logKqa.v,pp.logKqa.c)
        if 'dUoa' not in ic_inp.columns: #!!!This might be broken - need to check units & sign!!!
            ic_inp.loc[:,'dUoa'] = ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.dUoa.l,pp.dUoa.s,pp.dUoa.a,pp.dUoa.b,pp.dUoa.v,pp.dUoa.c)
        ic_inp.loc[:,'Kqa'] = vant_conv(ic_inp.dUoa,params.Value.TempK,10**ic_inp.LogKqa,T1 = 288.15)
        ic_inp.loc[:,'LogKqa'] = np.log10(ic_inp.Kqa)
        #Organic carbon-water (KocW), use octanol-water enthalpy (dUow)
        if 'LogKocW' not in ic_inp.columns:
            ic_inp.loc[:,'LogKocW'] = ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.logKocW.l,pp.logKocW.s,pp.logKocW.a,pp.logKocW.b,pp.logKocW.v,pp.logKocW.c)
        if 'dUow' not in ic_inp.columns: #!!!This might be broken - need to check units & sign!!!
            ic_inp.loc[:,'dUow'] = 1000 * ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.dUow.l,pp.dUow.s,pp.dUow.a,pp.dUow.b,pp.dUow.v,pp.dUow.c)
        ic_inp.loc[:,'KocW'] = vant_conv(ic_inp.dUow,params.Value.TempK,10**ic_inp.LogKocW)
        ic_inp.loc[:,'LogKocW'] = np.log10(ic_inp.KocW)
        #Storage Lipid Water (KslW), use ppLFER for dUslW (kJ/mol) convert to J/mol/K
        if 'LogKslW' not in ic_inp.columns:
            ic_inp.loc[:,'LogKslW'] = ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.logKslW.l,pp.logKslW.s,\
            pp.logKslW.a,pp.logKslW.b,pp.logKslW.v,pp.logKslW.c)
        if 'dUslW' not in ic_inp.columns:
            ic_inp.loc[:,'dUslW'] = 1000 * ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.dUslW.l,pp.dUslW.s,pp.dUslW.a,pp.dUslW.b,pp.dUslW.v,pp.dUslW.c)
        ic_inp.loc[:,'KslW'] = vant_conv(ic_inp.dUslW,params.Value.TempK,10**ic_inp.LogKslW,T1 = 310.15)
        ic_inp.loc[:,'LogKslW'] = np.log10(ic_inp.KslW)
        #Air-Water (Kaw) use dUaw
        if 'LogKaw' not in ic_inp.columns:
            ic_inp.loc[:,'LogKaw'] = ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.logKaw.l,pp.logKaw.s,\
            pp.logKaw.a,pp.logKaw.b,pp.logKaw.v,pp.logKaw.c)
        if 'dUaw' not in ic_inp.columns: #!!!This might be broken - need to check units & sign!!!
            ic_inp.loc[:,'dUaw'] = 1000 * ppLFER(ic_inp.L,ic_inp.S,\
            ic_inp.A,ic_inp.B,ic_inp.V,pp.dUaw.l,pp.dUaw.s,pp.dUaw.a,pp.dUaw.b,pp.dUaw.v,pp.dUaw.c)
        ic_inp.loc[:,'Kaw'] = vant_conv(ic_inp.dUaw,params.Value.TempK,10**ic_inp.LogKaw)
        ic_inp.loc[:,'LogKaw'] = np.log10(ic_inp.Kaw)
        #Define storage lipid-air (KslA) and organic carbon-air (KocA) using the thermodynamic cycle
        #No need to adjust these for temperature as they are defined based on temeprature adjusted values
        ic_inp.loc[:,'KslA'] = ic_inp.KslW / ic_inp.Kaw
        ic_inp.loc[:,'KocA'] = ic_inp.KocW / ic_inp.Kaw
        #Calculate Henry's law constant (H, Pa m³/mol)
        ic_inp.loc[:,'H'] = ic_inp.Kaw * R * params.Value.TempK
        
        #Calculate temperature-corrected media reaction rates
        #Air (air_rrxn /hr), 3600 converts from /s
        ic_inp.loc[:,'air_rrxn'] = 3600 * \
        arr_conv(params.Value.EaAir,params.Value.TempK,ic_inp.AirOHRateConst * params.Value.OHConc)
        #Air Particles (airq_rrxn) 3600 converts from /s, use 10% of AirOHRateConst if not present
        if 'AirQOHRateConst' not in ic_inp.columns:
            ic_inp.loc[:,'airq_rrxn'] = 0.1 * ic_inp.air_rrxn
        else:
            ic_inp.loc[:,'airq_rrxn'] = 3600 * \
            arr_conv(params.Value.EaAir,params.Value.TempK,ic_inp.AirQOHRateConst * params.Value.OHConc)
        #Water (wat_rrxn) converted from half life (h)
        ic_inp.loc[:,'wat_rrxn'] = \
        arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/ic_inp.WatHL)
        #Soil (soil_rrxn) converted from half life (h)
        ic_inp.loc[:,'soil_rrxn'] = arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/ic_inp.SoilHL)
        #Sediment (sed_rrxn) converted from half life
        ic_inp.loc[:,'sed_rrxn'] = arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/ic_inp.SedHL)
        #Vegetation is based off of air half life, this can be overridden if chemsumm contains a VegHL column
        if 'VegHL' in ic_inp.columns:
            ic_inp.loc[:,'veg_rrxn'] = arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/ic_inp.VegHL)
        else:
            ic_inp.loc[:,'veg_rrxn'] = 0.1*ic_inp.air_rrxn
        #Same for film
        if 'FilmHL' in ic_inp.columns:
            ic_inp.loc[:,'film_rrxn'] = arr_conv(params.Value.Ea,params.Value.TempK,np.log(2)/ic_inp.FilmHL)
        else:
            ic_inp.loc[:,'film_rrxn'] = ic_inp.air_rrxn/0.75
        
        #Convert back to half lives (h), good for error checking
        ic_inp.loc[:,'AirHL'] = np.log(2)/(ic_inp.air_rrxn)
        ic_inp.loc[:,'AirQHL'] = np.log(2)/(ic_inp.airq_rrxn)
        ic_inp.loc[:,'WatHL'] = np.log(2)/(ic_inp.wat_rrxn)
        
        #Calculate Z-values (mol/m³/Pa)
        #Air lower and upper Zla and Zua, in case they are ever changed
        ic_inp.loc[:,'Zla'] = 1/(R*params.Value.TempK)
        ic_inp.loc[:,'Zua'] = 1/(R*params.Value.TempK)
        #Dissolved water Zw
        ic_inp.loc[:,'Zw'] = 1/(ic_inp.loc[:,'H'])
        #Soil Solids Zs, index is 3 in the locsumm file
        ic_inp.loc[:,'Zsoil'] = ic_inp.KocA*ic_inp.Zla*locsumm.Density.Soil*locsumm.FrnOC.Soil/1000
        #Sediment Solids
        ic_inp.loc[:,'Zsed'] = ic_inp.KocW*ic_inp.Zw*locsumm.Density.Sediment*locsumm.FrnOC.Sediment/1000
        #Plant Storage
        ic_inp.loc[:,'Zveg'] = ic_inp.KslA*ic_inp.Zla*locsumm.FrnOC.Vegetation
        #Dissolved Film
        ic_inp.loc[:,'Zfilm'] = ic_inp.KslA*ic_inp.Zla*locsumm.FrnOC.Film
        #Film Aerosol - Kqa is whole particle not just organic fraction
        ic_inp.loc[:,'Zqfilm'] = ic_inp.Kqa*ic_inp.Zla*locsumm.loc['Lower_Air','PartDensity']*1000
        #Lower and Upper air Aerosol particles - composed of water and particle, with the water fraction defined
        #by hygroscopic growth of the aerosol. Growth is defined as per the Berlin Spring aerosol from Arp et al. (2008)
        if params.Value.RH > 100: #maximum RH = 100%
            params.Value.RH = 100
        #Hardcoded hygroscopic growth factor (GF) not ideal but ¯\_(ツ)_/¯
        GF = np.interp(params.Value.RH/100,xp = [0.12,0.28,0.77,0.92],fp = [1.0,1.08,1.43,2.2],\
                       left = 1.0,right = params.Value.RH/100*5.13+2.2)
        #Volume fraction of water in aerosol
        VFQW_la = (GF - 1) * locsumm.Density.Water / ((GF - 1) * \
                  locsumm.Density.Water + locsumm.loc['Lower_Air','PartDensity'])
        VFQW_ua = (GF - 1) * locsumm.Density.Water / ((GF - 1) * \
                  locsumm.Density.Water + locsumm.loc['Upper_Air','PartDensity'])
        #Volume fraction of nucleus
        VFQp_la = 1 - VFQW_la
        VFQp_ua = 1 - VFQW_ua
        #Calculate aerosol Z values
        ic_inp.loc[:,'Zq_la'] = ic_inp.loc[:,'Zla']*ic_inp.loc[:,'Kqa']*locsumm.loc['Lower_Air','PartDensity']\
        *1000*VFQp_la+ic_inp.Zw*VFQW_la
        ic_inp.loc[:,'Zq_ua'] = ic_inp.loc[:,'Zua']*ic_inp.loc[:,'Kqa']*locsumm.loc['Upper_Air','PartDensity']\
        *1000*VFQp_ua+ic_inp.Zw*VFQW_ua
        #Suspended Sediment in the water compartment (Z_qw)
        ic_inp.loc[:,'Z_qw'] = ic_inp.Zw*ic_inp.KocW*locsumm.PartFrnOC.Water * locsumm.PartDensity.Water/1000
        #Bulk Z Value (Zb_j) 
        #Air - consists of Zq and Za
        ic_inp.loc[:,'Zb_la'] = ic_inp.loc[:,'Zla'] + ic_inp.loc[:,'Zq_la'] * locsumm.VFPart.Lower_Air
        ic_inp.loc[:,'Zb_ua'] = ic_inp.loc[:,'Zua'] + ic_inp.loc[:,'Zq_ua'] * locsumm.VFPart.Upper_Air
        #Water
        ic_inp.loc[:,'Zb_wat'] = ic_inp.loc[:,'Zw'] + ic_inp.loc[:,'Z_qw'] * locsumm.VFPart.Water
        #Soil
        ic_inp.loc[:,'Zb_soil'] = ic_inp.loc[:,'Zla'] * locsumm.VFAir.Soil+\
            ic_inp.loc[:,'Zw'] * locsumm.VFWat.Soil + \
            ic_inp.loc[:,'Zsoil']* (1-locsumm.VFAir.Soil -locsumm.VFWat.Soil)
        #Sediment
        ic_inp.loc[:,'Zb_sed'] = ic_inp.loc[:,'Zw'] * locsumm.VFWat.Sediment + \
            ic_inp.loc[:,'Zsed']* (1-locsumm.VFWat.Sediment)
        #Vegetation
        ic_inp.loc[:,'Zb_veg'] = ic_inp.loc[:,'Zla'] * locsumm.VFAir.Vegetation+\
            ic_inp.loc[:,'Zw'] * locsumm.VFWat.Vegetation + \
            ic_inp.loc[:,'Zveg']* (1-locsumm.VFAir.Vegetation -locsumm.VFWat.Vegetation)
        #Film
        ic_inp.loc[:,'Zb_film'] = ic_inp.loc[:,'Zqfilm'] * locsumm.VFPart.Film + \
        ic_inp.loc[:,'Zfilm'] * params.Value.VFOCFilm
        
        #Partition dependent transport parameters
        #veg & Film side MTCs (m/h)
        ic_inp.loc[:,'k_vv'] = 10 ** (0.704 * ic_inp.LogKocW - 11.2 - ic_inp.LogKaw)
        ic_inp.loc[:,'k_ff'] = 10 ** (0.704 * ic_inp.LogKocW - 11.2 - ic_inp.LogKaw)
        #lower air particle fraction (phi)
        ic_inp.loc[:,'phi'] = (ic_inp.Zq_la*locsumm.VFPart.Lower_Air)/ic_inp.Zb_la
        
        #Calculate advective (G) inflows(mol/m³ * m³/h = mol/h)
        if 'LairInflow' in ic_inp.columns:
            ic_inp.loc[:,'Gcb_1'] = locsumm.AdvFlow.Lower_Air * ic_inp.LairInflow
        else:
            ic_inp.loc[:,'Gcb_1'] = 0
        if 'UairInflow' in ic_inp.columns:
            ic_inp.loc[:,'Gcb_2'] = locsumm.AdvFlow.Upper_Air * ic_inp.UairInflow
        else:
            ic_inp.loc[:,'Gcb_2'] = 0
        if 'WatInflow' in ic_inp.columns:
            ic_inp.loc[:,'Gcb_3'] = locsumm.AdvFlow.Water * ic_inp.WatInflow
        else:
            ic_inp.loc[:,'Gcb_3'] = 0
        if 'SoilInflow' in ic_inp.columns: #add groundwater advective inflow
            ic_inp.loc[:,'Gcb_4'] = locsumm.AdvFlow.Soil * ic_inp.SoilInflow
        else:
            ic_inp.loc[:,'Gcb_4'] = 0
            
        #D Values 
        #Advection out from atmosphere and water
        ic_inp.loc[:,'D_adv_la'] = locsumm.AdvFlow.Lower_Air * ic_inp.Zb_la
        ic_inp.loc[:,'D_adv_ua'] = locsumm.AdvFlow.Upper_Air * ic_inp.Zb_la
        ic_inp.loc[:,'D_adv_w'] = locsumm.AdvFlow.Water * ic_inp.Zb_wat
        #Reaction- did not do a good job of indexing to make code smoother but this works
        ic_inp.loc[:,'D_rxn_la'] = locsumm.V.Lower_Air * ((1 - locsumm.VFPart.Lower_Air)\
                  * ic_inp.Zla * ic_inp.air_rrxn + locsumm.VFPart.Lower_Air * ic_inp.Zq_la * ic_inp.airq_rrxn)
        ic_inp.loc[:,'D_rxn_ua'] = locsumm.V.Upper_Air * ((1 - locsumm.VFPart.Upper_Air)\
                  * ic_inp.Zua * ic_inp.air_rrxn + locsumm.VFPart.Upper_Air * ic_inp.Zq_ua * ic_inp.airq_rrxn)
        ic_inp.loc[:,'D_rxn_wat'] = locsumm.V.Water * ic_inp.loc[:,'Zb_wat']*ic_inp.wat_rrxn
        ic_inp.loc[:,'D_rxn_soil'] = locsumm.V.Soil * ic_inp.loc[:,'Zb_soil']*ic_inp.soil_rrxn
        ic_inp.loc[:,'D_rxn_sed'] = locsumm.V.Sediment * ic_inp.loc[:,'Zb_sed']*ic_inp.sed_rrxn
        ic_inp.loc[:,'D_rxn_veg'] = locsumm.V.Vegetation * ic_inp.loc[:,'Zb_veg']*ic_inp.veg_rrxn
        #For film particles use a rxn rate 20x lower than the organic phase
        ic_inp.loc[:,'D_rxn_film'] = locsumm.V.Film * ((params.Value.VFOCFilm* ic_inp.Zfilm)\
                   * ic_inp.film_rrxn + locsumm.VFPart.Film * ic_inp.Zqfilm * ic_inp.film_rrxn/20)
        
        #Inter-compartmental Transport, matrix values are D_ij others are as noted
        #Lower and Upper Air
        ic_inp.loc[:,'D_12'] = params.Value.Ua * locsumm.Area.Lower_Air * ic_inp.Zb_la #Lower to Upper
        ic_inp.loc[:,'D_21'] = params.Value.Ua * locsumm.Area.Upper_Air * ic_inp.Zb_ua #Upper to lower
        ic_inp.loc[:,'D_st'] = params.Value.Ust * locsumm.Area.Upper_Air * ic_inp.Zb_ua #Upper to stratosphere
        #Lower Air to Water #Do we want to separate out the particle fraction here too? (1-ic_inp.phi) *
        ic_inp.loc[:,'D_vw'] =  1 / (1 / (params.Value.kma * locsumm.Area.Water \
                  * ic_inp.Zla) + 1 / (params.Value.kmw * locsumm.Area.Water * ic_inp.Zw)) #Dry dep of gas
        ic_inp.loc[:,'D_rw'] = locsumm.Area.Water * ic_inp.Zw * params.Value.RainRate * (1-ic_inp.phi) #Wet dep of gas
        ic_inp.loc[:,'D_qw'] = locsumm.Area.Water * params.Value.RainRate * params.Value.Q \
                  *locsumm.VFPart.Lower_Air * ic_inp.Zq_la  * ic_inp.phi #Wet dep of aerosol
        ic_inp.loc[:,'D_dw'] = locsumm.Area.Water * params.Value.Up * locsumm.VFPart.Lower_Air\
                  * ic_inp.Zq_la #dry dep of aerosol
        ic_inp.loc[:,'D_13'] = ic_inp.D_vw + ic_inp.D_rw + ic_inp.D_qw + ic_inp.D_dw #Air to water
        ic_inp.loc[:,'D_31'] = ic_inp.D_vw #Water to air
        #Lair and Soil
        ic_inp.loc[:,'D_vs'] = 1/(1/(params.Value.ksa*locsumm.Area.Soil \
                  *ic_inp.Zla)+Y4/(locsumm.Area.Soil*ic_inp.Bea*ic_inp.Zla+locsumm.Area.Soil*ic_inp.Bew*ic_inp.Zw)) #Dry dep of gas
        ic_inp.loc[:,'D_rs'] = locsumm.Area.Soil * ic_inp.Zw * params.Value.RainRate \
                  * (1-params.Value.Ifw) * (1-ic_inp.phi) #Wet dep of gas
        ic_inp.loc[:,'D_qs'] = locsumm.Area.Soil * ic_inp.Zq_la * params.Value.RainRate \
                  * locsumm.VFPart.Lower_Air * params.Value.Q * (1-params.Value.Ifw)  * ic_inp.phi #Wet dep of aerosol
        ic_inp.loc[:,'D_ds'] = locsumm.Area.Soil * ic_inp.Zq_la *  params.Value.Up \
                  * locsumm.VFPart.Lower_Air * (1-Ifd) #dry dep of aerosol
        ic_inp.loc[:,'D_14'] = ic_inp.D_vs + ic_inp.D_rs + ic_inp.D_qs + ic_inp.D_ds #Air to soil
        ic_inp.loc[:,'D_41'] = ic_inp.D_vs #soil to air        
        #Lair and Veg
        ic_inp.loc[:,'D_vv'] = 1/(1/(ic_inp.k_av*locsumm.Area.Vegetation\
                  *ic_inp.Zla)+1/(locsumm.Area.Vegetation*ic_inp.k_vv*ic_inp.Zveg)) #Dry dep of gas
        ic_inp.loc[:,'D_rv'] = locsumm.Area.Vegetation * ic_inp.Zw * params.Value.RainRate \
                  * params.Value.Ifw * (1-ic_inp.phi) #Wet dep of gas
        ic_inp.loc[:,'D_qv'] = locsumm.Area.Vegetation * ic_inp.Zq_la * params.Value.RainRate \
                  * params.Value.Q * params.Value.Ifw * locsumm.VFPart.Lower_Air * ic_inp.phi #Wet dep of aerosol
        ic_inp.loc[:,'D_dv'] = locsumm.Area.Vegetation * ic_inp.Zq_la * locsumm.VFPart.Lower_Air \
                  *params.Value.Up *Ifd  #dry dep of aerosol
        ic_inp.loc[:,'D_16'] = ic_inp.D_vv + ic_inp.D_rv + ic_inp.D_qv + ic_inp.D_dv #Air to veg
        ic_inp.loc[:,'D_61'] = ic_inp.D_vv #veg to air
        #Lair and film
        ic_inp.loc[:,'D_vf'] = 1/(1/(ic_inp.k_af*locsumm.Area.Film\
                  *ic_inp.Zla)+1/(locsumm.Area.Film*ic_inp.k_ff*ic_inp.Zfilm)) #Dry dep of gas
        ic_inp.loc[:,'D_rf'] = locsumm.Area.Film*ic_inp.Zw*params.Value.RainRate*(1-ic_inp.phi) #Wet dep of gas
        ic_inp.loc[:,'D_qf'] = locsumm.Area.Film * ic_inp.Zq_la * params.Value.RainRate \
                  * params.Value.Q* locsumm.VFPart.Lower_Air*ic_inp.phi #Wet dep of aerosol
        ic_inp.loc[:,'D_df'] = locsumm.Area.Film * ic_inp.Zq_la * locsumm.VFPart.Lower_Air\
                  * params.Value.Up #dry dep of aerosol
        ic_inp.loc[:,'D_17'] = ic_inp.D_vf + ic_inp.D_rf + ic_inp.D_qf + ic_inp.D_df #Air to film
        ic_inp.loc[:,'D_71'] = ic_inp.D_vf #film to air
        #Zrain based on D values & DRain (total), used just for assessing rain concentrations
        ic_inp.loc[:,'DRain'] = ic_inp.D_rw + ic_inp.D_qw + ic_inp.D_rs + ic_inp.D_qs \
                  + ic_inp.D_rv + ic_inp.D_qv + ic_inp.D_rf + ic_inp.D_qf
        ic_inp.loc[:,'ZRain'] = ic_inp.DRain / (locsumm.Area.Lower_Air * params.Value.RainRate) 
        #Water and Soil
        ic_inp.loc[:,'D_sw'] = locsumm.Area.Soil * ic_inp.Zsoil * params.Value.Usw  #Solid run off to water
        ic_inp.loc[:,'D_ww'] = locsumm.Area.Soil * ic_inp.Zw * params.Value.Uww  #Water run off to water
        ic_inp.loc[:,'D_43'] = ic_inp.D_sw + ic_inp.D_ww #Soil to water
        ic_inp.loc[:,'D_34'] = 0 #Water to soil
        ic_inp.loc[:,'D_sg'] = locsumm.Area.Soil * ic_inp.Zw * Usg #Soil to groundwater
        #Water and Sediment (x)
        ic_inp.loc[:,'D_tx'] = 1/(1/(params.Value.kxw*locsumm.Area.Sediment\
                  *ic_inp.Zw)+Y5/(locsumm.Area.Sediment*ic_inp.Bwx*ic_inp.Zw)) #Uptake by sediment
        ic_inp.loc[:,'D_dx'] = locsumm.Area.Sediment * ic_inp.Z_qw * params.Value.Udx  #Sediment deposition - should we have VFpart.Water?
        ic_inp.loc[:,'D_rx'] = locsumm.Area.Sediment * ic_inp.Zsed * params.Value.Urx  #Sediment resuspension
        ic_inp.loc[:,'D_35'] = ic_inp.D_tx + ic_inp.D_dx #Water to Sed
        ic_inp.loc[:,'D_53'] = ic_inp.D_tx + ic_inp.D_rx #Sed to Water
        ic_inp.loc[:,'D_bx'] = locsumm.Area.Sediment * ic_inp.Zsed * params.Value.Ubx #Sed burial
        #Water and Film
        ic_inp.loc[:,'D_73'] = locsumm.Area.Film * kfw * ic_inp.Zb_film #Soil to water
        ic_inp.loc[:,'D_37'] = 0 #Water to film
        #Soil and Veg
        ic_inp.loc[:,'D_cd'] = locsumm.Area.Vegetation * params.Value.RainRate \
        *(params.Value.Ifw - params.Value.Ilw)*params.Value.lamb * ic_inp.Zq_la  #Canopy drip
        ic_inp.loc[:,'D_we'] = locsumm.Area.Vegetation * params.Value.kwe * ic_inp.Zveg   #Wax erosion
        ic_inp.loc[:,'D_lf'] = locsumm.V.Vegetation * ic_inp.Zb_veg * params.Value.Rlf  #litterfall
        ic_inp.loc[:,'D_46'] = locsumm.V.Soil * params.Value.Rs * ic_inp.Zb_soil #Soil to veg
        ic_inp.loc[:,'D_64'] = ic_inp.D_cd + ic_inp.D_we + ic_inp.D_lf #Veg to soil
        #Total D-Values
        ic_inp.loc[:,'DT1'] = ic_inp.D_12 + ic_inp.D_13 + ic_inp.D_14 + ic_inp.D_16 + ic_inp.D_17 + ic_inp.D_adv_la + ic_inp.D_rxn_la #Lair
        ic_inp.loc[:,'DT2'] = ic_inp.D_21 + ic_inp.D_st + ic_inp.D_adv_ua + ic_inp.D_rxn_ua #Uair
        ic_inp.loc[:,'DT3'] = ic_inp.D_31 + ic_inp.D_35 + ic_inp.D_adv_w + ic_inp.D_rxn_wat #Water
        ic_inp.loc[:,'DT4'] = ic_inp.D_41 + ic_inp.D_43 + ic_inp.D_46 + + ic_inp.D_rxn_soil + ic_inp.D_sg #Soil
        ic_inp.loc[:,'DT5'] = ic_inp.D_53 + ic_inp.D_rxn_sed + ic_inp.D_bx #Sediment
        ic_inp.loc[:,'DT6'] = ic_inp.D_61 + ic_inp.D_64 + ic_inp.D_rxn_veg #Vegetation
        ic_inp.loc[:,'DT7'] = ic_inp.D_71 + ic_inp.D_73 + ic_inp.D_rxn_film #Film
        
        #Define total inputs (RHS of matrix) for each compartment
        #Note that if you run backwards calcs to calculate the inputs for a cell these are NOT overwritten, so these
        #should not be referenced except with that in mind.
        if 'LairEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_1'] = ic_inp.loc[:,'Gcb_1'] + ic_inp.loc[:,'LairEmiss']
        else:
            ic_inp.loc[:,'inp_1'] = ic_inp.loc[:,'Gcb_1']
        if 'UairEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_2'] = ic_inp.loc[:,'Gcb_2'] + ic_inp.loc[:,'UairEmiss']
        else:
            ic_inp.loc[:,'inp_2'] = ic_inp.loc[:,'Gcb_2']
        if 'WatEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_3'] = ic_inp.loc[:,'Gcb_3'] + ic_inp.loc[:,'WatEmiss']
        else: 
            ic_inp.loc[:,'inp_3'] = ic_inp.loc[:,'Gcb_3']
        if 'SoilEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_4']  = ic_inp.loc[:,'Gcb_4'] + ic_inp.loc[:,'SoilEmiss']
        else:
            ic_inp.loc[:,'inp_4']  = ic_inp.loc[:,'Gcb_4']
        if 'SedEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_5']  = ic_inp.loc[:,'SedEmiss']
        else: 
            ic_inp.loc[:,'inp_5']  = 0
        if 'VegEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_6']  = ic_inp.loc[:,'VegEmiss']
        else: 
            ic_inp.loc[:,'inp_6']  = 0
        if 'FilmEmiss' in ic_inp.columns:
            ic_inp.loc[:,'inp_7']  = ic_inp.loc[:,'FilmEmiss']
        else: 
            ic_inp.loc[:,'inp_7']  = 0
            
        #Define target fugacity in mol/m³/Pa. Note that the target should be a fugacity!!
        if 'LAirConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_1'] = ic_inp.loc[:,'LAirConc']/ic_inp.MolMass/ic_inp.Zb_la
        if 'UAirConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_2'] = ic_inp.loc[:,'UAirConc']/ic_inp.MolMass/ic_inp.Zb_ua
        if 'WatConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_3'] = ic_inp.loc[:,'WatConc']/ic_inp.MolMass/ic_inp.Zb_wat
        if 'SoilConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_4']  = ic_inp.loc[:,'SoilConc']/ic_inp.MolMass/ic_inp.Zb_soil
        if 'SedConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_5']  = ic_inp.loc[:,'SedConc']/ic_inp.MolMass/ic_inp.Zb_sed
        if 'VegConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_6']  = ic_inp.loc[:,'VegConc']/ic_inp.MolMass/ic_inp.Zb_veg
        if 'FilmConc' in ic_inp.columns:
            ic_inp.loc[:,'targ_7']  = ic_inp.loc[:,'FilmConc']/ic_inp.MolMass/ic_inp.Zb_film
        

        return ic_inp
    
    
    
    
    
    
    
    
    
    
    