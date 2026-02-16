import numpy as np
import scipy.io as spio

intrstate_comp = 12
primr_comp = 47
sec_comp = 26

ncomp_deck = 11 #number of component bridges
ncomp_pav = intrstate_comp+primr_comp+sec_comp #number of component pavements
tot_comp = ncomp_deck + ncomp_pav
nstcomp_deck= 7 # number of states per bridge
nstcomp_CCI = 6 # number of states per pavement CCI
nstcomp_IRI = 5 # number of states per pavement IRI
nacomp = 10 # number of actions per component = 3*3 +1 when replacement
# no-inspection-{1,2,3 mainetenance}, low fed_insp-{1,2,3 mainetenance}, high_fed_insp-{1,2,3 mainetenance}, aand replacement 
nobs_CCI = 6 # number of observations for CCI
nobs_IRI = 5 # number of observations IRI
nobs_deck = 7 # number of observations



# Network geometry
file = ('link_lengths.mat')
mat1 = spio.loadmat(file, squeeze_me=True)
lenth_comp = mat1['length1']       #in meters
lenth_comp = lenth_comp[0:ncomp_pav]
n_lane_pav = np.zeros((lenth_comp.shape))
for i in range(ncomp_pav):
    if i<intrstate_comp:
        n_lane_pav[i] = 8
    elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
        n_lane_pav[i] = 4
    else:
        n_lane_pav[i] = 2
        
file = ('Nodes.mat')
mat = spio.loadmat(file, squeeze_me=True)
Nodes = mat['Nodes'][:,:]          

# Connectivity and adjacenecy matrices
file = ('Connectivity.mat')
mat = spio.loadmat(file, squeeze_me=True)
Connectivity = mat['Connectivity'][:,:]

file = ('Adjacency.mat')
mat = spio.loadmat(file, squeeze_me=True)
Adjm = mat['Adj'][:,:]

file = ('adjacency_2.mat')
mat = spio.loadmat(file, squeeze_me=True)
Adj2 = mat['Adj2'][:,:]




### pavement nodes ###
Type_1s = np.array([7, 7, 7,8, 9,12,19,20,49,49,62,62,62,62,64]);
Type_1e = np.array([8,19,20,9,12,13,68,63,68,67,63,63,64,67,71]);


Type_3s = np.array([3, 9, 9,21,22,23,24,24,24,26,28,34,34,35,40,45,46,50,53,53,53,54,56,57,57,60 ]);
Type_3e = np.array([4,14,15,70,23,24,25,26,28,27,29,35,40,36,41,70,48,53,54,56,61,55,57,58,62,61]);

Area_pav = np.zeros(ncomp_pav)
for i in range(ncomp_pav):
    Area_pav[i] = (1.61*1000)*lenth_comp[i]*n_lane_pav[i]*3.7    # in m sq.

Tot_intrst_area = sum(Area_pav[0:intrstate_comp])
Tot_primr_area = sum(Area_pav[intrstate_comp:(intrstate_comp+primr_comp)])
Tot_scndr_area = sum(Area_pav[(intrstate_comp+primr_comp):ncomp_pav])
Tot_prm_intr_area = Tot_primr_area + Tot_intrst_area
Total_Area_pav = Tot_primr_area + Tot_intrst_area+Tot_scndr_area
##################### bridge nodes ###################
brdg_nodes = np.array([[18, 19, 20, 5, 21, 49,49,50, 66,31, 53],[21, 68, 63, 6, 69, 66,50,62, 69,45, 54]])
btype_1s = np.array([18, 19, 20]);
btype_1e = np.array([21, 68, 63]);
btype_2s = np.array([5, 21, 49,49,50, 66]);
btype_2e = np.array([6, 69, 50,66,62, 69]);
btype_3s = np.array([31, 53]);
btype_3e = np.array([45, 54]);
brdg_on_links = np.array([22,  24, 25, 4, 27, 60, 59, 65, 87, 41, 67])

##### bridge properties ######
brdg_len = np.array([7.07, 7.4, 5.6, 1.140,  1.13, 1.5,  0.647,    0.64,   1.01, 0.32,    0.90])/(1.61);
n_lane   = np.array([4,    4,   4,   4,      2,    4,    8,        4,      4,   4,       4 ]);
lane_wdth= np.array([5,    4,   3.7, 3.7,    3.7,  3.7,  3.7,      3.7,    3.7, 3.7,     3.7]);

Area = brdg_len*n_lane*lane_wdth*1.61*1000   # in m sq.
Tot_deck_area = sum(Area)

# In[514]:

# Load do nothing Transitions for pavements CCI and IRI and bridges
file_pav06 = ('Smoothed_TP_MSI_06.mat')
mat_pav06 = spio.loadmat(file_pav06, squeeze_me=True)
file_pav08 = ('Smoothed_TP_MSI_08.mat')
mat_pav08 = spio.loadmat(file_pav08, squeeze_me=True)
file_pav20 = ('Smoothed_TP_MSI_20.mat')
mat_pav20 = spio.loadmat(file_pav08, squeeze_me=True)
file_deck = ('Deck_TP_for_DRL.mat')
mat_deck = spio.loadmat(file_deck, squeeze_me=True)

# In[515]:
# Do nothing transitions deck, CCI, and IRI
pcomp_deck = np.zeros((nstcomp_deck,nstcomp_deck,20,ncomp_deck))
pcomp_CCI = np.zeros((nstcomp_CCI,nstcomp_CCI,20,ncomp_pav))

# first fifteen components are type I, next 47 comp are type II and the rest are type III
for i in range(ncomp_pav):
    if i<intrstate_comp:
        pcomp_CCI[:,:,:,i] =  mat_pav06['prob2'][0:nstcomp_CCI,0:nstcomp_CCI]
    elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
        pcomp_CCI[:,:,:,i] =  mat_pav08['prob2'][0:nstcomp_CCI,0:nstcomp_CCI]
    elif i>=(intrstate_comp+primr_comp) and i<ncomp_pav:
        pcomp_CCI[:,:,:,i] =  mat_pav20['prob2'][0:nstcomp_CCI,0:nstcomp_CCI]
  
        
# first 3 bridge components are type I, next 6 comp are type II and rest are type III       
for i in range(ncomp_deck):
    pcomp_deck[:,:,:,i] =  mat_deck['Tp_1'][0:nstcomp_deck,0:nstcomp_deck]
    
# Do Nothing transition IRI
pcomp_IRI = np.array([[0.839,	0.121,	0.039,	  0.,    0.], 	
                       [ 0.,    0.787,	0.142,	0.07,    0.],	
                       [ 0.,     0.,    0.708,	0.192,	0.099],
                       [0.,     0.,       0.,  0.578,	0.421],
                       [0.,    0.,       0.,    0.,     1]])


# In[516]:

# Action probabilities for deck, CCI, and IRI 
pobs_minor_CCI= np.array([[0.97,	0.03,	0,	0,	0,	0],
                           [0.87,	0.1,	0.03,	0,	0,	0],
                           [0.4,	0.47,	0.1,	0.03,	0,	0],
                           [0,	0.4,	0.47,	0.1,	0.03,	0],
                           [0,	0,	0.4,	0.47,	0.1,	0.03],
                           [0,	0,	0,	0.4,	0.47,	0.13]])
                              
pobs_major_CCI = np.array([[1,	      0,	0,   0,	0,	0],
                           [0.96,	0.04,	0,	 0,	0,	0],
                           [0.8,	0.2,	0,	 0,	0,	0],
                           [0.65,	0.25,	0.1,	0,	0,	0],
                           [0.5,	0.3,	0.2,	0,	0,	0],
                           [0.4,	0.3,	0.3,	0,	0,	0]])

# IRI Action transition probabilities

# minor repair
pobs_minor_IRI = np.array([[0.97,	0.03,	0,	0,	0],
                           [0.85,	0.12,	0.03,	0,	0],
                           [0.45,	0.4,	0.12,	0.03,	0],
                           [0,	0.45,	0.4,	0.12,	0.03],
                           [0,	0,	0.45,	0.4,	0.15]])  

# Major repair
pobs_major_IRI = np.array([[1,	0,	0,	0,	0],
                          [0.95,	0.05,	0,	0,	0],
                          [0.80,	0.20,	0,	0,	0],
                          [0.7,	0.25,	0.05,	0,	0],
                          [0.45,	0.35,	0.2,	0,	0]])


# Action probabilities for Deck

pobs_minor_deck = np.array([       [0.97, 0.03, 0, 0, 0, 0, 0],
                                   [0.85, 0.12, 0.03, 0, 0, 0,0],
                                   [0.4, 0.45, 0.12, 0.03, 0, 0,0],
                                   [0, 0.4, 0.45, 0.12, 0.03, 0,0],
                                   [0, 0, 0.4, 0.45, 0.12, 0.03,0],
                                   [0, 0, 0, 0.40, 0.45,   0.15,0],
                                   [0, 0, 0, 0, 0, 0, 1]])
                              
pobs_major_deck = np.array([        [1, 0, 0, 0, 0, 0, 0],
                                    [0.95,	0.05,	0,	0,	0,	0, 0],
                                    [0.8,	0.2,	0, 0, 0, 0, 0],
                                    [0.6,	0.3,	0.1, 0, 0, 0, 0],
                                    [0.4,	0.4,	0.2, 0, 0, 0, 0],
                                    [0.3,	0.4,	0.3, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])

pobs_replac = np.array([[1.],[1.],[1.],[1.],[1.]])


# In[517]:
    # Observation probabilities
pobs_insp_low_fed_CCI = np.array([[0.688, 0.258, 0.054, 0.000, 0.000, 0.000],
                                  [0.277, 0.422, 0.297, 0.004, 0, 0],
                                  [0.024, 0.14, 0.648, 0.166, 0.022, 0.001],
                                  [0, 0.003, 0.266, 0.455, 0.249, 0.027],
                                  [0, 0, 0.031, 0.223, 0.486, 0.26],
                                  [0, 0, 0, 0.006, 0.061, 0.936]])
                             
pobs_insp_high_fed_CCI = np.array([[0.803, 0.195, 0.002, 0, 0, 0],
                                  [0.152,	0.664,	0.183,	0,	0,	0],
                                  [0.001,	0.078,	0.822,	0.1,	0,	0],
                                  [0,	0,	0.149,	0.693,	0.158,	0],
                                  [0,	0,	0.001,	0.137,	0.718,	0.144],
                                  [0,	0,	0,	0,	0.045,	0.97]])
    
pobs_insp_low_fed_IRI = np.array([[0.80,	0.20,	0.00,   0.00,	0.00],
                                  [0.20,	0.60,	0.20,	0.00,	0.00],
                                  [0.00,	0.20,	0.60,	0.20,	0.00],
                                  [0.00,	0.00,	0.20,	0.60,	0.20],
                                  [0.00,	0.00,	0.00,	0.20,	0.80]])
                             
pobs_insp_high_fed_IRI = np.array([[0.90,	0.10,	0.00,	0.00,	0.00],
                                   [0.05,	0.90,	0.05,	0.00,	0.00],
                                   [0.00,	0.05,	0.90,	0.05,	0.00],
                                   [0.00,	0.00,	0.05,	0.90,	0.05],
                                   [0.00,	0.00,	0.00,	0.10,	0.90]])
    
pobs_insp_low_fed_deck =np.array([[0.8,	0.15,	0.05,	0,	0,	0,	0],
                                  [0.15,	0.65,	0.15,	0.05,	0,	0,	0],
                                  [0.05,	0.15,	0.6,	0.15,	0.05,	0,	0],
                                  [0,	0.05,	0.15,	0.6,	0.15,	0.05,	0],
                                  [0,	0,	0.05,	0.15,	0.65,	0.15,	0],
                                  [0,	0,	0,	0.05,	0.15,	0.8,	0],
                                  [0,	0,	0,	0,	0,	0,	1]])

                             
pobs_insp_high_fed_deck =np.array([[0.9,	0.1,	0,	0,	0,	0,	0],
                                   [0.1,	0.8,	0.1,	0,	0,	0,	0],
                                   [0,	0.1,	0.8,	0.1,	0,	0,	0],
                                   [0,	0,	0.1,	0.8,	0.1,	0,	0],
                                   [0,	0,	0,	0.1,	0.8,	0.1,	0],
                                   [0,	0,	0,	0,	0.1,	0.9,	0],
                                   [0,	0,	0,	0,	0,	0,	1]])


# In[518]:
# observation prob on current damage state and action
pobs_IRI = np.zeros((ncomp_pav,nacomp,nstcomp_IRI,nobs_IRI))
pobs_CCI = np.zeros((ncomp_pav,nacomp,nstcomp_CCI,nobs_CCI))
pobs_deck = np.zeros((ncomp_deck,nacomp,nstcomp_deck,nobs_deck))

# for CCI
for i in range (ncomp_pav):
    for j in range(10):
        if j in [0,1,2]:
            pobs_CCI[i,j,:,:] = 1/nstcomp_CCI
        elif j in [3,4,5]:
            pobs_CCI[i,j,:,:] = pobs_insp_low_fed_CCI[:,:]
        elif j in [6,7,8]:
            pobs_CCI[i,j,:,:] = pobs_insp_high_fed_CCI[:,:]
        elif j ==9:
            pobs_CCI[i,j,:,:] = 0
            pobs_CCI[i,j,:,0] = 1

# for IRI
for i in range (ncomp_pav):
    for j in range(10):
        if j in [0,1,2]:
            pobs_IRI[i,j,:,:] = 1/nstcomp_IRI
        elif j in [3,4,5]:
            pobs_IRI[i,j,:,:] = pobs_insp_low_fed_IRI[:,:]
        elif j in [6,7,8]:
            pobs_IRI[i,j,:,:] = pobs_insp_high_fed_IRI[:,:]
        elif j ==9:
            pobs_IRI[i,j,:,:] = 0
            pobs_IRI[i,j,:,0] = 1


# for decks
for i in range (ncomp_deck):
    for j in range(10):
        if j in [0,1,2]:
            pobs_deck[i,j,0:6,0:6] = 1/(nstcomp_deck-1)
            pobs_deck[i,j,6,6] = 1
        elif j in [3,4,5]:
            pobs_deck[i,j,:,:] = pobs_insp_low_fed_deck[:,:]
        elif j in [6,7,8]:
            pobs_deck[i,j,:,:] = pobs_insp_high_fed_deck[:,:]
        elif j ==9:
            pobs_deck[i,j,:,:] = 0
            pobs_deck[i,j,:,0] = 1


# In[519]:

# Action and observation costs
A_pav = 3.7*np.array([[0, 20, 75, 350],[0,16,68,330],[0,10,52,250]])  # pav Action costs per meter of lane in USD 
A_deck = np.array([0, 400, 1200, 2650.00]) # deck action costs per meter sq. in USD
I_pav = 3.7*np.array([0, 0.10, 0.20]) #pav inspection cost per meter lane in USD 
I_deck = np.array([0, 0.48, 1.20])  # deck inspection costs per meter sq. in USD
#cost_comp_state = np.zeros((ncomp,nstcomp))
cost_comp_action_pav = np.zeros((ncomp_pav,nacomp))
cost_comp_action_deck = np.zeros((ncomp_deck,nacomp))
cost_comp_action = np.zeros((tot_comp,nacomp))
cost_comp_obsr_pav = np.zeros((ncomp_pav,nacomp))
cost_comp_obsr_deck = np.zeros((ncomp_deck,nacomp))
cost_comp_obsr = np.zeros((tot_comp,nacomp))
############ cost for pavements ##########
for i in range(ncomp_pav):
    for j in range(10):
        if j in [0,1,2]:
            cost_comp_obsr_pav[i,j] = 0
            if i<intrstate_comp:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[0][j%3])*8/1000000  #8 bc of 8 lanes assuming a 12 ft (3.658 m) lane in per million USD
                
            elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[1][j%3])*4/1000000  #4 bc of 4 lanes 
                
            else:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[2][j%3])*2/1000000  #2 bc of 2 lanes 
                
        elif j in [3,4,5]:
            if i<intrstate_comp:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[0][j%3])*8/1000000  #8 bc of 8 lanes assuming a 12 ft (3.658 m) lane in per million USD
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[1])*8/1000000
            elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[1][j%3])*4/1000000  #4 bc of 4 lanes 
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[1])*4/1000000
            else:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[2][j%3])*2/1000000  #2 bc of 2 lanes 
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[1])*2/1000000
        elif j in [6,7,8]:
            if i<intrstate_comp:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[0][j%3])*8/1000000  #8 bc of 8 lanes assuming a 12 ft (3.658 m) lane in per million USD
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[2])*8/1000000
            elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[1][j%3])*4/1000000  #4 bc of 4 lanes 
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[2])*4/1000000
            else:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[2][j%3])*2/1000000  #2 bc of 2 lanes  
                cost_comp_obsr_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(I_pav[2])*2/1000000
        elif j == 9:
            cost_comp_obsr_pav[i,j] = 0
            if i<intrstate_comp:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[0][3])*8/1000000  #6 bc of 8 lanes assuming a 12 ft (3.658 m) lane in per million USD
                
            elif i>=intrstate_comp and i<(intrstate_comp+primr_comp):
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[1][3])*4/1000000  #4 bc of 4 lanes 
            else:
                cost_comp_action_pav[i,j] = -(1.61*1000)*lenth_comp[i]*(A_pav[2][3])*2/1000000  #2 bc of 2 lanes 
          
############ cost actions for bridges ##########
for i in range(ncomp_deck):
    for j in range(10):
        if j in [0,1,2]:
            cost_comp_obsr_deck[i,j] = 0
            cost_comp_action_deck[i,j] = -(Area[i])*(A_deck[j%3])/1000000  #cost per million USD
        elif j in [3,4,5]:
            cost_comp_obsr_deck[i,j] = -(Area[i])*(I_deck[1])/1000000
            cost_comp_action_deck[i,j] = -(Area[i])*(A_deck[j%3])/1000000  #cost per million USD
        elif j in [6,7,8]:
            cost_comp_obsr_deck[i,j] = -(Area[i])*(I_deck[2])/1000000
            cost_comp_action_deck[i,j] = -(Area[i])*(A_deck[j%3])/1000000  #cost per million USD
        elif j == 9:
            cost_comp_obsr_deck[i,j] = 0
            cost_comp_action_deck[i,j] = -(Area[i])*(A_deck[3])/1000000  #cost per million USD


cost_comp_action[0:ncomp_pav][:] = cost_comp_action_pav
cost_comp_action[ncomp_pav:][:] = cost_comp_action_deck

cost_comp_obsr[0:ncomp_pav][:] = cost_comp_obsr_pav
cost_comp_obsr[ncomp_pav:][:] = cost_comp_obsr_deck

# In[520]
# traffic distribution based on VDOT 2019 VMT
# In Virginia 
# total interstate road = 1118 miles
# total primary road = 8111 miles
# total secondar road = 8111 miles
#nontruck
# Secondary = 1313 VMT per secondary miles perday 2
# Primary =  11323 VMT per primary miles perday   15
# Interstate = 56751 VMT per interstate miles perday 40

# truck miles for 
# Secondary = 1339359/48111= 27.9 VMT per secondary miles perday 1
# Primary =  5,115,444/8111 = 630.7 VMT per primary miles perday   23
# Interstate = 9137898/1118 = 8173 VMT per primary miles perday 293




# truck_miles = 1.13*10**6*365          #annual truck miles 
# nontruck_miles = (40-1.13)*10**6*365  # Annual non truck miles from hampton roads report
# inter_state_factor = [0.70,0.160,0.14];
# primary_hwy_factor = [0.77,0.19,0.04];
# sec_hwy_factor = [0.8568,0.1243,0.0189]


# bridge_truck_miles = np.sum(brdg_len*n_lane)/(np.sum(brdg_len*n_lane)+np.sum(n_lane_pav*lenth_comp))*truck_miles
# bridge_nontruck_miles = np.sum(brdg_len*n_lane)/(np.sum(brdg_len*n_lane)+np.sum(n_lane_pav*lenth_comp))*nontruck_miles
# pav_truck_miles = truck_miles-bridge_truck_miles
# pav_nontruck_miles = nontruck_miles-bridge_nontruck_miles

# bridge_truck_miles = 0.011563374*10**6*365
# bridge_nontruck_miles = 0.160406843*10**6*365
# pav_truck_intrstate = 0.333*10**6*365
# pav_truck_prim = 0.828*10**6*365
# pav_truck_sec = 0.009*10**6*365
# pav_nontruck_intrstate = 5.7*10**6*365
# pav_nontruck_prim = 6.727*10**6*365
# pav_nontruck_sec = 0.4769*10**6*365



# Distributing Vehicle miles
# nontruck_traff_link = np.zeros(tot_comp)
# truck_traff = np.zeros(tot_comp)
# car_traff = np.zeros(tot_comp)
# lite_truck_traff = np.zeros(tot_comp)

# for i in range(tot_comp): 
     
    # if i<intrstate_comp:
        # nontruck_traff_link[i] = pav_nontruck_intrstate*lenth_comp[i]/np.sum(lenth_comp[0:intrstate_comp])
        # car_traff[i] = (inter_state_factor[0]/np.sum(inter_state_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (inter_state_factor[1]/np.sum(inter_state_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (pav_truck_intrstate)*lenth_comp[i]/np.sum(lenth_comp[0:intrstate_comp])
    # elif i>=intrstate_comp and i<intrstate_comp+primr_comp:
        # nontruck_traff_link[i] = (pav_nontruck_prim)*lenth_comp[i]/np.sum(lenth_comp[intrstate_comp:intrstate_comp+primr_comp])
        # car_traff[i] = (primary_hwy_factor[0]/np.sum(primary_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (primary_hwy_factor[1]/np.sum(primary_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (pav_truck_prim)*lenth_comp[i]/np.sum(lenth_comp[intrstate_comp:intrstate_comp+primr_comp])
    # elif i>=intrstate_comp+primr_comp and i<ncomp_pav:
        # nontruck_traff_link[i] = (pav_nontruck_sec)*lenth_comp[i]/np.sum(lenth_comp[intrstate_comp+primr_comp:ncomp_pav])
        # car_traff[i] = (sec_hwy_factor[0]/np.sum(sec_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (sec_hwy_factor[1]/np.sum(sec_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (pav_truck_sec)*lenth_comp[i]/np.sum(lenth_comp[intrstate_comp+primr_comp:ncomp_pav])
    # elif i>=ncomp_pav and i<ncomp_pav+3:
        # nontruck_traff_link[i] = (bridge_nontruck_miles*(25/33.64))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[0:3]*n_lane[0:3])
        # car_traff[i] = (inter_state_factor[0]/np.sum(inter_state_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (inter_state_factor[1]/np.sum(inter_state_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (bridge_truck_miles*(77/(128)))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[0:3]*n_lane[0:3])
    # elif i>=ncomp_pav+3 and i<ncomp_pav+9:
        # nontruck_traff_link[i] = (bridge_nontruck_miles*(7.64/33.64))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[3:9]*n_lane[3:9])
        # car_traff[i] = (primary_hwy_factor[0]/np.sum(primary_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (primary_hwy_factor[1]/np.sum(primary_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (bridge_truck_miles*(50/(128)))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[3:9]*n_lane[3:9])
    # else:
        # nontruck_traff_link[i] = (bridge_nontruck_miles*(1/33.64))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[9:11]*n_lane[9:11])
        # car_traff[i] = (sec_hwy_factor[0]/np.sum(sec_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # lite_truck_traff[i] = (sec_hwy_factor[1]/np.sum(sec_hwy_factor[0:1]))*nontruck_traff_link[i] 
        # truck_traff[i] = (bridge_truck_miles*(1/(128)))*brdg_len[i-ncomp_pav]*n_lane[i-ncomp_pav]/np.sum(brdg_len[9:11]*n_lane[9:11])


# In[522]:
# Model for this is used from Dr. Stoffels paper
# User delay cost
# total_traff = nontruck_traff_link+truck_traff
# truck_percent  = truck_traff/total_traff
# X = np.zeros(len(total_traff)) 
# for i in range(tot_comp):
    # if i<intrstate_comp:
        # X[i] = total_traff[i]/8760/lenth_comp[i]/8
    # elif i>=intrstate_comp and i<intrstate_comp+primr_comp:
        # X[i] = total_traff[i]/8760/lenth_comp[i]/4
    # elif i>=intrstate_comp+primr_comp and i<ncomp_pav:
        # X[i] = total_traff[i]/8760/lenth_comp[i]/2
    # else:
        # X[i] = total_traff[i]/8760/brdg_len[i-ncomp_pav]/n_lane[i-ncomp_pav]

# X1 = (X+48600)/55 
# TTD_L = np.exp(1.67+0.0129*truck_percent+0.00000252*(X-X1)+0.00395*(X-X1))   # number of delayed hours 
# TTD_R = np.exp(1.67+0.0129*truck_percent+0.00000252*(X-X1)+0.00395*(X-X1)+0.602)
# for i in range(tot_comp):
    # if i<ncomp_pav:
        # TTD_L[i] = n_lane_pav[i]*TTD_L[i]*lenth_comp[i] 
        # TTD_R[i] = n_lane_pav[i]*TTD_R[i]*lenth_comp[i]
    # else:
        # TTD_L[i] = n_lane[i-ncomp_pav]*TTD_L[i]*brdg_len[i-ncomp_pav]
        # TTD_R[i] = n_lane[i-ncomp_pav]*TTD_R[i]*brdg_len[i-ncomp_pav] 


######
# Maintenance Action durations for each link 0, 1, 2, 3, is do nothing, minor, major and recon 
# User Delay Cost
# 21.89 $/hr passenger car, 29.65 $/hr for trucks 

# duration = np.array([0,3.5,6.5,32])
# extra  = np.array([0,1,2,10])
# brdg_duration  = np.array([[0,25,70,300],[0,12,30,150],[0,6,15,70]])
# Ta = np.zeros((tot_comp,nacomp))
# Dlay_cost = np.zeros((tot_comp,nacomp))
# for i in range(tot_comp):
    # for j in range(10):
        # if j==9:
            # if i<ncomp_pav:
                # Ta[i][j] = n_lane_pav[i]*lenth_comp[i]*duration[3] + lenth_comp[i]*extra[3]
            # else:
                # if i-ncomp_pav<3:
                    # Ta[i][j] = 300
                # elif i-ncomp_pav>=3 and i-ncomp_pav <9:
                    # Ta[i][j] = 150
                # else:
                    # Ta[i][j] = 70
  
            # if Ta[i][j]>730:
                # Ta[i][j]=730
            # Dlay_cost[i][j] = -0.5*(TTD_L[i]+TTD_R[i])*Ta[i][j]*24*((1-truck_percent[i])*21.89+truck_percent[i]*29.65)/1000000
        # else:
            # if i<ncomp_pav:
                # Ta[i][j] = n_lane_pav[i]*lenth_comp[i]*duration[j%3] + lenth_comp[i]*extra[j%3]
            # else:
                # if i-ncomp_pav<3:
                    # Ta[i][j] = brdg_duration[0][j%3]
                # elif i-ncomp_pav>=3 and i-ncomp_pav <9:
                    # Ta[i][j] = brdg_duration[1][j%3]
                # else:
                    # Ta[i][j] = brdg_duration[2][j%3]
            
            # if Ta[i][j]>365:
                # Ta[i][j]=365
            # Dlay_cost[i][j] = -0.5*(TTD_L[i]+TTD_R[i])*Ta[i][j]*24*((1-truck_percent[i])*21.89+truck_percent[i]*29.65)/1000000


# finding indices 
# Tag = (Ta[:,-1]>365)*1==1
# Tag = list(np.where(Tag)[0]) 















Dlay_cost = np.load('Dlay_Cost.npy')
intdx = list([])
intdx +=list(range(0,ncomp_pav))
intdx += list(range(ncomp_pav,ncomp_pav+11))
Dlay_cost = Dlay_cost[intdx,:]
Ta = np.load('Total_action_duration.npy')
Ta = Ta[intdx,:]

# finding indices 
Tag = (Ta[:,-1]>365)*1==1
Tag = list(np.where(Tag)[0]) 

