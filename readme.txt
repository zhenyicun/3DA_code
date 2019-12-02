🌍️DESCRIPTION OF FOLDERS:
/data/OCCIPUT/        : the extracted 50-member OCCIPUT simulation at GoM
/data/obs/            : the extracted real tracks position and time of the altimeters in 2004. It also saves the truth and the SSH along-tracks after you run get_obs.py
/data/catalog/        : is used to save the catalog for AnDA
/data/results/AnDA/   : is used to save the results of the last run of AnDA 
/data/results/OI/     : is used to save the results of the last run of OI
/data/results/OI_COA/ : is used to save the results of the last run of OI_COA (following the OI described in <LeTraon et.al. 1998> )
/figs/                : saves all the diagnostic figs. 



🌍️DESCRIPTION OF EXISTING RESULTS:
/data/results/AnDA_k1000_ROI10  : saves the result of AnDA where k=1000, and spatial localization radius = 10 degress 
/data/results/OI_6              : saves the result of OI where temporal correlation scale = 6 days 
/data/results/OI_10             : saves the result of OI where temporal correlation scale = 10 days 
/data/results/OI_15             : saves the result of OI where temporal correlation scale = 15 days 





🌍️DESCRIPTION OF PARAMETERS:

🌐️AnDA:
max_mode                  : number of EOFs used in the state variables
do_KS                     : "yes"---do Kalman smoother, "no"---only Kalman filter
Ne                        : ensemble size for EnKF and EnKS
R                         : observation error variance for EnKF
do_localization           : "yes"---do covariance spatial localization for EnKF and EnKS, "no"---no localization    
rloc                      : localization radius for covariance spatial localization used in EnKF and EnKS
AF_k                      : number of nearest neighbors to be search for in the Analog forecast algorithm 

🌐️OI:
r_spat           
s_spat
r_temp
s_temp  : temporal correlation scale
R       : observation error variance used in the algorithm




🌍️RUN THE CODE:

1,   run get_obs.py:  generate the truth using OCCIPUT member#1 at year 20, and the ssh at the given tracks at the given time

2,   run get_catalog.py: generate the catalog using OCCIPUT members#2-50, from year 1 to 19 

3.1, choose the parameters of AnDA, OI, and OI_COA by editting main_AnDA.py, main_OI.py, and main_OI_COA.py
3.2, run main_AnDA.py, or main_OI.py, or main_OI_COA.py. The new results will be saved in /data/results/AnDA/, /data/results/OI/, and /data/results/OI_COA/

4,   run generate_figs.py to generate all the figs and check the RMSE. You will see all the figs in this paper except the two spectrum figs produced by Sammy. 
