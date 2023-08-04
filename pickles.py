

## welcome to PICKLES --> Probabilistic Identification of Clusters using Kurtosis for Localizing Extraterrestrial Signals.


import subprocess
import sys
import os
# from boundary_checker import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statistics as stats
import pandas as pd
from scipy import signal
import sys
from datetime import datetime
import h5py
from scipy.stats import pearsonr 
import scipy.stats  
from tqdm import tqdm
import traceback
import hdf5plugin
import argparse
import pickle
import math

from cappuccino.cappuccino.__init__ import *


def main(batch_number):
    """This is the main function.

    Args:
        candidates_df_name (_type_): _description_
    """

    # check if candidates database is set up, if not then initialize it. This is where the candidates will be stored
    batch_number = int(batch_number)
    block_size = 4096
    significance_level = 10
    main_dir = '/mnt_blpc1/datax/scratch/calebp/k_scores/'

    # we will store the information for pickle in pickle_jar.csv
    df_name = f'all_cadences_mason_jar_batch_{batch_number}.csv'

    db_exists = os.path.exists(main_dir+df_name)
    if db_exists == False:
        print(main_dir+df_name)
        print("Creating candidates database as ",df_name)
        feature_table = pd.DataFrame(columns=["All Files","Index","Freq","range1","range2","range3","range","obs1 maxes","obs3 maxes","obs5 maxes","k1","k2","k3","k4","k5","k6","k_score","min_k","med_k","max_k","drift"])
        feature_table.to_csv(main_dir+df_name,index=False)
    else:
        print("feature table database already exists:",main_dir+df_name)

    # define batches
    # batches = [['AND_I', 'AND_X', 'AND_XI', 'AND_XIV', 'AND_XVI', 'AND_XXIII', 'AND_XXIV', 'BOL520', 'CVNI', 'DDO210','DRACO', 'DW1','HERCULES', 'HIZSS003', 'IC0010', 'IC0342', 'IC1613', 'LEOA', 'LEOII', 'LEOT'],['LGS3', 'MAFFEI1', 'MAFFEI2', 'MESSIER031', 'MESSIER033', 'MESSIER081', 'MESSIER101', 'MESSIER49', 'MESSIER59', 'MESSIER84', 'MESSIER86', 'MESSIER87', 'NGC0185', 'NGC0628', 'NGC0672 ', 'NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403'],['NGC2683', 'NGC2787', 'NGC3193', 'NGC3226', 'NGC3344', 'NGC3379', 'NGC4136', 'NGC4168', 'NGC4239', 'NGC4244', 'NGC4258', 'NGC4318', 'NGC4365', 'NGC4387', 'NGC4434', 'NGC4458', 'NGC4473', 'NGC4478', 'NGC4486B', 'NGC4489'],['NGC4551', 'NGC4559', 'NGC4564', 'NGC4600', 'NGC4618', 'NGC4660', 'NGC4736', 'NGC4826', 'NGC5194', 'NGC5195', 'NGC5322', 'NGC5638', 'NGC5813', 'NGC5831', 'NGC584', 'NGC5845', 'NGC5846', 'NGC596', 'NGC636', 'NGC6503'],['NGC6822', 'NGC6946', 'NGC720', 'NGC7454 ', 'NGC7640', 'NGC821', 'PEGASUS', 'SAG_DIR', 'SEXA', 'SEXB', 'SEXDSPH', 'UGC04879', 'UGCA127', 'UMIN']]
    
    # load all cadences
    with open('/mnt_blpc1/datax/scratch/calebp/boundaries/cappuccino/all_batches_all_cadences_1000.pkl', 'rb') as f:
        reloaded_batches = pickle.load(f)




    print(len(reloaded_batches))
    
    specific_batch = reloaded_batches[batch_number]
    feature_table = pd.read_csv(main_dir+df_name)

    # iterate through each target, grabbing the correct files. Files get grouped in cadences by node number and put in a list. 
    # for target in specific_batch:        
    #     print("Running boundary checker for target:",target)
    #     unique_h5_files,unique_nodes = get_all_h5_files(target)
    #     # print total number of files in target folder
    #     count = sum( [ len(listElem) for listElem in unique_h5_files])
    #     print(f"{count} files")
    #     # change back into main directory
    #     os.chdir(main_dir)

    try:
        # iterate through each node (cadence)
        for i in range(0,len(specific_batch)):
            print(f"Now on file {i} out of {len(specific_batch)}")
            # load current csv of file properties
            try:
                last_mason = pd.read_csv(main_dir+df_name)
                # grab the specific cadence to look at
                h5_files = specific_batch[i]
                # pass the files into the boundary_checker wrapper function. Returns flagged frequencies and respective scores
                print("Now running on file ",h5_files[0])
                k_score_table= pickler_wrapper(h5_files,block_size,significance_level)

                # append all flagged frequencies to the candidates database
                updated_mason = pd.concat([last_mason, k_score_table])
                updated_mason.to_csv(main_dir+df_name,index=False)

                print(updated_mason)
            except Exception:
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON CADENCE {i} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(traceback.print_exc())


    except Exception:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON TARGET {batch_number} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(traceback.print_exc())


def pickler_wrapper(h5_files,block_size,significance_level):
    hf_ON = h5py.File(h5_files[0], 'r')
    hf_OFF = h5py.File(h5_files[1], 'r')
    hf_ON2 = h5py.File(h5_files[2], 'r')
    hf_OFF2 = h5py.File(h5_files[3], 'r')
    hf_ON3 = h5py.File(h5_files[4], 'r')
    hf_OFF3 = h5py.File(h5_files[5], 'r')



    # grab specific rows which will be used to find hotspots
    obs1_row_16 = np.squeeze(hf_ON['data'][15:16,:,:])

    obs3_row_8 = np.squeeze(hf_ON2['data'][15:16,:,:])

    obs5_row_8 = np.squeeze(hf_ON3['data'][15:16,:,:])

    last_time_row_ON = obs1_row_16


    # find file frequency information
    fch1,foff = get_file_properties(hf_ON)

    # calculate number of iterations needed and find hotspots
    number = int(np.round(len(last_time_row_ON)/block_size))

    # record interesting freq chunks as 'warmspots'. This is the initial pass.
    hotspot_slices = [last_time_row_ON,obs3_row_8, obs5_row_8]

    number = int(np.round(len(last_time_row_ON)/block_size))

    all_warmspots = []

    print("Cutting cucumbers...")
    for i in range(0,len(hotspot_slices)):
        warmspots = find_warmspots(hotspot_slices[i],number,block_size)
        all_warmspots = all_warmspots+warmspots

   # keep only the unique blocks
    warmspots = [*set(all_warmspots)]

    # next filter out warmspots that fall in bad regions
    print("Throwing out rotten ones...")
    filtered_indexes = filter_hotspots(warmspots,fch1,foff,block_size)
    filtered_warmspots = np.delete(warmspots, filtered_indexes)

    # now sort through these warmspots and find hotspots --> higher signal
    all_hotspots = []
    for i in range(0,len(hotspot_slices)):
        hotspots = find_hotspots(hotspot_slices[i],filtered_warmspots,block_size,significance_level)
        all_hotspots=all_hotspots+hotspots

    filtered_hotspots = [*set(all_hotspots)]
    print("Final # of cucumbers:",len(filtered_hotspots))

    filtered_hotspots_slice_indexes = []
    for spot in filtered_hotspots:
        for i in range(0,len(hotspot_slices)):    
            row = hotspot_slices[i]
            slice_ON = row[spot*block_size:(spot+1)*block_size:]
            snr,threshold = get_snr(slice_ON,significance_level)
            if snr:
                filtered_hotspots_slice_indexes.append(i)
                break


    print("Salting and Seasoning...")
    k_score_table = get_k_scores(hf_ON,hf_OFF,hf_ON2,hf_OFF2,hf_ON3,hf_OFF3,filtered_hotspots,h5_files,fch1,foff,filtered_hotspots_slice_indexes,block_size)
    
    return k_score_table




def get_k_scores(hf_obs1,hf_obs2,hf_obs3,hf_obs4,hf_obs5,hf_obs6,filtered_hotspots,file_list,fch1,foff,filtered_hotspots_indexes,block_size):
    
    k_score_table = pd.DataFrame(columns=["All Files","Index","Freq","range1","range2","range3","range","obs1 maxes","obs3 maxes","obs5 maxes","k1","k2","k3","k4","k5","k6","k_score","min_k","med_k","max_k","drift"])
    # we iterate through all of the hotspots
    for i in tqdm(filtered_hotspots):

        # define the block region we are looking at
        try:
            lower = i * block_size
            upper = (i+1) * block_size 

            hotspot_index = filtered_hotspots.index(i)
            hotspot_slice = filtered_hotspots_indexes[hotspot_index]

            # get hit index
            observations_ON = [hf_obs1,hf_obs3,hf_obs5]
            primary_hf_ON = observations_ON[hotspot_slice]
            row_ON = np.squeeze(primary_hf_ON['data'][-1:,:,lower:upper],axis=1)[0]


                                
            Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
            obs1_int = Obs1.sum(axis=0)

            Obs2 = np.squeeze(hf_obs2['data'][:,:,lower:upper],axis=1)
            obs2_int = Obs2.sum(axis=0)

            Obs3 = np.squeeze(hf_obs3['data'][:,:,lower:upper],axis=1)
            obs3_int = Obs3.sum(axis=0)

            Obs4 = np.squeeze(hf_obs4['data'][:,:,lower:upper],axis=1)
            obs4_int = Obs4.sum(axis=0)

            Obs5 = np.squeeze(hf_obs5['data'][:,:,lower:upper],axis=1)
            obs5_int = Obs5.sum(axis=0)

            Obs6 = np.squeeze(hf_obs6['data'][:,:,lower:upper],axis=1)
            obs6_int = Obs6.sum(axis=0)
            
            on_sum = obs1_int+obs3_int+obs5_int
            off_sum = obs2_int+obs4_int+obs6_int

            whole_sum = obs1_int+obs3_int+obs5_int+obs2_int+obs4_int+obs6_int

            on_sum = on_sum/np.max(on_sum)
            off_sum = off_sum/np.max(off_sum)
            whole_sum = whole_sum/np.max(whole_sum)

            # calculate k-score
            cadence_max = np.max([np.max(Obs1),np.max(Obs2),np.max(Obs3),np.max(Obs4),np.max(Obs5),np.max(Obs6)])

            obs1_values = (Obs1/cadence_max).flatten()
            obs2_values = (Obs2/cadence_max).flatten()
            obs3_values = (Obs3/cadence_max).flatten()
            obs4_values = (Obs4/cadence_max).flatten()
            obs5_values = (Obs5/cadence_max).flatten()
            obs6_values = (Obs6/cadence_max).flatten()
        
        
            k1 = scipy.stats.kurtosis(obs1_values)
            k2 = scipy.stats.kurtosis(obs2_values)
            k3 = scipy.stats.kurtosis(obs3_values)
            k4 = scipy.stats.kurtosis(obs4_values)
            k5 = scipy.stats.kurtosis(obs5_values)
            k6 = scipy.stats.kurtosis(obs6_values)
            
            k_score = abs((k1+k3+k5)/(k2+k4+k6))

            off_k_sum = k2+k4+k6

            on_ks = [k1,k3,k5]

            med_k = np.median(on_ks)
            min_k = np.min(on_ks)
            max_k = np.max(on_ks)

            # calculate the ranges

            obs1_freq_int = Obs1.sum(axis=1)
            obs2_freq_int = Obs2.sum(axis=1)
            obs3_freq_int = Obs3.sum(axis=1)
            
            obs1_freq_int = obs1_freq_int/np.max(Obs1)
            obs2_freq_int = obs2_freq_int/np.max(Obs1)
            obs3_freq_int = obs3_freq_int/np.max(Obs1)

            range1 = np.max(obs1_freq_int)-np.min(obs1_freq_int)
            range2 = np.max(obs2_freq_int)-np.min(obs2_freq_int)
            range3 = np.max(obs3_freq_int)-np.min(obs3_freq_int)

            range_var = range1+range2+range3

            # calculate the changes in maximum

            observations = [Obs1/np.max(Obs1),Obs3/np.max(Obs3),Obs5/np.max(Obs5)]
            # also calculate max value at each time integration point

            obs_time_maxes = []
            for number in [0,1,2]:
                time_maxes = []
                for time in range(16):
                    time_max = np.max(observations[number][time])
                    time_maxes.append(time_max)
                obs_time_maxes.append(time_maxes)

            

            # check drift rate
            zero_drift = drift_index_checker(on_sum, row_ON,10,10)
            drift = 1
            if zero_drift == True:
                drift = 0

            
            frequency = fch1+foff*(i*block_size)
            k_score_table.loc[len(k_score_table.index)] = [file_list,i,frequency,range1,range2,range3,range_var,obs_time_maxes[0],obs_time_maxes[1],obs_time_maxes[2],k1,k2,k3,k4,k5,k6,k_score,min_k,med_k,max_k,drift]
        except Exception:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON BLOCK {i} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(traceback.print_exc())
            k_score_table.loc[len(k_score_table.index)] = [file_list,i,fch1+foff*(i*block_size),math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan,math.nan]

    return k_score_table
























"""Data Functions"""





def get_all_h5_files(target):
    """Returns a list containaing cadences grouped together as tuples, as well as a list of all unique nodes

    Args:
        target (str): Galaxy/Star (or overarching file folder) you are looking at

    :Returns:
        - h5_list (list): list containaing cadences grouped together as tuples
        - unique_nodes (list): list of all unique nodes
    """


    # initialize list to store h5 files
    h5_list = []

    # first change directory into the target directory
    os.chdir(target)
    data_dir = os.getcwd() + "/"

    # we want to get all the unique nodes
    unique_nodes = get_unique_nodes(data_dir)

    for node in unique_nodes:
    # then loop through and grab all the file names
        node_set = get_node_file_list(data_dir,node)
        h5_list.append(node_set)

    return h5_list, unique_nodes


def get_unique_nodes(data_dir):
    """Grabs the unique blc nodes in a given directory

    Args:
        data_dir (str): Data directory to search through

    Returns:
        unique_nodes (list): List of all unique nodes in the directory, sorted.
    """
    node_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            # we remove the start and end nodes as these have low sensitivity
            if "blc" in filename and (filename[4] != '7') and (filename[4] != '0'):
                node_list.append(filename[:5])

    node_set = set(node_list)
    print("Unique nodes:", node_set)

    unique_nodes = sorted(node_set)
    unique_nodes.sort()
    return unique_nodes

def get_node_file_list(data_dir,node_number):
    """Returns the list of h5 files associated with a given node

    Args:
        data_dir (str): Data directory to search through
        node_number (str): Node number to filter on

    Returns:
        data_list (list): List of h5 files that make up the cadence, sorted chronlogically
    """


    ## h5 list
    data_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename[-3:] == '.h5' and node_number in filename:
                data_list.append(data_dir + filename)
                
    data_list = sorted(data_list, key=lambda x: (x,x.split('_')[5]))

    return data_list

if __name__ == '__main__':
    batch_number = sys.argv[1]
    main(batch_number)