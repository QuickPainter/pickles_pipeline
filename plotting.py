import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import traceback
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import scipy
from matplotlib import pyplot as plt, cm
from matplotlib import colors

def get_file_properties(f):
    """Get file properties of given h5 file.

    Args:
        f (h5 object): h5 file corresponding to desired observation

    Returns:
        fch1 (float): start frequency of observation in Mhz
        foff (float): frequency of each bin in Mhz

    """
    tstart=f['data'].attrs['tstart']
    fch1=f['data'].attrs['fch1']
    foff=f['data'].attrs['foff']
    nchans=f['data'].attrs['nchans']
    ra=f['data'].attrs['src_raj']
    decl=f['data'].attrs['src_dej']
    target=f['data'].attrs['source_name']
    # print("tstart %0.6f fch1 %0.10f foff %0.30f nchans %d cfreq %0.10f src_raj %0.10f src_raj_degs %0.10f src_dej %0.10f target %s" % (tstart,fch1,foff,nchans,(fch1+((foff*nchans)/2.0)),ra,ra*15.0,decl,target))

    return fch1, foff, nchans, ra, decl, target
    
def plot_candidates_sparse(hf1,hf2,hf3,hf4,hf5,hf6,lower,upper,file_ON,foff,fch1,block_size,batch_info,ax,parent_grid,position,buffer):
    obs1 = np.squeeze(hf1['data'][:,:,lower:upper],axis=1)
    obs2 = np.squeeze(hf2['data'][:,:,lower:upper],axis=1)
    obs3 = np.squeeze(hf3['data'][:,:,lower:upper],axis=1)
    obs4 = np.squeeze(hf4['data'][:,:,lower:upper],axis=1)
    obs5 = np.squeeze(hf5['data'][:,:,lower:upper],axis=1)
    obs6 = np.squeeze(hf6['data'][:,:,lower:upper],axis=1)

    cadence_max = np.max([np.max(obs1),np.max(obs2),np.max(obs3),np.max(obs4),np.max(obs5),np.max(obs6)])
    
    obs1_values = (obs1/cadence_max).flatten()
    obs2_values = (obs2/cadence_max).flatten()
    obs3_values = (obs3/cadence_max).flatten()
    obs4_values = (obs4/cadence_max).flatten()
    obs5_values = (obs5/cadence_max).flatten()
    obs6_values = (obs6/cadence_max).flatten()

    k1 = scipy.stats.kurtosis(obs1_values)
    k2 = scipy.stats.kurtosis(obs2_values)
    k3 = scipy.stats.kurtosis(obs3_values)
    k4 = scipy.stats.kurtosis(obs4_values)
    k5 = scipy.stats.kurtosis(obs5_values)
    k6 = scipy.stats.kurtosis(obs6_values)


    obs1 = obs1/np.max(obs1)
    obs2 = obs2/np.max(obs2)
    obs3 = obs3/np.max(obs3)
    obs4 = obs4/np.max(obs4)
    obs5 = obs5/np.max(obs5)
    obs6 = obs6/np.max(obs6)

    obs1_summed = np.sum(obs1, axis=1)
    obs3_summed = np.sum(obs3, axis=1)
    obs5_summed = np.sum(obs4, axis=1)
    


    full_cadence = np.squeeze([np.concatenate((obs1,obs2,obs3,obs4,obs5,obs6))])

    
    center_freq = fch1+foff*(lower)    
    name = file_ON.split('/')[-1]

    target = name.split("_")[-2]
    if target == 'OFF':
        target = name.split("_")[-3]
    obs_num = name.split("_")[-1]
    MJD = name.split("_")[4]
    node = name.split("_")[0]


    obs_list = [obs1, obs2, obs3, obs4, obs5, obs6]
    group_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=parent_grid[position])
    ax = plt.subplot(group_grid[0])
    
    # Now, instead of creating a new figure, use the provided axes to plot
    # We mimic creating 6 subplots vertically within this single subplot area
    inner_grid = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=group_grid[0], hspace=0)

    on_ks = list(np.around(np.array([k1,k3,k5]),2))
    off_k_sum = np.round((k2+k4+k6),2)
    for i in range(6):
        ax = plt.subplot(inner_grid[i])
        ax.imshow(obs_list[i], aspect='auto', extent=[-((block_size/2)*foff*10**3), ((block_size/2)*foff*10**3), 299, 0], norm=colors.LogNorm(), cmap='afmhot')
        ax.set_ylabel("Time [s]", fontsize=6)
        if i == 0:
            ax.set_title(f"Target: {target} -- MJD: {MJD} -- Node: {node}", fontsize=10)  # Update with dynamic title if needed
        if i == 5:
            ax.set_xlabel(f"Rel. Freq. [kHz] from {np.round(center_freq,5)} Mhz",fontsize=10)
            ax.annotate(f'File: {name}', (0,0), (400, 500), xycoords='axes fraction', textcoords='offset points', va='top',rotation='vertical',fontsize=12)
            ax.annotate(f'lower,upper: {(lower,upper)} ---- batch info: {batch_info} ---- off_k: {off_k_sum}, on_ks: {on_ks}', (0,0), (415, 500), xycoords='axes fraction',rotation='vertical', textcoords='offset points', va='top',fontsize=12)
        ax.tick_params(axis='y', labelsize=6)
        ax.axvline(x=-((block_size/2-(buffer-50))*foff*10**3),linewidth=1,linestyle='--',color='white')
        ax.axvline(x=((block_size/2-(buffer-50))*foff*10**3),linewidth=1,linestyle='--',color='white')



def single_plot_wrapper(num,high_k_outliers):
    fig = plt.figure(figsize=(7, 9))
    fig_grid = gridspec.GridSpec(1, 1, hspace=0.2)

    k_regions = high_k_outliers
    all_files = np.array(k_regions["All Files"])[num]
    batch_info = np.array(k_regions["Batch Info"])[num]
    drift2 = np.array(k_regions["drift2"])[num]

    all_files = eval(all_files)
    # freq = np.array(k_regions["freq"])[num]

    hf_ON = h5py.File(all_files[0], 'r')
    hf_OFF = h5py.File(all_files[1], 'r')
    hf_ON2 = h5py.File(all_files[2], 'r')
    hf_OFF2 = h5py.File(all_files[3], 'r')
    hf_ON3 = h5py.File(all_files[4], 'r')
    hf_OFF3 = h5py.File(all_files[5], 'r')
    file_ON = all_files[0]
    hf_ON = h5py.File(all_files[0], 'r')
    fch1, foff, nchans, ra, decl, target = get_file_properties(hf_ON)

    block_size = int(np.array(k_regions["Block Size"])[num])
    i = np.array(k_regions["Index"])[num]
    freq = np.array(k_regions["Freq"])[num]
    i = ((freq-fch1)/foff)/block_size

  
    med_k =np.array(k_regions["med_k"])[num]
    min_k = np.array(k_regions["min_k"])[num]
    freq = np.array(k_regions["Freq"])[num]
    
    new_k = med_k * min_k**2

    k2 = np.array(k_regions["k2"])[num]
    k4 = np.array(k_regions["k4"])[num]
    k6 = np.array(k_regions["k6"])[num]

    off_ks = k2+k4+k6
    
    lower = int((i) * block_size)
    upper = int((i+1) * block_size)

    
    file_ON = all_files[0]
    name = file_ON.split('/')[-1]

    target = name.split("_")[-2]
    if target == 'OFF':
        target = name.split("_")[-3]
    obs_num = name.split("_")[-1]
    MJD = name.split("_")[4]
    node = name.split("_")[0]

    
    plt.rcParams["figure.figsize"] = (5,5)  
    
    buffer = 250
    lower = lower - buffer
    upper = upper + buffer
    plot_candidates_sparse(hf_ON,hf_OFF,hf_ON2,hf_OFF2,hf_ON3,hf_OFF3,lower,upper,file_ON,foff,fch1,block_size+2*buffer,batch_info,0,fig_grid,0,buffer)
    file_name = f'/mnt_blpc1/datax/scratch/calebp/pickles/candidates/candidate_batch_{eval(batch_info)[0]}_number_{num}_source_{target}_date_{MJD}_node_{node}_lower_{lower}_upper_{upper}.png'
    plt.savefig(file_name,dpi=100)  # Adjust path as needed
    plt.close()

    return file_name, (fch1, ra, decl, target)


    
def main():
    """
    Loads a CSV file into a pandas DataFrame, adds a new column with file names
    for saving plots, and applies a function to each row to populate that column.
    
    Args:
    - file_path: The path to the CSV file.
    
    Returns:
    - A pandas DataFrame with the additional 'saved_plot_file_name' column.
    """
    # Load the CSV file into a DataFrame
    csv_file = 'databases/all_outiers_4_9_24.csv'
    
    all_outliers = pd.read_csv(csv_file)

    if 'saved_plot_file_name' not in all_outliers.columns: 
        all_outliers['saved_plot_file_name'] = pd.Series(dtype='object')
    if 'info' not in all_outliers.columns: 
        all_outliers['info'] = pd.Series(dtype='object')

    print(len(all_outliers))
    print(all_outliers.head())
    filenames = []
    obs_infos = []
    for num in tqdm(range(5,len(all_outliers))):
        try:
            filename, obs_info = single_plot_wrapper(num,all_outliers)
            all_outliers.at[num, 'saved_plot_file_name'] =filename
            print(obs_info)
            all_outliers.at[num, 'info'] =obs_info
        except Exception:
            print(traceback.print_exc())

    print(all_outliers.head())
    all_outliers.to_csv(csv_file)


if __name__ == '__main__':
    main()