import torch
if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'
    
    
import pandas as pd

import numpy as np
import os
import json
import yaml
import h5py as h5
from os import listdir
from gwpy.signal import filter_design
from scipy.signal import sosfilt_zi
import torchaudio.functional as fn
from tqdm import tqdm

class TFrame:
    
    __slots__=['tdata','row_list','col_list','ann_dict']
    
    def __init__(self,data,rows,cols,anns):
        
        self.tdata=data
        self.row_list=rows
        self.col_list=cols
        
        #possible annotations: sr, u.m.
        self.ann_dict=anns
        #"sample_rate"
        
    def thead(self):
        
        df = pd.DataFrame(self.tdata[:5,:,0].numpy(), index=self.row_list[:5], columns=self.col_list)
        
        return df
    
    

def oddpadding(x, padlen):
    
    if padlen == 0:
        return x
    
    # Estremi del segnale (primo e ultimo campione)
    x_start = x[..., :1]  # shape (..., 1)
    x_end = x[..., -1:]   # shape (..., 1)
    
    # Padding sinistro: 2*x[0] - x[1:padlen+1][::-1]
    left_pad = 2 * x_start - x[..., 1:padlen+1].flip(-1)
    
    # Padding destro: 2*x[-1] - x[-padlen-1:-1][::-1]
    right_pad = 2 * x_end - x[..., -padlen-1:-1].flip(-1)
    
    
    
    # Concatena tutto
    return torch.cat([left_pad, x, right_pad], dim=-1)
    
def bandpass_chb1(data,matr,matr_zi,dev,padlen=30):
    
    #const=1e10
    
    dim_tensor=torch.tensor(data.shape,device=dev)
    
    
    #data/=const
    
    
    
    data_reshaped=data.view(-1,data.shape[2])
    
    
    data_reshaped=oddpadding(data_reshaped, padlen)
    
    
    zi_pad_len=matr_zi.shape[1]
    
    
    b=matr[0,:3]
    a=matr[0,3:]
    
    
    zi0=data_reshaped[...,0]*matr_zi[0,0]
    zi1=data_reshaped[...,0]*matr_zi[0,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    
    
    data_reshaped=torch.cat([zi ,data_reshaped],dim=-1)
    
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    
    
    
    b=matr[1,:3]
    a=matr[1,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[1,0]
    zi1=data_reshaped[...,0]*matr_zi[1,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi ,data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    
    b=matr[2,:3]
    a=matr[2,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[2,0]
    zi1=data_reshaped[...,0]*matr_zi[2,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi,data_reshaped],dim=1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    b=matr[3,:3]
    a=matr[3,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[3,0]
    zi1=data_reshaped[...,0]*matr_zi[3,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    
    data_reshaped=torch.cat([zi,data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    
    b=matr[4,:3]
    a=matr[4,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[4,0]
    zi1=data_reshaped[...,0]*matr_zi[4,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi,data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    
    
    data_reshaped=data_reshaped.flip(-1)
    
    
    b=matr[0,:3]
    a=matr[0,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[0,0]
    zi1=data_reshaped[...,0]*matr_zi[0,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    
    
    data_reshaped=torch.cat([zi, data_reshaped],dim=1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
   
    
    b=matr[1,:3]
    a=matr[1,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[1,0]
    zi1=data_reshaped[...,0]*matr_zi[1,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi, data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    b=matr[2,:3]
    a=matr[2,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[2,0]
    zi1=data_reshaped[...,0]*matr_zi[2,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi, data_reshaped],dim=-1)
    
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    b=matr[3,:3]
    a=matr[3,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[3,0]
    zi1=data_reshaped[...,0]*matr_zi[3,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi, data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    b=matr[4,:3]
    a=matr[4,3:]
    
    zi0=data_reshaped[...,0]*matr_zi[4,0]
    zi1=data_reshaped[...,0]*matr_zi[4,1]
    
    zi=torch.stack([zi0,zi1],dim=0).T
    
    data_reshaped=torch.cat([zi, data_reshaped],dim=-1)
   
    data_reshaped=fn.lfilter(data_reshaped,a,b,clamp=False)
    
    data_reshaped=data_reshaped[...,zi_pad_len:]
    
    
    
    data_reshaped=data_reshaped.flip(-1)
    
    
    
    data_reshaped=data_reshaped[...,padlen:-padlen]
    
    
    
    
    
    data_reshaped=data_reshaped.view(dim_tensor[0],dim_tensor[1],dim_tensor[2])
    
    
    
    
    return data_reshaped


def crop(data,clen,srate):
    
    tlen=data.shape[2]*(1/srate)
    
    
    
    
    idx0=int(((tlen-clen)/2)*srate)
    idx1=int(((tlen+clen)/2)*srate)
    
    
    
     
    data=data[:,:,idx0:idx1]
    
    
    
    return data
        

def construct_tensor_data(path, channel_list=None, target_channel='V1:Hrec_hoft_16384Hz', 
                        n1_events=None, n2_events=None, n1_channels=None, n2_channels=None, print_=True, sr=False,crp=False,batch=2000,type='chb1',f1=8,f2=500):
    """
    Construct a DataFrame from data stored in HDF5 files.

    Parameters:
    - path (str): The directory path where the HDF5 files are located.
    - channel_list (list): A list of channel names to include in the DataFrame. If not provided, it defaults to None, and the code will load all channels in the file.
    - target_channel (str): The target channel to include in the DataFrame. Default value is 'V1:Hrec_hoft_16384Hz'.
    - n1_events (int): The starting index of events to consider. Default value is None, which corresponds to first file in directory.
    - n2_events (int): The ending index of events to consider. Default value is None, which corresponds to last file in directory.
    - n1_channels (int): The starting index of channels to consider. Default value is None, which corresponds to first channel in file.
    - n2_channels (int): The ending index of channels to consider. Default value is None, which corresponds to last channel in directory.
    - print_ (bool): A boolean indicating whether to print progress information. Default value is True.
    - sr (float or bool): New sample rate for resampling the data. Default value is False, which stands for no resampling.
    - tensor_mode (bool): A boolean indicating whether to load the timeseries as torch tensors

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the HDF5 files.
    """

    # Set default values for event and channel indices if not provided
    if not n1_events:
        n1_events = 0
    if not n2_events:
        n2_events = len(listdir(path))
    
    # Ensure n2_events does not exceed the total number of files in the directory
    if n2_events > len(listdir(path)):
        n2_events = len(listdir(path))
    
    # Get the list of files in the specified directory
    lstdr = listdir(path)[n1_events:n2_events]
    
    # Print the list of files being processed if print_ is True
    #if print_:
        #print(f'LIST DIR: {lstdr}')
    
    # Extract the name of a sample file from the directory
    sample_file = listdir(path)[0]
    
    # Create a list of files to process
    files = [f for f in lstdr]
    
    # Initialize lists to store DataFrame and event data
    df_list = []
    event_data = []
    sampr=0.0
    t0=0.0
    # If channel_list is not provided, get all channels from the first HDF5 file
    if not channel_list:
        n_all_channels = 0
        all_channels = []
        with h5.File(os.path.join(path, sample_file), 'r') as fout:
            event_id = list(fout.keys())[0]
            all_channels = list(fout[event_id])
            n_all_channels = len(list(fout[event_id]))
            sampr=fout[event_id][target_channel].attrs['sample_rate']
            t0=fout[event_id][target_channel].attrs['t0']
        
        # Set default values for channel indices if not provided
        if not n1_channels:
            n1_channels = 0
        if not n2_channels:
            n2_channels = n_all_channels
        
        # Ensure n2_channels does not exceed the total number of channels
        if n2_channels > n_all_channels:
            n2_channels = n_all_channels
        
        # Select channels based on provided indices
        channels = all_channels[n1_channels:n2_channels]
    else:
        channels = channel_list
        with h5.File(os.path.join(path, sample_file), 'r') as fout:
            sampr=fout[event_id][target_channel].attrs['sample_rate']
            t0=fout[event_id][target_channel].attrs['t0']
        
       #CHANNELS, CHANNELS DICT
    
    # Remove the target channel from the list of channels
    try:
        channels.remove(target_channel)
    except:
        pass
    
    #nerr=0
    #nfop=0
   
    
    # Iterate over each file and extract data
    for i, file in enumerate(files):
        if print_:
            print(f"Added {i + 1}/{n2_events - n1_events} files to dataframe", end='\r')
            
        np_list=[]    
       
        try:
            # Open the HDF5 file
            with h5.File(os.path.join(path, file), 'r') as fout:
                event_id = list(fout.keys())[0]
                #dictionary = {'Event ID': event_id}
                #print(event_id,"\n")
                
                #EVENT DICT
                #event_data.append(event_id)
                
                # Extract data for the target channel
                tmsrs = fout[event_id][target_channel][:]
               
                
               
                
                
                
        
                        
                        
                
                tmsrs=torch.tensor(tmsrs,dtype=torch.float64)
                
                np_list.append(tmsrs)
                
                # Extract data for each channel
                for j, channel in enumerate(channels):
                    try:
                       
                        #if i use torch i don't have to use gwpy's timeseries
                        tmsrs = fout[event_id][channel][:]
                        
                        
                       
                            
                        
                        tmsrs=torch.tensor(tmsrs,dtype=torch.float64)
                        np_list.append(tmsrs)
                        #print("Chan","\n")
                        #print(tmsrs.shape)
                    except Exception as e:
                        # Handle errors in extracting data
                        tmsrs = np.nan
                        tmsrs=torch.from_numpy(tmsrs).float()
                        np_list.append(tmsrs)
                        print(e)
                        
                        
                
                #Convert the list to a tensor and append to df_list
                #tensor_list = [torch.from_numpy(item).float() for item in np_list]
                
                row=torch.stack(np_list,axis=0)
                #print(row.shape)
                
                df_list.append(row)
                event_data.append(event_id)
                #print("Added row: ",i, event_id,len(event_data))
                
                #nfop+=1
                
               
                
        
        except Exception as e:
            # Handle errors in opening files
            if print_:
                print(f'COULD NOT OPEN {os.path.join(path, file)}')
                print(e)
            
    # Concatenate all DataFrames in df_list into a single DataFrame
    
    
    
    channels.insert(0,target_channel)
    
    
    
    #send to gpu to process
    
    
    
    
    #PREPROCESSING STEP
    
    #BANDPASS
    
    
    
    
    #ord,w=cheb1_min_ord([8,500],sampr)
    #b,a=cheby1(ord,2,w,btype='band')
    
    #ord,w=butter_min_ord([8,500],sampr)
    
    
    
    #if(type=='btt'):
     #sos=filter_design.bandpass(8, 500, 4096,ftype='butter',output='sos')
     
    
    if(type=='chb1'):
        sos=filter_design.bandpass(8, 500, 4096,ftype='cheby1',output='sos')
     
    
    
   
    
    sos_tensor=torch.tensor(sos,dtype=torch.float64,device=device)
    #print(sos)
    #sos_tensor[:,3:]=sos_tensor[:,3:]/sos_tensor[:,3:4]
    sos_zi=torch.tensor(sosfilt_zi(sos),dtype=torch.float64,device=device)
    
    
    
    processed_list=[]
    
    
    print('\n','Processing dataset...')
    for i in tqdm(range(0, len(df_list), batch)):
     
     
    
        df=torch.stack(df_list[i:i+batch],axis=0)
        
         
     
    
        df=df.to(device)
        
    
    

    
    
     #df=bandpass_chb1(df,sos_tensor,device)
        
        
    
       
        df=bandpass_chb1(df,sos_tensor,sos_zi,device)
        
     
    
        if(crp):
            df=crop(df,crp,sampr)
    
     
        if(sr):
            df=fn.resample(df,sampr,sr)
            
            
        ########add whiten############    
            
        #############################    
        
        df=df.to('cpu')
        
        
        processed_list.append(df)
     
    
    
    df_proc=torch.cat(processed_list,dim=0)
    del sos_tensor
    torch.cuda.empty_cache()
    
    
    #print("Device transfer time: ", (e-s) )
    return df_proc, event_data, channels


def merge_tframes(tf_list):
    
    comm_ids=set(tf_list[0].row_list)
    for lst in tf_list[1:]:
        comm_ids &=set(lst.row_list)
    
    new_rows=list(comm_ids)
    
    new_cols=[]
    new_cols+=tf_list[0].col_list
    for lst in tf_list[1:]:
        new_cols+=lst.col_list
        
        
        
    temp_list=[]
    stack_list=[]
    for id in new_rows:  
        temp_list=[]
        buff=tf_list[0].tdata[tf_list[0].row_list.index(id),:,:]
        temp_list.append(buff)
        
        for lst in tf_list[1:]:
            buff=lst.tdata[lst.row_list.index(id),:,:]   
            temp_list.append(buff)
            
        row=torch.cat(temp_list,axis=0)
        stack_list.append(row)
   
    dfm=torch.stack(stack_list,axis=0)
    
    tf_stack=TFrame(dfm,new_rows,new_cols,{"sample_rate":tf_list[0].ann_dict['sample_rate']})
    
    return tf_stack
    

def save_tensor(tensor,savepath,name):
    
    torch.save(tensor, f'{savepath}/{name}.pt')
    
    
    
def save_to_json(data, filename, folder="."):
    """
    Save a dictionary or list to a JSON file in the specified folder.
    
    Args:
        data: Dictionary or list to be saved.
        filename (str): Name of the JSON file (without .json extension).
        folder (str): Target folder path (default: current directory).
    
    Returns:
        str: Full path to the saved file.
    
    Raises:
        TypeError: If data is not a dictionary or list.
        OSError: If the folder cannot be created.
    """
    # Validate input type
    if not isinstance(data, (dict, list)):
        raise TypeError("Data must be a dictionary or list.")
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Add .json extension if not present
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Build full path
    full_path = os.path.join(folder, filename)
    
    # Write JSON data with pretty formatting
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return full_path


def read_tensor_lib(path):
    
    strerr=''
   
    addtensors=[]
    
    manifestpath=f'{path}/manifest.yaml'
    tensorpath=f'{path}/tensor.pt'
    
    
    rowspath=f'{path}/rows.json'
    colspath=f'{path}/cols.json'
    metapath=f'{path}/meta.json'
   
    
    
    
    try:
        tensordata=torch.load(tensorpath,weights_only=True)
        #weight_only experimental feature, prevents code injection
    
    
        with open(manifestpath, 'r') as file:
            datasets = yaml.safe_load(file)
        
        for additional in datasets['tensors']['additional']:
            addpath=f'{path}/{additional}'
            
            addtensor=torch.load(addpath,weights_only=True)
            addtensors.append(addtensor)
            
            
        with open(rowspath, 'r') as file:
            rows = json.load(file)
            
        with open(colspath, 'r') as file:
            cols = json.load(file)
            
        with open(metapath, 'r') as file:
            meta = json.load(file)    
            
            
        
        
    except Exception as e:
        strerr=repr(e)
        
        
        
        
    if(not(strerr)):
        tf=TFrame(tensordata,rows,cols,meta)
    else:
        tf=None
        addtensors=None
    
    return strerr,tf,addtensors

#--------------------------------------Training functions------------------------------------------------------------------------------------------#

def augment_data(tensor, num_slices):
    B, C, H, W = tensor.shape
    W0 = H  # Target width is now H
    offset = (W - num_slices * W0) // 2

    selected_chunks = tensor[:, :, :, offset:offset + num_slices * W0].view(B, C, H, num_slices, W0)
    tensor_permuted = selected_chunks.permute(0, 3, 1, 2, 4)
    augmented_tensor = tensor_permuted.contiguous().view(B * num_slices, C, H, W0)
    return augmented_tensor


def augment_dataset(train_data,test_data):
    
    # Augment training data (3 slices)
    train_data_augmented_3 = augment_data(train_data, 3)

    # Augment training data (2 slices)
    train_data_augmented_2 = augment_data(train_data, 2)

    train_data_2d = torch.cat([train_data_augmented_3, train_data_augmented_2], dim=0)

    # Augment validation data (3 slices)
    val_data_augmented_3 = augment_data(test_data, 3)

    # Augment validation data (2 slices)
    val_data_augmented_2 = augment_data(test_data, 2)

    test_data_2d = torch.cat([val_data_augmented_3, val_data_augmented_2], dim=0)
    
    return train_data_2d, test_data_2d


def filter_rows_below_threshold(data, threshold):
    """
    Filters rows in the data tensor where all channels are below a certain threshold.

    Input:
    - data (torch.Tensor): dataset
    - threshold (torch.Tensor): threshold value for each channel

    Return:
    - filtered_data (torch.Tensor): filtered dataset
    """
    # Calculate the maximum value for each channel across all examples
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]
    #print(max_vals.shape)
    #print(threshold.unsqueeze(0).shape)
    # Check if all three values in each row are below the respective threshold
    mask = (max_vals < threshold.unsqueeze(0)).all(dim=1)
    #print(mask.shape)
    
    # Use the boolean mask to filter and keep only the rows in the dataset that satisfy the condition
    filtered_data = data[mask]

    return filtered_data,mask


def filter_rows_above_threshold(data, threshold):
    """
    Filters rows in the data tensor where all channels are below a certain threshold.

    Input:
    - data (torch.Tensor): dataset
    - threshold (torch.Tensor): threshold value for each channel

    Return:
    - filtered_data (torch.Tensor): filtered dataset
    """
    # Calculate the maximum value for each channel across all examples
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]
    #print(max_vals.shape)
    #print(threshold.unsqueeze(0).shape)
    # Check if all three values in each row are below the respective threshold
    mask = (max_vals >= threshold.unsqueeze(0)).all(dim=1)
    #print(mask.shape)
    
    # Use the boolean mask to filter and keep only the rows in the dataset that satisfy the condition
    filtered_data = data[mask]

    return filtered_data,mask


def find_max(data):
    #print(data.shape)
    """
    Normalizes the qplot data to the range [0,1] for NN convergence purposes
    
    Input:
    - data (torch.Tensor) : dataset of qtransforms
    
    Return:
    - data (torch.tensor) : normalized dataset
    """
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]  # Compute the maximum value for each 128x128 tensor
    max_global = data.view(data.shape[0], data.shape[1], -1).max(0)[0].max(1)[0]
    #print(max_global)
    print("Maximum value for each element tensor:", max_vals.shape)
    max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)  # Add dimensions to match the shape of data for broadcasting
    return max_vals


def normalize_ch_mean(data, channel_means, channel_std=None):
    
    #MODIFY WITH MEDIAN
    """
    Normalizes the data by dividing each channel by its respective mean value,
    or by subtracting the mean and dividing by the standard deviation if channel_std is provided.

    Input:
    - data (torch.Tensor): dataset
    - channel_means (list or torch.Tensor): list of mean values for each channel
    - channel_std (list or torch.Tensor, optional): list of standard deviation values for each channel. Defaults to None.

    Return:
    - normalized_data (torch.Tensor): normalized dataset
    """
    # Convert channel_means and channel_std to tensors if they're not already
    if not isinstance(channel_means, torch.Tensor):
        channel_means = torch.tensor(channel_means)
    if channel_std is not None and not isinstance(channel_std, torch.Tensor):
        channel_std = torch.tensor(channel_std)


    # Check if channel_means has the correct shape
    if channel_means.shape[0] != data.shape[1]:
        raise ValueError("Number of elements in channel_means must match the number of channels in data.")

    # Reshape channel_means and channel_std to match the shape of data for broadcasting
    channel_means = channel_means.view(1, -1, 1, 1)
    if channel_std is not None:
        if channel_std.shape[0] != data.shape[1]:
            raise ValueError("Number of elements in channel_std must match the number of channels in data.")
        channel_std = channel_std.view(1, -1, 1, 1)

    # Normalize data
    if channel_std is None:
        normalized_data = data / channel_means
    else:
        normalized_data = (data - channel_means) / channel_std

    return normalized_data
