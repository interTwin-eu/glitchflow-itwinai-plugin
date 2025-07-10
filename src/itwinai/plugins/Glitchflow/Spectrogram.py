import torch
from .config import device
    
    
from torch.utils.data import DataLoader
import numpy as np


import itwinai
from itwinai.components import Trainer, monitor_exec

from typing import List,Optional 
#another possible types: Tuple Optional


import matplotlib.pyplot as plt

from tqdm import tqdm

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"#VRAM defrag
import time
from datetime import datetime
import yaml


from . Data import TFrame,YamlWrapper,plot_to_f,set_frequency_ticks,plot_stacked_tensor

from . Data import save_tensor, saveYaml

from . Annalisa_gpu import * 


class Preprocess(Trainer):
    def __init__(self,params: YamlWrapper,conf:YamlWrapper,whiteconf:YamlWrapper,logger: None=None,save:Optional[bool] = False,source:str='',
                 chans:Optional[str]|Optional[List]=None,
                 desired_ticks: List=[8, 20, 30, 50, 100, 200, 500],
                 log_base:float=np.e,
                  phase:bool=False,
                 num_plots:int=10,
                 v_max:int=25,
                 v_min:int=0,length_in_s:int=6,save_gr_par:bool=False) -> None:
        
        """
        params: wrapper for process.yaml. A list with parameters will be read
        conf: wrapper for saveconf.yaml
        whiteconf: wrapper for whiten.yaml
        save_gr_par: if True a sample of images will be saved with the dataset. The parameters used will be also saved

        source: path to dataset
        chans: path to channels or python list

        These parameters are necessary for a standalone use

        desired_ticks: graph frequency ticks
        log_base log :scale
        phase: only fpr phase mode
        num_plots: number of saved images
        v_max, vmin: colormap normalization
        length_in_s: time axis length
        
        """
        
        self.logger=logger
        self.params=params.flist
        self.fconf=conf.flist
        self.save=save
        self.source=source
        self.chans=chans
        self.whiteconf=whiteconf.flist['whiten']
        self.desired_ticks=desired_ticks
        self.log_base=log_base
        self.phase=phase
        self.num_plots=num_plots
        self.v_max=v_max
        self.v_min=v_min
        self.length_in_s=length_in_s
        self.save_gr_par=save_gr_par
        
    def get_or_load_chs(self, chs: Optional[str]|Optional[List] = None):
        """If the dataset is not given, load it from disk."""
        if type(chs)==str:
            
            channels=torch.load(chs,weights_only=True)
            
            
            
            return channels.tolist()
        
        return chs           
        
    def get_or_load(self, dataset: Optional[TFrame] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            print("WARNING: loading time series dataset from disk.")
            
            tf=TFrame(0,[],[],[],{})
            
            tf.load(self.source)
            
            return tf
        
        return dataset
    
    
    
        
        
        

    @monitor_exec
    def execute(self,datalist:Optional[TFrame]=None) -> int:
        
        
        ##############Processing parameters space############
        
        proc_set=self.get_or_load(datalist)
        
        if(self.chans):
            indices_to_select=self.get_or_load_chs(self.chans)
            
        else:
            indices_to_select=proc_set.ann_dict["chans"]
            
        print(indices_to_select)    
            
        
        
        data=proc_set.tdata
        rows_list=proc_set.row_list
        column_list=proc_set.col_list
        gps_list=proc_set.gps
        ann_dict=proc_set.ann_dict
        timestamp=proc_set.ann_dict["id"]
        tag=proc_set.ann_dict["tag"]
        tslen=data.shape[2]
        
        print(f'Found tensor of shape({data.shape[0]},{data.shape[1]},{tslen})')
        #indices_to_select=proc_set[1]
        #print(data)
        sr=proc_set.ann_dict["sample_rate"]
        print(sr)
        
        
        num_batch=self.params['dataset_proc']['parameters']['num_batch']
        shuffle=self.params['dataset_proc']['parameters']['shuffle']
        qslicemin=self.params['dataset_proc']['parameters']['qslicemin']
        qslicemax=self.params['dataset_proc']['parameters']['qslicemax']
        
        q=self.params['dataset_proc']['parameters']['q']
        frange=self.params['dataset_proc']['parameters']['frange']
        
        fres=self.params['dataset_proc']['parameters']['fres']
        tres=self.params['dataset_proc']['parameters']['tres']
        num_t_bins=self.params['dataset_proc']['parameters']['num_t_bins']
        num_f_bins=self.params['dataset_proc']['parameters']['num_f_bins']
        logf=self.params['dataset_proc']['parameters']['logf']
        qtile=self.params['dataset_proc']['parameters']['qtile']
        whiten=self.params['dataset_proc']['parameters']['whiten']
        psd=self.params['dataset_proc']['parameters']['psd']
        
        #new params
        energy_mode=self.params['dataset_proc']['parameters']['energy_mode']
        phase_mode=self.params['dataset_proc']['parameters']['phase_mode']
        window_param=self.params['dataset_proc']['parameters']['window_param']
        tau=self.params['dataset_proc']['parameters']['tau']
        beta=self.params['dataset_proc']['parameters']['beta']
        
        
        
        
        ############################################################################
        
        ################# results######################
        resdir=self.fconf['res']['processed']['dir']
        qtensor=self.fconf['res']['qtensor']['file']
        qparameters=self.fconf['res']['qparameters']['file']
        
        #####################################################
        
        
        ################## dataloader #################
        batch_size = num_batch
        
        ##########################################
        
        
        
        
        
        #indices_to_select=torch.flatten(indices_to_select)
        #print(indices_to_select)
        #print(indices_to_select.shape)
        #indices_to_select =torch.tensor([0, 5, 6, 8, 9, 11, 12, 13, 15, 22, 23, 24, 25, 26])
        #print(indices_to_select)
        print('Selected shape:')
        print(data[:,indices_to_select,:].shape)
        
        
    
        
        dataloader = DataLoader(
         data[:,indices_to_select,:],
         batch_size,
         shuffle=shuffle,
        )
        
        
        
        print('Processing...')
        
        qtransform=QT_dataset_custom(tslen, 
                                     sr, 
                                     device=device, 
                                     q=q, 
                                     frange=frange, 
                                     fres=fres, 
                                     tres=tres, 
                                     num_t_bins=num_t_bins, 
                                     num_f_bins=num_f_bins,
                                     logf=logf,
                                     qtile_mode=qtile,
                                     whiten=whiten,
                                     whiteconf=self.whiteconf,
                                     psd=psd,
                                     energy_mode=energy_mode,
                                     phase_mode=phase_mode,
                                     window_param=window_param,
                                     tau=tau,
                                     beta=beta).to(device)
        
  
        
        #qtransform=QT_dataset(6144,1024,device=device,whiten=False,num_t_bins=None,num_f_bins=None).to(device)    
        
        torch.cuda.empty_cache()
        
        qtransform_list=[]
        
        
        #tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)
        for batch in tqdm(dataloader):
             
            with torch.no_grad():
                 transformed= qtransform(batch.to(device=device)).detach().cpu() #torch.Tensor(event.value).to(device)
            print(transformed.shape)   
            qtransform_list.append(transformed[:,:,:,qslicemin:qslicemax])
            del transformed
            torch.cuda.empty_cache()
            gc.collect()
        
        
        stacked_tensor_2d =torch.cat(qtransform_list, dim=0).detach().cpu()
        
        print(stacked_tensor_2d.shape)
        
        
        
        
        #f_range = (8, 410)#take from configuration
        #desired_ticks = [8, 20, 30, 50, 100, 200, 500]
        #log_base = 10  # Or np.e for natural log
        
        timestamp_now=str(int(time.time())*1000)
        
        
        
        where=f'{resdir}/{timestamp}_{tag}_{timestamp_now}'
        
        print(where)
        
        os.mkdir(where)
        
        yaml_data = {
         'metadata': {
            'timestamp': datetime.now().isoformat(),
            'id': timestamp 
          },
          'qt':self.params,
          'whiten':self.whiteconf  
        }
        
        #save_tensor(stacked_tensor_2d,where,qtensor)
        qtf=TFrame(stacked_tensor_2d,rows_list,column_list,gps_list,ann_dict)
        qtf.save(where+"/QT.pt")
        
        print(self.save_gr_par)
        
        
        
        if(self.save_gr_par):
                print('Saving parameters and sample graphs.')
                saveYaml(where,qparameters,yaml_data)
                plot_stacked_tensor(stacked_tensor_2d.unsqueeze(2), 
                           column_list, 
                            frange, 
                            self.desired_ticks, 
                            self.log_base,where,self.phase,
                            self.v_max,self.v_min,self.num_plots,self.length_in_s)
        
         
        
        return 0   