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



class AnnalisaScan(Trainer):
    def __init__(self,params: YamlWrapper,
                 conf:YamlWrapper,
                 logger: None=None,
                 source:str=''
                 ) -> None:
        
        self.logger=logger
        self.params=params.flist
        self.fconf=conf.flist
        self.source=source
        
        """
        params: wrapper containing a list with scan.yaml parameters
        conf: wrapper containing a list with saveconf.yaml parameters
        source: path to dataset
        Use this option for a standalone execution or a pipeline processing of an existing dataset

        return a TFrame containing the scanned dataset with channels saved as metadata.

        All results will be saved depending on the saveconf.yaml file
        
        """
        
        
        
        
        
    def get_or_load(self, dataset: Optional[TFrame] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            print("WARNING: loading time series dataset from disk.")
            
            tf=TFrame(0,[],[],[],{})
            
            tf.load(self.source)
            
            
            
            return tf
        
        return dataset    
        
        

    @monitor_exec
    def execute(self,pipe_set:Optional[TFrame]=None) -> TFrame:
        
        
        
        
        
        ##############Annalisa parameters ############
        
        proc_set=self.get_or_load(pipe_set)
        data=proc_set.tdata
        tslen=proc_set.tdata.shape[2]
        print(f'Found tensor of shape({proc_set.tdata.shape[0]},{proc_set.tdata.shape[1]},{tslen})')
        sr=proc_set.ann_dict["sample_rate"]
        timestamp=proc_set.ann_dict["id"]
        tag=proc_set.ann_dict["tag"]
        
        
        num_batch=self.params['dataset_scan']['parameters']['num_batch']
        num_chan=self.params['dataset_scan']['parameters']['num_chan']
        refchan=self.params['dataset_scan']['parameters']['refchan']
        threshold=self.params['dataset_scan']['parameters']['threshold']
        
        time_window=self.params['dataset_scan']['parameters']['time_window']
        time_only_mode=self.params['dataset_scan']['parameters']['time_only_mode']
        tolerance_distance=self.params['dataset_scan']['parameters']['tolerance_distance']
        q=self.params['dataset_scan']['parameters']['q']
        frange=self.params['dataset_scan']['parameters']['frange']
        
        fres=self.params['dataset_scan']['parameters']['fres']
        tres=self.params['dataset_scan']['parameters']['tres']
        num_t_bins=self.params['dataset_scan']['parameters']['num_t_bins']
        num_f_bins=self.params['dataset_scan']['parameters']['num_f_bins']
        logf=self.params['dataset_scan']['parameters']['logf']
        qtile=self.params['dataset_scan']['parameters']['qtile']
        whiten=self.params['dataset_scan']['parameters']['whiten']
        
        threshold_corr=self.params['dataset_scan']['parameters']['threshold_corr']
        threshold_iou=self.params['dataset_scan']['parameters']['threshold_iou']
        
        ############################################################################
        
        ################# results######################
        resdir=self.fconf['params']['resdir']
        graphcorr=self.fconf['res']['graphcorr']['file']
        graphiou=self.fconf['res']['graphiou']['file']
        indices=self.fconf['res']['indices']['file']
        annalisaotp=self.fconf['res']['annalisaotp']['file']
        #####################################################
        
        
        ################## dataloader #################
        event_batch_size = num_batch
        channel_batch_size =num_chan
        reference=proc_set.col_list.index(refchan)
        ##########################################
        
        
        
        
        
        
        
        
        
        strain_dataloader = DataLoader(
        data[:,reference,:],
        batch_size=event_batch_size,
        )
        
        
        
       
        aux_dataloader = NestedMultiDimBatchDataLoader(data[:,:,:], event_batch_size, channel_batch_size)
        print('Scanning dataset...')
        
        
        
        annalisa_scan=Annalisa(tslen, sr, device=device, threshold=threshold, time_window=time_window, time_only_mode=time_only_mode,
                 tolerance_distance= tolerance_distance, q=q, frange=frange, fres=fres, tres=tres, num_t_bins=num_t_bins, num_f_bins=num_f_bins,
                 logf=logf, qtile_mode=qtile,whiten=whiten).to(device)
        
        
        
        stacked_corr_coeffs = []
        stacked_iou_coeffs = []

        for batch_s, batch_a in tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)):
            
            iou_coeff_batch,corr_coeff_batch=annalisa_scan(batch_s, batch_a)
            stacked_iou_coeffs.append(iou_coeff_batch)  # Append corr_coeff to the list 
            stacked_corr_coeffs.append(corr_coeff_batch)  # Append corr_coeff to the list  
            
        corr_coeff_dataset=torch.cat(stacked_corr_coeffs, dim=0)
        iou_coeff_dataset=torch.cat(stacked_iou_coeffs, dim=0)
        
        
        #####saving results##############

        column_list=proc_set.col_list
        row_list=proc_set.row_list
        
        timestamp_now=str(int(time.time())*1000)
        
        where=f'{resdir}/{timestamp}_{tag}_{timestamp_now}'
        
        print(where)
        
        os.mkdir(where)
        

        
        corrmean=torch.mean(corr_coeff_dataset,axis=0)
        ioumean=torch.mean(iou_coeff_dataset,axis=0)
        
        
        
        plot_to_f(column_list, 
                            corrmean, 
                            where,
                            graphcorr,
                            title='Correlation histogram vs Hrec')
        
        plot_to_f(column_list, 
                  ioumean,
                  where,
                  graphiou,
                  title='IOU histogram vs Hrec')
        
        
       
        
        
       
        
        mask=(torch.mean(corr_coeff_dataset,axis=0)>threshold_corr) & (torch.mean(iou_coeff_dataset,axis=0)>threshold_iou)
        indices_to_select = torch.nonzero(mask)
        indices_to_select=indices_to_select.flatten()
        
        
        proc_set.addmeta('chans',indices_to_select.tolist())
        print(indices_to_select)

        selected_elements = [column_list[i] for i in indices_to_select]

        print(selected_elements)
        
        save_tensor(indices_to_select,where,indices)
        
        yaml_data = {
         'metadata': {
            'timestamp': datetime.now().isoformat(),
            'id': timestamp,
            'tag': tag 
          },
          'configuration': {
            'parameters':self.params, 
              
            'channels': column_list
          },
          'results': {
            'indeces':indices_to_select.tolist(),  
            'correlations':corrmean.tolist(),
            'iou':ioumean.tolist(),
            'selected': selected_elements    
          }
        }
        
        saveYaml(where,annalisaotp,yaml_data)
                
                
        
        
        
        
        
        
        return proc_set
    
    
    

    
    
    
    
