import torch
if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'
    
    
from torch.utils.data import DataLoader
import numpy as np


import itwinai
from itwinai.components import Trainer, monitor_exec

from typing import List 
#another possible types: Tuple Optional

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import time
from datetime import datetime
import yaml


from . Data import TFrame

from . Data import save_tensor, save_to_json

from . Annalisa_gpu import * 



class AnnalisaScan(Trainer):
    def __init__(self,params: List,logger: itwinai.loggers.TensorBoardLogger,conf:List,save:Optional[bool] = None) -> None:
        
        self.logger=logger
        self.params=params
        self.fconf=conf
        self.save=save
        
        
        

    @monitor_exec
    def execute(self,proc_set:TFrame) -> int:
        
        
        ##############Annalisa parameters space############
        
        data=proc_set.tdata
        tslen=proc_set.tdata.shape[2]
        print(f'Found tensor of shape({proc_set.tdata.shape[0]},{proc_set.tdata.shape[1]},{tslen})')
        sr=proc_set.ann_dict["sample_rate"]
        
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
        
        torch.cuda.empty_cache()
        
        stacked_corr_coeffs = []
        stacked_iou_coeffs = []

        for batch_s, batch_a in tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)):
            
            
            
                    
            #print('###############################################################')
            ######add bar with tqdm
            #print(f'{iter=}')
            iou_coeff_batch,corr_coeff_batch=annalisa_scan(batch_s, batch_a)
            stacked_iou_coeffs.append(iou_coeff_batch)  # Append corr_coeff to the list 
            stacked_corr_coeffs.append(corr_coeff_batch)  # Append corr_coeff to the list  
            #iter+=1
        corr_coeff_dataset=torch.cat(stacked_corr_coeffs, dim=0)
        iou_coeff_dataset=torch.cat(stacked_iou_coeffs, dim=0)
        
        
        #####saving results##############

        column_list=proc_set.col_list
        row_list=proc_set.row_list
        metadata=proc_set.ann_dict
        
        timestamp=str(int(time.time())*1000)
        where=f'{resdir}/{timestamp}'
        os.makedirs(where)
        
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
        print(indices_to_select)

        selected_elements = [column_list[i] for i in indices_to_select]

        print(selected_elements)
        
        save_tensor(indices_to_select,where,indices)
        
        yaml_data = {
         'metadata': {
            'timestamp': datetime.now().isoformat(),
            'id': timestamp
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
        
        filepath = os.path.join(where, 'results.yaml')
        with open(filepath, 'w', encoding='utf-8') as f:
             yaml.dump(
                 yaml_data,
                 f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=80,
                indent=4
             )
        
        if(self.save):
            
            
            yaml_data = {
            'tensors': {
              'additional': ['index.pt']
            
              }
          
            }
            
            
            dtsdir=self.fconf['res']['dtsmerged']['dir']
           
            where=f'{dtsdir}/{timestamp}'
            
            os.makedirs(where)
            save_tensor(data,where,'tensor')
            save_tensor(indices_to_select,where,'index')
            save_to_json(column_list,'cols',where)
            save_to_json(row_list,'rows',where)
            save_to_json(metadata,'meta',where)
            
            filepath = os.path.join(where, 'manifest.yaml')
            with open(filepath, 'w') as f:
                yaml.dump(yaml_data,f)
       
        
        
        
        
        
        
        return 1
    
    
    
class Preprocess(Trainer):
    def __init__(self,params: List,logger: itwinai.loggers.TensorBoardLogger,conf:List,save:Optional[bool] = None) -> None:
        
        self.logger=logger
        self.params=params
        self.fconf=conf
        self.save=save
        
        
        

    @monitor_exec
    def execute(self,proc_set:List) -> int:
        
        
        ##############Processing parameters space############
        
        data=proc_set[1].tdata
        column_list=proc_set[1].col_list
        
        tslen=data.shape[2]
        print(f'Found tensor of shape({data.shape[0]},{data.shape[1]},{tslen})')
        indices_to_select=proc_set[2][0]
        #print(data)
        sr=proc_set[1].ann_dict["sample_rate"]
        #print(sr)
        
        
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
        
        
        
        
        ############################################################################
        
        ################# results######################
        resdir=self.fconf['res']['processed']
        
        #####################################################
        
        
        ################## dataloader #################
        batch_size = num_batch
        
        ##########################################
        
        indices_to_select=torch.flatten(indices_to_select)
        print(indices_to_select)
        print(indices_to_select.shape)
        print(data[:,indices_to_select,:].shape)
    
        
        dataloader = DataLoader(
         data[:,indices_to_select,:],
         batch_size,
         shuffle=shuffle,
        )
        
        
        
        print('Processing...')
        
        qtransform=QT_dataset(tslen, sr, device=device, q=q, frange=frange, fres=fres, tres=tres, num_t_bins=num_t_bins, num_f_bins=num_f_bins,
                 logf=logf, qtile_mode=qtile,whiten=whiten,psd=psd).to(device)
          
        
        #qtransform=QT_dataset(6144,1024,device=device,whiten=False,num_t_bins=None,num_f_bins=None).to(device)    
        
        torch.cuda.empty_cache()
        
        qtransform_list=[]
        

       #tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)
        for batch in tqdm(dataloader):
             
        
             transformed= qtransform(batch.to(device=device)).detach().cpu() #torch.Tensor(event.value).to(device)
             #print(f'{transformed.shape=}')
    
             #print(f'Torch transform time: {etorch-storch}s')
             qtransform_list.append(transformed[:,:,:,qslicemin:qslicemax])
             torch.cuda.empty_cache()
        
        
        stacked_tensor_2d =torch.cat(qtransform_list, dim=0).detach().cpu() 
        print(stacked_tensor_2d.shape)
        
        
        f_range = (8, 410)#take from configuration
        desired_ticks = [8, 20, 30, 50, 100, 200, 500]
        log_base = 10  # Or np.e for natural log
        
        for idx in range(10):
        
            channel_idx = idx
        

            for i in range(stacked_tensor_2d.shape[1]):
                fig, axes = plt.subplots(1, 1, figsize=(20, 5))
                qplt_strain = torch.flipud(stacked_tensor_2d[channel_idx, i, :, :].detach().cpu())
                im = axes.imshow(qplt_strain, aspect='auto', vmin=0, vmax=25)
                axes.set_title(column_list[channel_idx])
                axes.set_xlabel('Time (s)')
                axes.set_ylabel('Frequency (Hz)')

                new_height = stacked_tensor_2d.shape[2]

                # 1. Log-transform the frequency range and desired ticks
                log_f_range = (np.log(f_range[0]) / np.log(log_base), np.log(f_range[1]) / np.log(log_base))
                log_desired_ticks = np.log(desired_ticks) / np.log(log_base)

                # 2. Map the LOG-TRANSFORMED TICKS to pixel values (REVERSED PIXEL RANGE)
                y_ticks_pixel = np.interp(log_desired_ticks, log_f_range, [new_height - 1, 0])

                y_ticks_pixel = [int(p) for p in y_ticks_pixel]
                y_ticks_pixel = np.clip(y_ticks_pixel, 0, new_height - 1)  # Keep in bounds

                y_ticks_pixel, unique_indices = np.unique(y_ticks_pixel, return_index=True)
                desired_ticks = np.array(desired_ticks)[unique_indices].tolist()

                axes.grid(True, axis='y', which='both')
                axes.set_yticks(y_ticks_pixel)
                axes.set_yticklabels(desired_ticks)
                axes.set_xticks([0,127,256])
                axes.set_xticklabels([0,2,4])

                fig.colorbar(im, ax=axes)
                plt.savefig(f'spectrogram{idx}.png')
        
         
        
        return 1    