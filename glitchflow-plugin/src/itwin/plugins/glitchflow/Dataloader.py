import torch
from torch.utils.data import random_split
if torch.cuda.is_available():
    device = 'cuda'
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.version.cuda)
else:
    device = 'cpu'
print(f'{device=}')

import numpy as np

import itwinai
from itwinai.components import DataGetter,DataSplitter,DataProcessor,  monitor_exec

from typing import List, Optional, Tuple
import os
from . Data import (construct_tensor_data,
                    TFrame,merge_tframes,
                    save_tensor,read_tensor_lib,
                    augment_data,augment_dataset,
                   filter_rows_below_threshold,
                    filter_rows_above_threshold,
                   find_max,
                   normalize_ch_mean)


class DtsToTensor(DataGetter):
    def __init__(self, datalist: List,logger: itwinai.loggers.TensorBoardLogger,conf:Optional[List] = None) -> None:
        
        self.logger=logger
        self.datalist=datalist
        
        
        

    @monitor_exec
    def execute(self) -> TFrame:
         
        self.logger.log('Reading data...','MESSAGE',kind='text')
        
        tf_list=[]
        
        for dataset in self.datalist['datasets']:
            
            
            
            path=dataset['path']
            target=dataset['target']
            ch_list=dataset['ch_list']
            
            channel1=dataset['channels']['min']
            channel2=dataset['channels']['max']
            
            event1=dataset['events']['min']
            event2=dataset['events']['max']
            
            f1=dataset['processing']['minf']
            f2=dataset['processing']['maxf']
            sr=dataset['processing']['sr']
            tslen=dataset['processing']['len']
            batchs=dataset['processing']['batch']
            whiten=dataset['processing']['whiten']
            
            
            print(f"Reading{path}/{target}")
            
            df,ids,chans=construct_tensor_data(path=path, 
                           channel_list=ch_list, 
                           target_channel=target, 
                           n1_events=event1, 
                           n2_events=event2, 
                           n1_channels=channel1, 
                           n2_channels=channel2, 
                           print_=True, sr=sr,crp=tslen,batch=batchs,type='chb1',f1=f1,f2=f2)
            
            tf=TFrame(df,ids,chans,{"sample_rate":sr})
            
            tf_list.append(tf)
            
            
            
            
            

            
        #print(tf_list[0].tdata.shape,tf_list[1].tdata.shape)
        tfm=merge_tframes(tf_list)
        
        
            
            
       
        return tfm
    
    


class ReadTensor(DataGetter):
    def __init__(self, tpath: str,logger: itwinai.loggers.TensorBoardLogger) -> None:
        
        self.path=tpath
        self.logger=logger
        
        
        
        

    @monitor_exec
    def execute(self) -> List:
         
        self.logger.log('Reading data from tensor','MESSAGE',kind='text')
        
        additional_list=[]
        output=[]
        
        error,dataframe,additional_list=read_tensor_lib(self.path)
        
        output.extend([error,dataframe,additional_list])
            
        #if error is true I should log it then the DAG should fail, but how?    
        
        
        return output
    
    
#-----------------------------------------Classes for training-----------------------------------------------------------------------------------#    
class QTDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        logger: itwinai.loggers.TensorBoardLogger,
        validation_proportion: int | float = 0.0,
        
        
        rnd_seed: Optional[int] = 42,
        
        name: Optional[str] = None,
        images_dataset: Optional[str] = None,
        
    ) -> None:
        """Class for splitting of smaller datasets. Use this class in the pipeline if the
        entire dataset can fit into memory.

        Args:
            train_proportion (int | float): _description_
            validation_proportion (int | float, optional): _description_. Defaults to 0.0.
            test_proportion (int | float, optional): _description_. Defaults to 0.0.
            rnd_seed (Optional[int], optional): _description_. Defaults to None.
            images_dataset (str, optional): _description_.
                Defaults to "data/Image_dataset_synthetic_64x64.pkl".
            name (Optional[str], optional): _description_. Defaults to None.
        """
        test=1 - train_proportion
        super().__init__(train_proportion,validation_proportion,test , name)
        self.save_parameters(**self.locals2params(locals()))
        #self.validation_proportion =validation_propo 
        self.rnd_seed = rnd_seed
        self.images_dataset = images_dataset
        self.logger=logger
       
        

    def get_or_load(self, dataset: Optional[torch.Tensor] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            #LOG
            print("WARNING: QT dataset from disk.")
            return torch.load(self.images_dataset,weights_only=True)
        return dataset

    @monitor_exec
    def execute(self, dataset: Optional[torch.Tensor] = None) -> List:
        """Splits a dataset into train, validation and test splits.

        Args:
            dataset (pd.DataFrame): input dataset.

        Returns:
            Tuple: tuple of train, validation and test splits. Test is None.
        """
        dataset = self.get_or_load(dataset)
        print('Read dataset: ',dataset.shape)
        
        
        torch.manual_seed( self.rnd_seed)  # Choose any integer as the seed
        data=dataset
        num_aux_channels=data.shape[1]-1
        '''
        # Specify indices of interest along the second dimension
        aux_indices = torch.tensor([2, 3, 4, 6, 7, 8, 16, 17, 19, 20])
        num_aux_channels=aux_indices.shape[0]
        # Select specific auxiliary channels
        data= loaded_tensor[:, torch.cat([torch.tensor([0]) ,aux_indices],dim=0), :, :]
        '''

        # Set split sizes: 90% for training, 10% for testing
        train_size = int(self.train_proportion * len(data))
        test_size = len(data) - train_size

        # Perform the train-test split with the fixed seed
        train_data_list, test_data_list = random_split(data, [train_size, test_size])


       # Convert the Subset objects back to tensors
        train_data = torch.stack([data[idx] for idx in train_data_list.indices])
        test_data = torch.stack([data[idx] for idx in test_data_list.indices])


       # Check the final concatenated shapes
        print(f'{train_data.shape=}\n{test_data.shape=}')

        
        
        return [train_data, test_data,num_aux_channels]
    
    
class QTProcessor(DataProcessor):
    def __init__(self,logger: itwinai.loggers.TensorBoardLogger, name: str | None = None,maxclamp: int | float = 10000) -> None:
       
        """
        Args:
            name (str | None, optional): Defaults to None.
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.logger=logger
        self.max_value=maxclamp

    @monitor_exec
    def execute(self, dataset:List) -> List:
        
        """Pre-process datasets: rearrange and normalize before training.

        Args:
            train_dataset (Tuple): training dataset.
            validation_dataset (Tuple): validation dataset.
            test_dataset (Any, optional): unused placeholder. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, None]: train, validation, and
                test (placeholder) datasets. Ready to be used for training.
        """
        
       #AUGMENTATION
    
        num_aux_channels=dataset[2]
        
        train_data_2d,test_data_2d=augment_dataset(dataset[0],dataset[1])
        
        print('Augmented dataset:')
        print(f'{train_data_2d.shape=}\n{test_data_2d.shape=}')
        
        del dataset
        
        
        #CLAMPING
        
        print('Clamp dataset:')
        
        
        train_data_2d_clamp=torch.clamp(train_data_2d, min=0,max=self.max_value)
        test_data_2d_clamp=torch.clamp(test_data_2d, min=0,max=self.max_value)
        try:
            background_tensor_clamp=torch.clamp(background_tensor, min=0,max=max_value)
        except:
            print('No background tensor')
            
            
        #train_data_2d_norm/=max_value
        #test_data_2d_norm/=max_value   
            
        
        #FILTERING
        """
        print('Filtering dataset below treshold:')
        
        #hardcoded
        filtered_data_train_2d_below,mask_train=filter_rows_below_threshold(train_data_2d_clamp,
                                                torch.tensor([6,self.max_value,
                                                            self.max_value,
                                                            self.max_value,
                                                            self.max_value,
                                                            self.max_value,
                                                            self.max_value,
                                                            self.max_value,
                                                            self.max_value]))

        #hardcoded
        filtered_data_test_2d_below,mask_test=filter_rows_below_threshold(test_data_2d_clamp,
                                              torch.tensor([6,self.max_value,
                                              self.max_value,
                                              self.max_value,
                                              self.max_value,
                                              self.max_value,
                                              self.max_value,
                                              self.max_value,self.max_value]))     
            
         
        print('-Train')
        print(filtered_data_train_2d_below.shape)
        print('-Test')
        print(filtered_data_test_2d_below.shape)
        
        print('-Background')
        
        background=torch.cat((filtered_data_train_2d_below,filtered_data_test_2d_below))
        print(background.shape)
         
        
        
        print('Filtering dataset above treshold:')
        
        
        filtered_data_train_2d,mask_train_above=filter_rows_above_threshold(train_data_2d_clamp,torch.tensor([10,0,0,0,0,0,0,0,0]))
        filtered_data_test_2d, mask_test_above=filter_rows_above_threshold(test_data_2d_clamp,torch.tensor([10,0,0,0,0,0,0,0,0]))
        
        print('-Train')
        print(filtered_data_train_2d.shape)
        print('-Test')
        print(filtered_data_test_2d.shape)
        """
        
        #STAST AND NORMALIZATION
        print('Normalize and print statistics')
        #Unfiltered data
        max_train = find_max(train_data_2d_clamp)
        max_test = find_max(test_data_2d_clamp)
        background=torch.cat((train_data_2d_clamp,test_data_2d_clamp))

        #Filtered data
        #max_train = find_max(filtered_data_train_2d)
        #max_test = find_max(filtered_data_test_2d)


        # Flatten the tensor along the channel dimension
        flattened_tensor = max_train.view(-1, num_aux_channels+1)
        flattened_tensor_test = max_test.view(-1, num_aux_channels+1)

        # Convert tensor to numpy array
        numpy_array = flattened_tensor.numpy()
        numpy_array_test= flattened_tensor_test.numpy()
        
        channel_means = np.mean(numpy_array, axis=0)
        channel_means_test = np.mean(numpy_array_test, axis=0)
        channel_std = np.std(numpy_array, axis=0)
        channel_std_test = np.std(numpy_array_test, axis=0)
        
        print('\n\nTRAIN')
        for i, mean in enumerate(channel_means):
            print(f'Average of Channel {i+1} train: {mean}')
            print(f'std of Channel {i+1} train: {channel_std[i]}')
            print(f'-----------------------------------------')
        print('\n\n TEST')   
        for i, mean in enumerate(channel_means_test):
            print(f'Average of Channel {i+1} test: {mean}')
            print(f'STD of Channel {i+1} train: {channel_std_test[i]}')
            print(f'-----------------------------------------')
            
        norm_factor=torch.tensor(channel_means[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        
        train_data_2d_norm=normalize_ch_mean(filtered_data_train_2d,channel_means) #,channel_std
        #,channel_means,channel_std # not channel_means_test, it should be the same as train data
        test_data_2d_norm=normalize_ch_mean(filtered_data_test_2d,channel_means)
        #,channel_means,channel_std # not channel_means_test, it should be the same as train data
        
        background_norm=normalize_ch_mean(background,channel_means)  
        

        return {train_data_2d_norm,test_data_2d_norm,background_norm, norm_factor,num_aux_channels}   
    
    
    

    
