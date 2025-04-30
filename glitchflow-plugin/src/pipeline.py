import yaml
import os

######delete when implementing the plugin###
import sys
sys.path.append('./itwin/plugins/')
###########################################

from itwinai.pipeline import Pipeline
from itwinai.loggers import TensorBoardLogger


######modify path when implementing the plugin#### DATALOADER
from glitchflow.Dataloader import DtsToTensor ,ReadTensor, QTDatasetSplitter, QTProcessor
#############################################

######modify path when implementing the plugin#### SCAN
from glitchflow.Scanner import AnnalisaScan, Preprocess 
#############################################


######transform into env variables##########
runpath='./conf/annalisarun.yaml'
dtspath='./conf/datasets.yaml'
scanpath='./conf/scan.yaml'
processpath='./conf/process.yaml'
tpath='./datasets/target'
############################################

######train param##########################
QTpath='./processed/QT_small.pt'
seed=42
tprop=0.9

##########################################

logger=TensorBoardLogger('./logs',)
logger.create_logger_context()

###put a flag for fylesystem error

try:
    with open(dtspath, 'r') as file:
        datasets = yaml.safe_load(file)
except Exception as e:
        print(e)
        logger.log(str(e),'ERRF',kind='text')
    
try:
    with open(runpath, 'r') as file:
        runf = yaml.safe_load(file)
except Exception as e:
        print(e)
        logger.log(str(e),'ERRF',kind='text')    

        
try:
    with open(scanpath, 'r') as file:
        scan = yaml.safe_load(file)
except Exception as e:
        print(e)
        logger.log(str(e),'ERRF',kind='text') 
        
try:
    with open(processpath, 'r') as file:
        process = yaml.safe_load(file)
except Exception as e:
        print(e)
        logger.log(str(e),'ERRF',kind='text')         
        
        
        
#['datasets'][0]['channels'][min]    
#for dataset in datasets['datasets']:


pipeline = Pipeline([
    DtsToTensor(datalist=datasets,logger=logger,conf=runf),
    AnnalisaScan(params=scan,logger=logger,conf=runf,save=True)
   
    
])



pipelinetrain=Pipeline([ReadTensor(tpath=tpath,logger=logger),Preprocess(params=process,logger=logger,conf=runf,save=True)])

pipeglitch=Pipeline([QTDatasetSplitter(train_proportion=tprop,logger=logger,rnd_seed=seed,images_dataset=QTpath),
                     QTProcessor(logger=logger)])
    
#trained_model = pipeline.execute()

trained_model = pipeglitch.execute()

logger.destroy_logger_context()        