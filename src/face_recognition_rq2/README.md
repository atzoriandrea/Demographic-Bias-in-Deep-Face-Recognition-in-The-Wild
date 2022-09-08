### model_tester.py parameters description
 - `--model` : path to a single model that you want to evaluate
 - `--models` : path to a .txt file containing full paths to all models you want to evaluate
 - `--cmps` : path of the .csv file in which all couples are listed
 - `--base` : path to dataset basepath

NOTE: this script returns also the inferenced cosines and other useful data for each given image pair


### preprocess.py parameters description
 - `--imgs` : dataset basepath (or image folder) that  you want to preprocess
 - `--processes` : number of individual processes that will run in parallel

### backbone_head_combo_trainer.py parameters description
 - `--data_root_train` : dataset basepath
 - `--train_file` : path to the .txt file containing all images paths and classes (see .txt files in Drive folder)
 - `--outfolder` : path to the folder where you want to save your trained models
 - 
NOTE: you have to manually edit the following variables in order to perform your trainings:
- backbone_conf_file
- head_conf_file
- backbones
- heads
