## preprocess.py parameters description
 - `--imgs` : dataset basepath (or image folder) that  you want to preprocess
 - `--processes` : number of individual processes that will run in parallel

## backbone_head_combo_trainer.py parameters description
 - `--data_root_train` : dataset basepath
 - `--train_file` : path to the .txt file containing all images paths and classes (see .txt files in <a href="https://drive.google.com/drive/folders/1LNLOO157NPUOAfCWDtMxJpmDVkbn8eu4?usp=sharing" target="_blank">Drive Folder</a>)
 - `--outfolder` : path to the folder where you want to save your trained models
 - 
NOTE: you have to manually edit the following variables in order to perform your trainings:
- backbone_conf_file
- head_conf_file
- backbones
- heads
