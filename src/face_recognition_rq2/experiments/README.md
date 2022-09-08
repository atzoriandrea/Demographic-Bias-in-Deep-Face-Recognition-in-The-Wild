# Scripts description

## plot_TSNE_distribution.py parameters description
 This script plots your data in order to obtain a menagerie-related overview 
 - `--jsonsf` : path to the json file containing all annotations for your dataset
 - `--results` : path to .npy file containing the results obtained by pairs comparison
 - `--csv` : path of the .csv file in which all image pairs are listed

## features_merge.py 
This file contains a library function that returns a Pandas Dataframe with image embeddings and protected attributes for each image


## get_metrics_across_groups.py
This script returns (in latex format), a cross sensitive-attribute analisys on given data (see FAR and FRR tables in main page)
### Parameters
 - `--jsonsf` : path to the json file containing all annotations for your dataset
 - `--results` : path to .npy file containing the results obtained by pairs comparison
 - `--csv` : path of the .csv file in which all image pairs are listed
 - `--save` : (optional) if True, the table will be saved as .txt file. Otherwise, it will be printed.

## get_info_on_results.py
 This script plots, for each run (you MUST give as input a list of results belonging to the same dataset): 
- The correlation value(s) between attributes and metrics like FAR (False Acceptance Rate) and FRR (False Rejection Rate) (see last figure on main page)
- Boxplots exposing the R2 scores of linear regressors trained on metrics such as FAR and FRR 
- Heatmaps exposing the weights of the individual characteristics on linear regressors trained on metrics such as FAR and FRR 

### Parameters
 - `--jsonsf` : path to the json file containing all annotations for your dataset
 - `--results_list` : path to the txt file containing results files paths (the .npy ones)
 - `--csv` : path of the .csv file in which all image pairs are listed
 - `--suffix` : (optional) suffix string in order to divide different runs of this script (default value: timestamp)
 - `--iterations` : iterations for samples picking in linear regressor training step
 - `--samples` : number of samples to pick at each iteration of linear regression training

<br>
<div align="center">
 <img src="../../images/boxplot.png" height="250" width="250" alt="Boxplots exposing the R2 scores of linear regressors trained on metrics such as FAR and FRR "/> 
 <img src="../../images/heatmap.png" height="250" width="250" alt="Heatmaps exposing the weights of the individual characteristics on linear regressors trained on metrics such as FAR and FRR "/>
</div>
<br>