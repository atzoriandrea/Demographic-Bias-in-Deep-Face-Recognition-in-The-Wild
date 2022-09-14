# get_discriminators_scores.py parameters description
- `--datarootHR` : root folder for your selected High Resolution Dataset (the folder containing all the sub-folders)
 - `--datarootGEN` : root folder for your selected Low Resolution (degraded) Dataset (the folder containing all the sub-folders)
 - `--h2l` : path of your trained H2L model (contains both generator and discriminator)
 - `--out` : output path 

# compute_distributions_similarity.py parameters description
- `--dataroot` : Path where all scores obtained with the discriminator were saved (divide into folders-one per dataset-to automate analysis)

# plot_discriminators_scores.py parameters description
- `--files_basepath` : Path where all scores obtained with the discriminator were saved (divide into folders-one per dataset-to automate analysis)
 - `-qmul_results` : root folder of your real Low-Resolution dataset (the reference one)