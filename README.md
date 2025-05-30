# MRI_Degad_Model

Model for the MRI Degad program
To learn more about the MRI Degad program, please visit the workflow repository: https://github.com/mackenziesnyder/MRI-DeGad

**Preprocessing:**

The preprocessing pipeline utilizes snakemake with snakebids. In order for the preprocessing pipeline to work, the input data must be in standard bids formatting.

Mri degad preprocessing python package dependencies are managed with Poetry, which you’ll need installed on your machine. You can find instructions on the poetry website: https://python-poetry.org/docs/.

Setting up the environment:

```
git clone https://github.com/mackenziesnyder/MRI_Degad_Model.git
cd mri_degad_preprocessing
poetry shell 
poetry install 
```

To run the preprocessing pipeline, run the following command: 
```
mri_degad_preprocessing {input_dir} {output_dir} participant --cores all 
```

**Model Training**

This model was trained using resources provided by ComputeCanada with SLURM.
To train the model with SLURM on ComputeCanada, ensure the cloned repository is located on graham, ssh into graham, and run the following commands:

```
cd mri_degad_model
module load python/3.11.5
```

Create a virtual environment and install packages
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

Run the following bash script, if you want to change any of the model parameters or resources, that can all be done in the designated run script. 

run scripts to choose from:
- *run_CNN.sh* - trains on whole images
- *run_CNN_patches.sh* - trains on patches of whole images
- *run_GAN.sh* - trains on whole images
- *run_GAN_patches.sh* - trains on patches of whole images


```
sbatch {desired run script}
```

**Test new models with the MRI-Degad Program:**


To test newly created models, please upload the model to OSF, switch the model link in https://github.com/mackenziesnyder/MRI-DeGad, and run the program in MRI-Degad. 