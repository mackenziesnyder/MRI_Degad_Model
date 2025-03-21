# MRI_Degad_Model

CNN model for the MRI Degad program

**Preprocessing Command (Bids):**

```
mri_degad_preprocessing {input_dir} {output_dir} participant --use-singularity --cores all 
```

**Model Training Command**

```
cd mri_degad_cnn
```

```
snakemake --config input_dir={output dir of preprocessing} output_dir={your chosen output dir} subject_file={path to subject file}
```

to customize model parameters, add ```{designated model parameter}={changed value}```

*defaults:*

- patch_size: 32
- batch_size: 256
- learning_rate: 0.0005
- initial_filter: 512
- depths: [3, 4]
- num_convolution: 2
- loss: "mae"
- train_ratio: 0.8
- val_ratio: 0.2

*Example with customized model parameters:*

```
snakemake --config input_dir={output dir of preprocessing} output_dir={your chosen output dir} subject_file={path to subject file} num_convolution=3 
```

**Important Note:**

To test the model you create through this program, please use data the model was not trained on. If you are running this model training end-to-end, please seperate your data before hand into model data and test data. A recommended  split is 85% model data and 15% test data. Once split, this code will take care of splitting the train and validation data. 

To test newly created models, please upload the model to OSF, switch the model link in https://github.com/mackenziesnyder/MRI-DeGad, and run the program in MRI-Degad. 