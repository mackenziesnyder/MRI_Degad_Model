# MRI_Degad_Model

CNN model for the MRI Degad program

**Preprocessing Command (Bids):**

```
mri_degad_preprocessing {input_dir} {output_dir} participant --use-singularity --cores all 
```

**Model Training Command**

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
- train_ratio: 0.7
- val_ratio: 0.15
- test_ratio: 0.15

*Example with customized model parameters:*

```
snakemake --config input_dir={output dir of preprocessing} output_dir={your chosen output dir} subject_file={path to subject file} num_convolution=3 
```