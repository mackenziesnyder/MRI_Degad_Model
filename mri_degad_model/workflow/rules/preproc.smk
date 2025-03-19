# bias correction for gad files
rule n4_bias_correction:
    input:
        im = bids(
            root=str(Path(config["bids_dir"])),
            datatype="anat",
            suffix="T1w.nii.gz",
            **inputs["t1w"].wildcards
        )     
    output:
        corrected_im = bids(
            root=work,
            datatype="bias_correction",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            **inputs["t1w"].wildcards
        )
    script:
        "../scripts/n4_bias_corr.py"

# isotropic resampling
rule isotropic_resampling:
    input:
        input_im = bids(
            root=work,
            datatype="bias_correction",
            desc="n4_bias_corr",
            suffix="T1w.nii.gz",
            **inputs["t1w"].wildcards
        ) 
    params:
        res=resolution
    output:
        resam_im = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            **inputs["t1w"].wildcards
        ) 
    script:
        "../scripts/resample_img.py"

# image registration rigid
rule run_greedy_rigid:
    input:
        nongad_moving = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        gad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        rigid_mat = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad",
            suffix=".mat",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i {input.gad_fixed} {input.nongad_moving} -o {output.rigid_mat} -ia-image-centers -n 100x50x0"

# apply registration rigid
rule apply_registration_rigid:
    input:
        nongad_moving = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        gad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        rigid_mat = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad",
            suffix=".mat",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        registered_nongad = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -dof 12 -rf {input.gad_fixed} -rm {input.nongad_moving} {output.registered_nongad} -r {input.rigid_mat}"

# normalize non gad with registration applied
rule nongad_zscore_normalization:
    input:
        norm_in = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        norm_out = bids(
            root=work,
            datatype="normalize",
            desc="normalized_zscore",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    params: 
        norm_method="zscore"
    script: '../scripts/normalize.py'

# normalize gad
rule gad_minmax_normalization:
    input:
        norm_in = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"} 
        )
    output:
        norm_out = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"} 
        )
    params: norm_method="minmax"
    script: '../scripts/normalize.py'

# create patches for model inputs 
rule create_patches:
    input:
        norm_nongad = bids(
            root=work,
            datatype="normalize",
            desc="normalized_zscore",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        norm_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalize_minmax",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"} 
        )
    params:
        radius_vector = '31x31x31', #patch dimensions will be (x*2)+1 voxels
        n = '5', #sample n randomly augment patches
        angle = '10', #stdev of normal distribution for sampling angle (in degrees)
        frequency = '4500' #sample 1 patch for every n voxels

    output:
        patches = bids(
            root=work,
            datatype="patches",
            desc="samples_31",
            suffix=".dat",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"} 
        )
    resources:    
        mem_mb = 32000    
    container:
        config["singularity"]["itksnap"]
    shell: 'c3d {input.norm_gad} {input.norm_nongad} {input.norm_nongad} -xpa {params.n} {params.angle} -xp {output.patches} {params.radius_vector} {params.frequency}'