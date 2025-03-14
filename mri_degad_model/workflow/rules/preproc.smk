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
        "scripts/n4_bias_corr.py"

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
        "scripts/resample_img.py"

# # image registration rigid
rule run_greedy_rigid:
    input:
        nogad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        gad_moving = bids(
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
            desc="rigid_resliced",
            suffix=".mat",
            **inputs["t1w"].wildcards
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i {input.nogad_fixed} {input.gad_moving} -o {output.rigid_mat} -ia-image-centers -n 100x50x0"

# # image registration affine
rule run_greedy_affine:
    input:
        nongad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        gad_moving = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        affine_mat = bids(
            root=work,
            datatype="greedy",
            desc="affine_resliced",
            suffix=".mat",
            **inputs["t1w"].wildcards
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -a -dof 12 -m NCC 2x2x2 -i {input.nongad_fixed} {input.gad_moving} -o {output.affine_mat} -ia-image-centers -n 100x50x0"

# apply registration rigid
rule apply_registration_rigid:
    input:
        nongad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        rigid_mat = bids(
            root=work,
            datatype="greedy",
            desc="rigid_resliced",
            suffix=".mat",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        rigid_nongad = bids(
            root=work,
            datatype="greedy",
            desc="rigid_mat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -dof 12 -rf {input.nongad_fixed} -rm {input.nongad_fixed} {output.rigid_nongad} -r {input.rigid_mat}"

# apply registration affine
rule apply_registration_affine:
    input:
        nongad_fixed = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        affine_mat = bids(
            root=work,
            datatype="greedy",
            desc="affine_resliced",
            suffix=".mat",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        affine_nongad = bids(
            root=work,
            datatype="greedy",
            desc="affine_mat",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    container: config["singularity"]["itksnap"]
    shell:
        "greedy -d 3 -dof 12 -rf {input.nongad_fixed} -rm {input.nongad_fixed} {output.affine_nongad} -r {input.affine_mat}"

# normalize non gad with registration applied
rule zscore_nongad_rigid:
    input:
        norm_in = bids(
            root=work,
            datatype="greedy",
            suffix="T1w.nii.gz",
            desc="rigid_mat",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        norm_out = bids(
            root=work,
            datatype="normalize",
            desc="normalized_rigid",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    params: 
        norm_method="zscore"
    script: 'scripts/normalize.py'

rule zscore_nongad_affine:
    input:
        norm_in = bids(
            root=work,
            datatype="greedy",
            suffix="T1w.nii.gz",
            desc="affine_mat",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output:
        norm_out = bids(
            root=work,
            datatype="normalize",
            desc="normalized_affine",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    params: 
        norm_method="zscore"
    script: 'scripts/normalize.py'

# normalize gad
rule minmax_gad:
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
            desc="normalized",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"} 
        )
    params: norm_method="minmax"
    script: 'scripts/normalize.py'

rule create_patches_rigid:
    input:
        norm_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalized",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        norm_nongad = bids(
            root=work,
            datatype="normalize",
            desc="normalized_rigid",
            suffix="T1w.nii.gz",
            acq="nongad",
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
            desc="samples_31_rigid",
            suffix=".dat",
            **inputs["t1w"].wildcards
        )
    resources:    
        mem_mb = 32000    
    container:
        config["singularity"]["itksnap"]
    shell: 'c3d {input.norm_gad} {input.norm_nongad} {input.norm_nongad} -xpa {params.n} {params.angle} -xp {output.patches} {params.radius_vector} {params.frequency}'


# rule create patches
rule create_patches_affine:
    input:
        norm_gad = bids(
            root=work,
            datatype="normalize",
            desc="normalized",
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        norm_nongad = bids(
            root=work,
            datatype="normalize",
            desc="normalized_affine",
            suffix="T1w.nii.gz",
            acq="nongad",
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
            desc="samples_31_affine",
            suffix=".dat",
            **inputs["t1w"].wildcards
        )
    resources:    
        mem_mb = 32000    
    container:
        config["singularity"]["itksnap"]
    shell: 'c3d {input.norm_gad} {input.norm_nongad} {input.norm_nongad} -xpa {params.n} {params.angle} -xp {output.patches} {params.radius_vector} {params.frequency}'
