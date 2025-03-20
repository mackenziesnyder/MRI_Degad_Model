rule run_preprocessing_qc:
    input: 
        gad_resampled = bids(
            root=work,
            datatype="resampled",
            desc="resampled",
            res=config["res"],
            suffix="T1w.nii.gz",
            acq="gad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        registered_nongad_rigid = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad_rigid",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        ),
        registered_nongad_affine = bids(
            root=work,
            datatype="greedy",
            desc="nongad_to_gad_affine",
            suffix="T1w.nii.gz",
            acq="nongad",
            **{k: v for k, v in inputs["t1w"].wildcards.items() if k != "acq"}
        )
    output: 
        html=bids(
            root=root,
            datatype="prepocessing_qc",
            desc="qc",
            suffix=".html",
            **inputs["t1w"].wildcards
        )
    script: '../scripts/registration_qc.py