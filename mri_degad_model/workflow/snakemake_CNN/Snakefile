PATCH_SIZES = [32]
BATCH_SIZES = [256]
LEARNING_RATES = [0.0005]
INITIAL_FILTER = [512]
DEPTHS = [3,4]
NUM_CONVS = [2]
LOSS = ['mae']
TMP_DIR =  os.environ.get("SLURM_TMPDIR", "/tmp")

rule all:
    input:
        expand("output/patch-{p_size}_batch-{b_size}_LR-{lr}_filter-{filter}_depth-{depth}_convs-{convs}_loss-{loss}/model_log.txt", p_size=PATCH_SIZES, b_size=BATCH_SIZES, lr=LEARNING_RATES, filter=INITIAL_FILTER,depth=DEPTHS,convs=NUM_CONVS, loss= LOSS)
               
rule copy_patches:
    output:
        train=f"{TMP_DIR}/training_samples_{{p_size}}.dat", 
        val=f"{TMP_DIR}/validation_samples_{{p_size}}.dat"
    input:
        train="data/patches/training_samples_{p_size}.dat",
        val="data/patches/validation_samples_{p_size}.dat"
    params:
        tmp_DIR = f"{TMP_DIR}"
    shell:
        "cp {input.train} {input.val} {params.tmp_DIR}" 

rule train:
    input:
        train=f"{TMP_DIR}/training_samples_{{p_size}}.dat", 
        val=f"{TMP_DIR}/validation_samples_{{p_size}}.dat"
    threads: 32
    output:
        out="output/patch-{p_size}_batch-{b_size}_LR-{lr}_filter-{filter}_depth-{depth}_convs-{convs}_loss-{loss}/model_log.txt"
    params:
        patch_size=lambda wildcards: wildcards.p_size,
        batch_size=lambda wildcards: wildcards.b_size,
        lr=lambda wildcards: wildcards.lr,
        ini_filter=lambda wildcards: wildcards.filter,
        depth=lambda wildcards: wildcards.depth,
        num_convs= lambda wildcards: wildcards.convs,
        loss=lambda wildcards: wildcards.loss
    shell:
        "python3 scripts/training_degad_CNN.py --input {input.train} {input.val} --patch_size {params.patch_size} --batch_size {params.batch_size} --lr {params.lr} --ini_filter {params.ini_filter} --depth {params.depth} --num_conv {params.num_convs} --loss {params.loss}  > {output.out}"
