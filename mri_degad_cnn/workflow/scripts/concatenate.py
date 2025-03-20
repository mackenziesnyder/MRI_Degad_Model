import os

def concatenate(train_file, val_file, patches, training_patches, validation_patches):
    with open(train_sub_file, 'r') as f:
        train_sub = f.read().splitlines()

    with open(val_sub_file, 'r') as f:
        val_sub = f.read().splitlines()
    
    with open(training_patches, 'a') as train_file:
        for sub in train_sub:
            with open(patches, 'r') as patch_file_content:
                train_file.write(patch_file_content.read())
    
    with open(validation_patches, 'a') as val_file:
        for sub in val_sub:
            with open(patches, 'r') as patch_file_content:
                val_file.write(patch_file_content.read())
    
    return training_patches, validation_patches

def main():
    train_file = snakemake.input["train_file"]
    val_file = snakemake.input["val_file"]
    patches = snakemake.input["patches"]
    training_patches = snakemake.output["training_patches"]
    validation_patches = snakemake.output["validation_patches"]

    concatenate(train_file, val_file, patches, training_patches, validation_patches)

if __name__ == "__main__":
    main()
