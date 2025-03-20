import os
import glob

def concatenate(train_file, val_file, patch_dir, training_patches, validation_patches):
    """
    Reads train and val subject lists, concatenates corresponding .dat files from patch_dir,
    and writes to output .dat files.
    """

    # read subject lists
    with open(train_file, 'r') as f:
        train_subs = f.read().splitlines()

    with open(val_file, 'r') as f:
        val_subs = f.read().splitlines()

    with open(training_patches, 'a') as train_out, open(validation_patches, 'a') as val_out:
        
        # process training subjects
        for sub in train_subs:
            patch_paths = glob.glob(os.path.join(patch_dir, f"sub-{sub}*_samples_31.dat"), recursive=True)
            
            if patch_paths:
                for patch_path in patch_paths:
                    with open(patch_path, 'r') as patch_file:
                        train_out.write(patch_file.read())
            else:
                print(f"Warning: Missing patch file for subject {sub}")

        # process validation subjects
        for sub in val_subs:
            patch_paths = glob.glob(os.path.join(patch_dir, f"sub-{sub}*_samples_31.dat"), recursive=True)
            
            if patch_paths:
                for patch_path in patch_paths:
                    with open(patch_path, 'r') as patch_file:
                        val_out.write(patch_file.read())
            else:
                print(f"Warning: Missing patch file for subject {sub}")

def main():
    """
    Uses Snakemake inputs/outputs to call concatenate function.
    """

    train_file = snakemake.input["train"]
    val_file = snakemake.input["val"]   
    patch_dir = snakemake.input["patch_dir"] 
    training_patches = snakemake.output["train_dat"] 
    validation_patches = snakemake.output["val_dat"]

    concatenate(train_file, val_file, patch_dir, training_patches, validation_patches)

if __name__ == "__main__":
    main()


