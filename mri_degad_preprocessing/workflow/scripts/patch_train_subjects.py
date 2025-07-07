import subprocess
from pathlib import Path
import sys

def patch_sub(input_gad, input_nogad, input_mask, output_patches, n, angle, radius_vector, frequency, train_subject_path):
    
    with open(train_subject_path) as f:
        print("opening train subjects")
        TRAIN_SUBJECTS = {line.strip() for line in f if line.strip()}

    subject = snakemake.wildcards["subject"]
    print("subject: ",subject)

    if subject not in TRAIN_SUBJECTS:
        print(f"Skipping patch creation for subject {subject} (not in train_subjects.txt)")
        # Create empty output file so Snakemake considers this rule done
        Path(snakemake.output["patches"]).touch()
        sys.exit(0)

    cmd = [
        "c3d",
        input_gad, input_nogad, input_mask,
        "-xpa", n, angle,
        "-xp", output_patches,
        radius_vector,
        frequency,
    ]

    print(f"Running patch extraction for subject {subject}...")
    subprocess.run(cmd, check=True)
if __name__ == "__main__":
    patch_sub(
        input_gad=snakemake.input["input_gad"],
        input_nogad=snakemake.input["input_nogad"],
        input_mask=snakemake.input["input_mask"],
        output_patches = snakemake.output["patches"],
        radius_vector = snakemake.params["radius_vector"],
        n = snakemake.params["n"],
        angle = snakemake.params["angle"],
        frequency = snakemake.params["frequency"],
        train_subject_path=snakemake.params["train_subject_path"]
    )