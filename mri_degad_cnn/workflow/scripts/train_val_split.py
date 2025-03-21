import random
import argparse

def create_files(subject_file,train,val):
    with open(subject_file, "r") as f:
        subjects = f.read().splitlines()

    num_subjects = len(subjects)
    num_train = int(train * num_subjects)
    num_val = int(val * num_subjects)

    random.shuffle(subjects)

    train_set = subjects[:num_train]
    val_set = subjects[num_train:num_train + num_val]

    with open("../subject_info/train_subjects.txt", "w") as f:
        f.write("\n".join(train_set))

    with open("../subject_info/val_subjects.txt", "w") as f:
        f.write("\n".join(val_set))

def main():
    sub_file = snakemake.input["subject_file"]
    train= snakemake.params["train_ratio"]
    val= snakemake.params["val_ratio"]

    if train + val != 1:
        raise ValueError("Ratios do not sum up to 1.")

    create_files(sub_file,train,val)

if __name__ == "__main__":
    main()