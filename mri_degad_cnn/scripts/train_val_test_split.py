import random
import argparse

def create_files(subject_file,train,val,test):
    with open(subject_file, "r") as f:
        subjects = f.read().splitlines()

    num_subjects = len(subjects)
    num_train = int(train * num_subjects)
    num_val = int(val * num_subjects)
    num_test = num_subjects - num_train - num_val

    random.shuffle(subjects)

    train_set = subjects[:num_train]
    val_set = subjects[num_train:num_train + num_val]
    test_set = subjects[num_train + num_val:]

    with open("../subject_info/train_subjects.txt", "w") as f:
        f.write("\n".join(train_set))

    with open("../subject_info/val_subjects.txt", "w") as f:
        f.write("\n".join(val_set))

    with open("../subject_info/test_subjects.txt", "w") as f:
        f.write("\n".join(test_set))

def main():
    sub_file = snakemake.input["subject_file"]
    train= snakemake.params["train_ratio"]
    val= snakemake.params["val_ratio"]
    test= snakemake.params["test_ratio"]

    if train + val+ test != 1:
        raise ValueError("Ratios do not sum up to 1.")

    create_files(sub_file,train,val,test)

if __name__ == "__main__":
    main()