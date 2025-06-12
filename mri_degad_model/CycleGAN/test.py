from test_functions import load_gad_nogad_pairs, slice_image, apply_model, rebuild_3d, compute_image_similarity
import argparse
import os

def test_model(bids_dir, output_dir):
    data = load_gad_nogad_pairs(bids_dir)
    print("Loaded data:", data)

    # Slice gad and nogad images, returns list of slice file paths or PIL images
    slices_gad = slice_image(data[0]["image"], output_dir, tag="gad")
    slices_nogad = slice_image(data[0]["label"], output_dir, tag="nogad")

    # Apply the model to the gad slices, returns list of numpy arrays or tensors of generated slices
    slices_degad = apply_model(slices_gad, output_dir)

    # Rebuild 3D volumes from slices (lists of slices)
    nifti_degad = rebuild_3d(slices_degad, output_dir, "degad")
    nifti_nogad = rebuild_3d(slices_nogad, output_dir, "nogad")

    # Compute similarity metric (e.g. SSIM) between generated and ground truth
    
    similarity = compute_image_similarity("degad.nii", "nogad.nii")
    print("Image similarity:", similarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test cyclegan model")
    parser.add_argument("--bids_dir", required=True, help="Path to the test data")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory")
    args = parser.parse_args()

    # Create output dir if doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    test_model(args.bids_dir, args.output_dir)