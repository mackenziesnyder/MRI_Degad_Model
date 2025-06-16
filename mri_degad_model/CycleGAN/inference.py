import argparse
import torch

def inference(input_dir, output_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_pairs = get_test_pairs(input_dir)
    print("data_pairs", data_pairs)
    model_path = "genz.pth.tar"

    for data in data_pairs:
        gad_image_path = data["image"]
        nogad_image_path = data["label"]
        slice_dir, sub = slice_gad_2d(gad_image_path, output_dir)
        degad_dir = apply_model(model_path, slice_dir, output_dir, sub, device)
        degad_path = rebuild_3d(degad_dir, gad_image_path)
        resampled_gad = resample_degad_to_gad(degad_path, gad_image_path)
        compute_metrics(resampled_gad,  nogad_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cyclegan.")
    parser.add_argument("--input", required=True, help="Test directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    inference(input_dir, output_dir)
