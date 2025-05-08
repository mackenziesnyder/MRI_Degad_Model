import os
import nibabel as nib 
import argparse
import glob
import numpy as np
from svg_transforms import extract_svg
import matplotlib.pyplot as plt
from nilearn import plotting

plot_args = dict(
    cmap=plt.cm.gray,
    draw_cross=False,
)
# Helper to generate inline SVG string
def get_svg_image(fg_img, bg_img=None, display_mode="z", cut_coords=[0]):
    display = plotting.plot_anat(bg_img or fg_img, display_mode=display_mode, cut_coords=cut_coords, **plot_args)
    if bg_img is not None:
        display.add_overlay(fg_img, alpha=0.7)
    tmp_svg = extract_svg(display, dpi=300)  # You should define extract_svg
    display.close()
    return tmp_svg

def output_html_qc_file(original_image_bids, degad_images, output_path): 
    
    # subject directories from degad path
    
    degad_bids_path = os.path.join(degad_images, "bids")
    subject_dirs = glob.glob(os.path.join(degad_bids_path, "sub-*"))
    
    data_dicts = []
    
    for sub_path in subject_dirs:
        if not os.path.isdir(sub_path):
            continue
        
        sub_id = os.path.basename(sub_path)
        print(f"Processing {sub_id}")
        
        # degad image 
        degad_image_path = os.path.join(
            sub_path, "ses-pre", "anat", f"{sub_id}_acq-degad_T1w.nii.gz"
        )
        
        # gad/nogad images from original dataset
        gad_images = glob.glob(os.path.join(
            original_image_bids, "work", sub_id, "ses-pre", "normalize", "*acq-gad*_T1w.nii.gz"
        ))
        nogad_images = glob.glob(os.path.join(
            original_image_bids, "work", sub_id, "ses-pre", "normalize", "*acq-nongad*_T1w.nii.gz"
        ))

        if gad_images and nogad_images:
            data_dicts.append({
                "gad_image": gad_images[0],
                "nogad_image": nogad_images[0],
                "degad_image": degad_image_path
            })
        else:
            print(f"Missing gad or nogad images for {sub_id}, skipping.")
    
    print("Loaded", len(data_dicts), "triplet samples.")
    
    with open(output_path, "w") as f:
        f.write("<html><body><center>")
        f.write("<h3 style='font-size:42px'>Model QC</h3>")

        for triplet in data_dicts:
            
            degad_image = triplet["degad_image"]
            gad_image = triplet["gad_image"]
            nogad_image = triplet["nogad_image"]
            
            # degad image
            degad_img = nib.load(degad_image)
            degad_img = nib.Nifti1Image(
                degad_img.get_fdata().astype(np.float32),
                header=degad_img.header,
                affine=degad_img.affine,
            )

            # nongad image
            nogad_img = nib.load(nogad_image)
            nogad_img = nib.Nifti1Image(
                nogad_img.get_fdata().astype(np.float32),
                header=nogad_img.header,
                affine=nogad_img.affine,
            )

            # gad image
            gad_img = nib.load(gad_image)
            gad_img = nib.Nifti1Image(
                gad_img.get_fdata().astype(np.float32),
                header=gad_img.header,
                affine=gad_img.affine,
            )

            # Generate SVGS
            svg_gad = get_svg_image(gad_img)
            svg_nogad = get_svg_image(nogad_img)
            svg_degad = get_svg_image(degad_img)

            f.write(f"""
                <div style="display:flex; justify-content:space-around; margin:40px;">
                    <div>
                        <h4>Gad Image</h4>
                        <p>{gad_img}</p>
                    </div>
                    <div>
                        <h4>No-Gad Image</h4>
                        <p>{nogad_img}</p>
                    </div>
                    <div>
                        <h4>De-Gad Image</h4>
                        <p>{degad_img}</p>
                    </div>
                </div>
                <hr style="height:2px; background-color:black; margin:30px 0;">
            """)

        f.write("</center></body></html>")

    print(f"HTML output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual QC of test results")
    parser.add_argument("--original_image_bids", required=True, help="Path to bids dataset with original images")
    parser.add_argument("--degad_images", required=True, help="Path to bids dataset with degad images")
    parser.add_argument("--output_path", required=True, help="Path to html output")
    args = parser.parse_args()

    output_html_qc_file(args.original_image_bids, args.degad_images, args.output_path)
