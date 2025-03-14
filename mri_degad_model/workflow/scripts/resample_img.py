from nilearn.image import resample_img
import numpy as np
import nibabel as nib

def resample_to_isotropic_resolution(
    input_image_path, isotropic_resolution, output_img
):
    """
    Resample a NIfTI MRI image to the specified isotropic resolution.

    Parameters
    ----------
        input_image_path:: str
            Path to the NIfTI MRI image file.

        isotropic_resolution:: float
            The desired isotropic resolution in mm.

        output_img :: str
            Name of the modified image

    Returns
    -------
        None

    """
    # Load the NIfTI image
    img = nib.load(input_image_path)

    # Resample the image
    resampled_image = resample_img(
        img, target_affine=np.eye(3), interpolation="linear"
    )

    # save image
    resampled_image.to_filename(output_img)

if __name__ == "__main__":
    resample_to_isotropic_resolution(
        input_image_path=snakemake.input["input_im"],
        isotropic_resolution=snakemake.params["res"],
        output_img=snakemake.output["resam_im"],
    )