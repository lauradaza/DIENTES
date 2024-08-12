import torch
import numpy as np
from pathlib import Path
import SimpleITK as sitk

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from src.utils.config import args
from src.test import ToothFairy_inference


def preprocessing(x):
    print("Preprocessing...")
    # To tensor
    x = torch.from_numpy(x.astype(np.float32))

    # "RAS" following training (transforming to nifti did something weird)
    x = x.permute(2, 1, 0)
    x = torch.flip(x, (0, 1))

    # Scale intensities between [-1000, 4500]
    amin, amax = -1000, 4500
    x = (x - amin) / (amax - amin)
    x = torch.clip(x, 0, 1)

    # Add channel and batch dimension
    return x[None, None]


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Toothfairy_algorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path("/input/images/cbct/"),
            output_path=Path("/output/images/oral-pharyngeal-segmentation/"),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)
        print("=== Segmentation Algorithm initialized ===")

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image):
        input_array = sitk.GetArrayFromImage(input_image)
        input_tensor = preprocessing(input_array)

        output = ToothFairy_inference(
            input_tensor,
            device=get_default_device(),
        )

        # Return to original orientation
        output = np.transpose(output, (2, 1, 0))
        output = sitk.GetImageFromArray(output)
        return output


if __name__ == "__main__":
    Toothfairy_algorithm().process()
