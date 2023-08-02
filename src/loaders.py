import os
import numpy as np
import pathlib
import rasterio
import cv2
import torch

from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Dict, List



# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class SEN12MS_RGB(Dataset):
    """Custom dataset for SEN12MS data"""

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int):# -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = rasterio.open(str(image_path)).read([2,3,4])
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #norm_image = (image - image.min()) / (image.max() - image.min())
        return np.transpose(image_norm,(1,2,0))
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

class SEN12MS_FULL(Dataset):
    """Custom dataset for SEN12MS data"""

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        self.label = "Spring"
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int):# -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = rasterio.open(str(image_path)).read()
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #norm_image = (image - image.min()) / (image.max() - image.min())
        return np.transpose(image_norm,(1,2,0))
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

class overfit(Dataset):
    def __init__(self, image_path, length, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, x):
        # open image here as PIL / numpy
        image = Image.open(self.image_path)
        label = "blue"
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class overfit_sen12(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 5000

    def __getitem__(self, x):
        # open image here as PIL / numpy
        image = rasterio.open(str(self.image_path)).read()
        image_norm = cv2.normalize(image.astype(np.float32), dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        image = np.transpose(image_norm,(1,2,0))
        label = torch.tensor(0).int()
        if self.transform is not None:
            image = self.transform(image)

        return image, label
