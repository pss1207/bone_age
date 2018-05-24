import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)




def make_train_dataset(dir, csv):
    images = []
    dir = os.path.expanduser(dir)

    for index in range(len(csv['id'])):
        fname = dir+'/{}.png'.format(csv['id'][index])
        if is_image_file(fname):
            male = csv['male'][index]
            boneage = csv['boneage'][index]
            item = (fname, male, boneage)
            images.append(item)

    return images
def make_test_dataset(dir, csv):
    images = []
    dir = os.path.expanduser(dir)

    for index in range(len(csv['Case ID'])):
        fname = dir+'/{}.png'.format(csv['Case ID'][index])
        if is_image_file(fname):
            sex = csv['Sex'][index]
            if sex == 'M':
                male = True
            else:
                male = False
            item = (fname, male)
            images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomTrainImageFolder(data.Dataset):

    def __init__(self, root, csv, transform=None,
                 loader=default_loader):
        imgs = make_train_dataset(root, csv)

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, male, boneage = self.imgs[index]
        img = self.loader(path)
        male = male.astype('int')
        if self.transform is not None:
            img = self.transform(img)

        return img, male, boneage

    def __len__(self):
        return len(self.imgs)


class CustomTestImageFolder(data.Dataset):

    def __init__(self, root, csv, transform=None,
                 loader=default_loader):
        imgs = make_test_dataset(root, csv)

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, male = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        male = male.astype('int')

        return img, male

    def __len__(self):
        return len(self.imgs)
