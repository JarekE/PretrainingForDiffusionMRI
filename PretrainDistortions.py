from typing import Optional, Sequence, Union
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import matplotlib as mpl
from matplotlib import pyplot
import config


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class Transformation(DeviceDtypeModuleMixin, torch.nn.Module):
    """Class representing arbitrary transformations."""

    def __init__(self, image_size: Sequence, previous_trafo: Optional[torch.Tensor] = None):
        """

        Args:
            image_size: the size of the target image
            previous_trafo: the previous transformation.
                If given it will always be added to the current transformation. Defaults to None.
        """
        super().__init__()

        self.dim = len(image_size)
        self.register_buffer("image_size", torch.tensor(image_size))
        self.register_buffer("grid", create_grid(self.image_size, dtype=self.dtype, device=self.device))
        self.image_size = torch.tensor(image_size)

        # no previous transfomation in my case
        if isinstance(previous_trafo, torch.Tensor):
            previous_trafo = previous_trafo.detach()

            # resample previous transform if necessary
            if previous_trafo.shape[1:-1] != image_size:
                previous_trafo = interpolate_displacement(previous_trafo, new_size=image_size)
        self.register_buffer("previous_trafo", previous_trafo)

    # not used here?!?
    def outside_mask(self, moving: torch.Tensor) -> torch.Tensor:
        """returns a mask specifying which pixels are transformed outside the image domain

        Args:
            moving: the moving image

        Returns:
            torch.Tensor: the mask
        """
        mask = torch.zeros_like(self.image_size, dtype=torch.uint8, device=self.device)

        # exclude points which are transformed outside the image domain
        for dim in range(self.displacement.size()[-1]):
            mask += self.displacement[..., dim].gt(1) + self.displacement[..., dim].lt(-1)

        mask = mask == 0

        return mask

    # called
    def compute_displacement(self) -> torch.Tensor:
        # white noise
        displacement = torch.rand(1, 3, self.image_size[0].item(), self.image_size[1].item(), self.image_size[2].item())
        #displacement_gaussian = torch.rand(1, 1, self.image_size[0].item(), self.image_size[1].item(), self.image_size[2].item())

        # Conv3d with gaussian kernel
        #9
        #smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=1, dim=3)
        #4
        #displacement_gaussian = F.pad(displacement_gaussian, (1, 1, 1, 1, 1, 1))
        #displacement[0, 0, :, :, :] = smoothing(displacement_gaussian)

        # correction of dimension
        displacement = displacement.numpy()
        displacement = np.moveaxis(displacement, 1, 4)
        displacement = torch.from_numpy(displacement)

        # zeros
        displacement[0, :, :, :, 0] = torch.ones_like(displacement[0, :, :, :, 0])
        displacement[0, :, :, :, 1] = (torch.ones_like(displacement[0, :, :, :, 1])*(random.randrange(90, 110, 1)/100))
        displacement[0, :, :, :, 2] = torch.ones_like(displacement[0, :, :, :, 2])

        return displacement

    # called
    def forward(self, moving_image: torch.Tensor) -> torch.Tensor:
        """Applies the current transform to the given image

        Args:
            moving_image: the image to transform

        Returns:
            torch.Tensor: the transformed image
        """

        #Add transform
        #displacement = self.compute_complete_transform_displacement() + self.grid

        #Non_liearity
        #grid = self.grid
        #plane_grid = grid[:, :, :, :, 1:2]
        #grid[:, :, :, :, 1:2] = torch.where(plane_grid < 0, -torch.pow(-plane_grid, 0.7), torch.pow(plane_grid, 0.7))
        # displacement = grid

        #Distortion with multiplication
        x = self.compute_complete_transform_displacement()
        displacement = self.grid * x

        if 0:
            new_array = torch.rand(80, 96)
            new_array[:, :] = displacement[0, :, :, 32, 0]
            fig = pyplot.figure(2)
            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                 ['white', 'grey', 'black'],
                                                                 256)

            new_array = new_array.numpy()
            img2 = pyplot.imshow(new_array.T, interpolation='nearest',
                                 cmap=cmap2,
                                 origin='lower')

            pyplot.colorbar(img2, cmap=cmap2)
            fig.savefig("test1.png")

            fig = pyplot.figure(3)
            test_array = self.grid.numpy()
            img3 = pyplot.imshow(test_array[0, :, :, 10, 0].T, interpolation='nearest',
                                 cmap=cmap2,
                                 origin='lower')
            pyplot.colorbar(img3, cmap=cmap2)
            fig.savefig("test2.png")

        moving_image = torch.tensor(moving_image)

        warped_moving = F.grid_sample(moving_image[None], displacement)[0]
        return warped_moving


    def compute_complete_transform_displacement(self) -> torch.Tensor:
        displacement = self.compute_displacement()
        return displacement


# Just the grid here
def create_grid_2d(
    image_size: Sequence,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """creates a 2d grid for a certain image size

    Args:
        image_size: the size of the target image
        dtype: the dtype of the resulting grid. Defaults to None.
        device: [the device the resulting grid should lie on. Defaults to None.

    Returns:
        torch.Tensor: the created grid
    """
    nx = image_size[0]
    ny = image_size[1]

    x = torch.linspace(-1, 1, steps=ny, dtype=dtype, device=device)
    y = torch.linspace(-1, 1, steps=nx, dtype=dtype, device=device)

    x = x.expand(nx, -1)
    y = y.expand(ny, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(3)
    y.unsqueeze_(0).unsqueeze_(3)

    return torch.cat((x, y), 3).to(dtype=dtype, device=device)

def create_grid_3d(
    image_size: Sequence,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """creates a 3d grid for a certain image size

    Args:
        image_size: the size of the target image
        dtype: the dtype of the resulting grid. Defaults to None.
        device: [the device the resulting grid should lie on. Defaults to None.

    Returns:
        torch.Tensor: the created grid
    """
    nz = image_size[0]
    ny = image_size[1]
    nx = image_size[2]

    x = torch.linspace(-1, 1, steps=nx, dtype=dtype, device=device)
    y = torch.linspace(-1, 1, steps=ny, dtype=dtype, device=device)
    z = torch.linspace(-1, 1, steps=nz, dtype=dtype, device=device)

    x = x.expand(ny, -1).expand(nz, -1, -1)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(4)
    y.unsqueeze_(0).unsqueeze_(4)
    z.unsqueeze_(0).unsqueeze_(4)
    return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)

def create_grid(
    image_size: Sequence,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """creates a Nd grid for a certain image size

    Args:
        image_size: the size of the target image
        dtype: the dtype of the resulting grid. Defaults to None.
        device: [the device the resulting grid should lie on. Defaults to None.

    Raises:
        ValueError: Invalid dimensionality

    Returns:
        torch.Tensor: the created grid
    """
    dim = len(image_size)

    if dim == 2:
        return create_grid_2d(image_size=image_size, dtype=dtype, device=device)

    elif dim == 3:
        return create_grid_3d(image_size=image_size, dtype=dtype, device=device)
    else:
        raise ValueError("Error " + str(dim) + " is not a valid grid type")


def interpolate_displacement(
    displacement: torch.Tensor, new_size: Sequence, interpolation: str = "linear"
) -> torch.Tensor:
    """interpolates the displacement field to fit for a new target size

    Args:
        displacement: the displacement field to interpolate
        new_size: the new image size
        interpolation: the interpolation type. Defaults to "linear".

    Returns:
        torch.Tensor: the interpolated displacement field
    """
    dim = displacement.size(-1)
    if dim == 2:
        displacement = displacement.permute(0, 3, 1, 2)
        if interpolation == "linear":
            interpolation = "bilinear"
        else:
            interpolation = "nearest"
    elif dim == 3:
        displacement = displacement.permute(0, 4, 1, 2, 3)
        if interpolation == "linear":
            interpolation = "trilinear"
        else:
            interpolation = "nearest"

    interpolated_displacement = F.interpolate(displacement, size=new_size, mode=interpolation, align_corners=False)

    if dim == 2:
        interpolated_displacement = interpolated_displacement.permute(0, 2, 3, 1)
    elif dim == 3:
        interpolated_displacement = interpolated_displacement.permute(0, 2, 3, 4, 1)

    return interpolated_displacement