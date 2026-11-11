import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision
from packaging import version
from torchvision.transforms.v2 import functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PILImageResampling = PIL.Image.Resampling
else:
    PILImageResampling = PIL.Image

ImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    List["PIL.Image.Image"],
    List[np.ndarray],
    List["torch.Tensor"],
]  # noqa

VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarray"]],
    List[List["torch.Tensor"]],
]  # noqa


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please "
            f"select one of {list(cls._value2member_map_.keys())}"
        )


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """ # noqa: E501

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"


def is_valid_image(img):
    allowed_types = (PIL.Image.Image, np.ndarray, torch.Tensor)
    return isinstance(img, allowed_types)


def is_valid_list_of_images(images: List):
    return images and all(is_valid_image(image) for image in images)


def is_pil_image(img):
    return isinstance(img, PIL.Image.Image)


def make_flat_list_of_images(
    images: Union[List[ImageInput], ImageInput],
) -> ImageInput:
    """
    Ensure that the output is a flat list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a nested list of images, it is converted to a flat list of images.
    Args:
        images (`Union[List[ImageInput], ImageInput]`):
            The input image.
    Returns:
        list: A list of images or a 4d array of images.
    """ # noqa: E501
    # If the input is a nested list of images, we flatten it
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) for images_i in images)
    ):
        return [img for img_list in images for img in img_list]

    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == 3:
            return images
        if images[0].ndim == 4:
            return [img for img_list in images for img in img_list]

    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == 3:
            return [images]
        if images.ndim == 4:
            return list(images)

    raise ValueError(f"Could not make a flat list of images from {images}")


def make_batched_videos(videos) -> VideoInput:
    """
    Ensure that the input is a list of videos.
    Args:
        videos (`VideoInput`):
            Video or videos to turn into a list of videos.
    Returns:
        list: A list of videos.
    """
    if (
        isinstance(videos, (list, tuple))
        and isinstance(videos[0], (list, tuple))
        and is_valid_image(videos[0][0])
    ):
        # case 1: nested batch of videos so we flatten it
        if not is_pil_image(videos[0][0]) and videos[0][0].ndim == 4:
            videos = [
                [video for batch_list in batched_videos for video in batch_list]
                for batched_videos in videos
            ]
        # case 2: list of videos represented as list of video frames
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if is_pil_image(videos[0]) or videos[0].ndim == 3:
            return [videos]
        elif videos[0].ndim == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos):
        if is_pil_image(videos) or videos.ndim == 3:
            return [[videos]]
        elif videos.ndim == 4:
            return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


def valid_images(imgs):
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not valid_images(img):
                return False
    # If not a list of tuple, we have been given
    # a single image or batched tensor of images
    elif not is_valid_image(imgs):
        return False
    return True


def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False


def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    """
    Ensure that the output is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """ # noqa: E501
    if is_batched(images):
        return images

    # Either the input is a single image, in which case we create a list of length 1
    if is_pil_image(images):
        # PIL images are never batched
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # Batch of images
            images = list(images)
        elif images.ndim == expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {expected_ndims + 1} "
                f"or {expected_ndims} dimensions, but got"
                f" {images.ndim} dimensions."
            )
        return images
    raise ValueError(
        "Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, "
        f"torch.Tensor, tf.Tensor or jax.ndarray, but got {type(images)}."
    )


def validate_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[Dict[str, int]] = None,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: Optional["PILImageResampling"] = None,
):
    """
    Checks validity of typically used arguments in an `ImageProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    Many incompatibilities are model-specific. `do_pad` sometimes needs `size_divisor`,
    sometimes `size_divisibility`, and sometimes `size`. New models and processors added should follow
    existing arguments when possible.

    """ # noqa: E501
    if do_rescale and rescale_factor is None:
        raise ValueError(
            "`rescale_factor` must be specified if `do_rescale` is `True`."
        )

    if do_pad and size_divisibility is None:
        # Here, size_divisor might be passed as the value of size
        raise ValueError(
            "Depending on the model, `size_divisibility`, `size_divisor`, "
            "`pad_size` or `size` must be specified if `do_pad` is `True`."
        )

    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError(
            "`image_mean` and `image_std` must both be specified "
            "if `do_normalize` is `True`."
        )

    if do_center_crop and crop_size is None:
        raise ValueError("`crop_size` must be specified if `do_center_crop` is `True`.")

    # if do_resize and (size is None or resample is None):
    #    raise ValueError("`size` and `resample` must be specified "
    #                     "if `do_resize` is `True`.")


def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """ # noqa: E501
    if not isinstance(image, PIL.Image.Image):
        return image

    if image.mode == "RGB":
        return image

    image = image.convert("RGB")
    return image


def to_numpy_array(img) -> np.ndarray:
    if not is_valid_image(img):
        raise ValueError(f"Invalid image type: {type(img)}")
    if isinstance(img, PIL.Image.Image):
        return np.array(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise ValueError("Create array from image failed!")


def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning(
            "The channel dimension is ambiguous. Got image shape %s. "
            "Assuming channels are the first dimension.",
            image.shape,
        )
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_image_size(
    image: np.ndarray, channel_dim: ChannelDimension = None
) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """ # noqa: E501
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class FastVideoFeature:

    def __init__(
        self,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        device: str = "cpu",
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.device = device

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: F.InterpolationMode = F.InterpolationMode.BICUBIC,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: "torch.Tensor" = None,
        image_std: "torch.Tensor" = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            image = torch.from_numpy(image).to(self.device)
            # convert to ChannelDimension.FIRST
            if input_data_format == ChannelDimension.LAST:
                image = image.permute(2, 0, 1)
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = F.resize(
                    image, size=(resized_height, resized_width), interpolation=resample
                )
            if do_normalize:
                image = F.normalize(image.float(), image_mean, image_std)
            processed_images.append(image)

        patches = torch.stack(processed_images, dim=0)
        if data_format == ChannelDimension.LAST:
            patches = patches.permute(0, 3, 1, 2)
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = (
                patches[-1].unsqueeze(0).repeat(self.temporal_patch_size - 1, 1, 1, 1)
            )
            patches = torch.cat([patches, repeats], dim=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        patches = patches.view(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def image_process(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: F.InterpolationMode = F.InterpolationMode.BICUBIC,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        min_pixels: int = None,
        max_pixels: int = None,
    ):
        if images is not None:
            images = make_flat_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)
        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if image_mean is not None:
            if do_rescale:
                image_mean = [x / rescale_factor for x in image_mean]
            image_mean = torch.tensor(image_mean).to(self.device, torch.float32)
        if image_std is not None:
            if do_rescale:
                image_std = [x / rescale_factor for x in image_std]
            image_std = torch.tensor(image_std).to(self.device, torch.float32)

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = torch.tensor(vision_grid_thws)
            data = {
                "pixel_values": pixel_values.cpu(),
                "image_grid_thw": vision_grid_thws.cpu(),
            }

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = torch.tensor(vision_grid_thws)
            data = {
                "pixel_values_videos": pixel_values.cpu(),
                "video_grid_thw": vision_grid_thws.cpu(),
            }

        return data
