from __future__ import annotations

from collections.abc import Mapping, Sequence, Hashable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

import monai
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.spatial.functional import resize
from monai.transforms.inverse import TraceableTransform
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import (
    LazyTransform,
    Randomizable,
    MapTransform,
    Transform,
)
from monai.transforms.utils import (
    generate_label_classes_crop_centers,
    check_non_lazy_pending_ops,
)
from monai.transforms.utils_pytorch_numpy_unification import (
    nonzero,
    ravel,
)
from monai.utils import ImageMetaKey as Key
from monai.networks import one_hot
from monai.utils import (
    convert_data_type,
    fall_back_tuple,
    TransformBackends,
    convert_to_tensor,
    ensure_tuple_rep,
)
from monai.utils.type_conversion import convert_to_dst_type

from src.utils.utils import get_key
from src.utils.constants import NUM_CLASS


def z_foreground(im):
    # im: [1, H, W, D], values in [0, Classes]
    if len(im.unique()) < 10:
        mask = torch.zeros_like(im)
    else:  # BTCV. In np because torch.any is weird
        z = np.any(im[0] > 0, axis=(0, 1), keepdims=False)
        zmin, zmax = np.where(z)[0][[0, -1]]
        mask = np.ones_like(im)
        mask[:, :, :, zmin:zmax] = 0
        mask = torch.tensor(mask).to(im)
    return mask.squeeze_(0).bool()  # 0 preserve, 1 mask out


class zMasking(Transform):
    """
    Create mask of regions without annotations (z axis)

    Args:
        classes: min number of classes in the dataset to do the processing.
            If number of classes is smaller, do nothing.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        classes: int | int = 33,
        threshhold: int | int = 10,
        **kwargs,
    ) -> None:
        self.classes = classes
        self.threshhold = threshhold
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert,

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor)  # [1, H, W, D]

        if len(img_t.unique()) < self.threshhold:
            mask = torch.zeros_like(img_t[0])
        else:
            z = torch.any(img_t[0].view(-1, img_t.shape[-1]), dim=0, keepdim=False)
            zmin, zmax = torch.where(z)[0][[0, -1]]

            mask = torch.ones_like(img_t[0])
            mask[:, :, zmin:zmax] = 0
            mask = torch.tensor(mask).to(img_t)
        mask = mask.bool()  # 0 preserve, 1 mask out
        img_t[:, mask] = self.classes + 2

        img, *_ = convert_to_dst_type(
            img_t, img, dtype=self.kwargs.get("dtype", torch.float)
        )
        return img


class zMaskingd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.zMasking`.
    """

    backend = zMasking.backend

    def __init__(
        self,
        keys: KeysCollection,
        classes: int | int = 33,
        threshhold: int | int = 10,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            classes: min number of classes in the dataset to do the processing.
                If number of classes is smaller, do nothing.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.

        """
        super().__init__(keys)

        self.converter = zMasking(classes=classes, threshhold=threshhold)
        self.converter.kwargs = kwargs

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class MaskedOneHot(Transform):
    """
    Convert the input tensor/array into one hot format

    Args:
        classes: if not None, convert input data into the one-hot format with specified number of classes.
            Defaults to ``None``.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        classes: int | None = None,
        categories: Sequence[int] | int = [8, 9, 10],
        **kwargs,
    ) -> None:
        assert classes is not None, "Need the total number of classes"
        self.classes = classes
        self.categories = categories
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the input tensor data to convert

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor)

        img_t = one_hot(
            img_t,
            num_classes=self.classes + 2,  # last channels: mask, lb mask
            dim=self.kwargs.get("dim", 0),
            dtype=self.kwargs.get("dtype", torch.float),
        )
        mask = img_t[-2].bool()
        mask_lb = img_t[-1].bool()
        img_t = img_t[:-2]  # eliminate masks

        img_t[:, mask] = -1

        mask_ = torch.zeros_like(img_t).bool()
        mask_[self.categories] = mask_lb
        img_t[mask_] = -1

        img, *_ = convert_to_dst_type(
            img_t, img, dtype=self.kwargs.get("dtype", torch.float)
        )
        return img


class MaskedOneHotd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MaskedOneHot`.
    """

    backend = MaskedOneHot.backend

    def __init__(
        self,
        keys: KeysCollection,
        classes: Sequence[int | None] | int | None = None,
        categories: Sequence[int] | int = [8, 9, 10],
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            classes: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.

        """
        super().__init__(keys)

        self.converter = MaskedOneHot(classes=classes, categories=categories)
        self.converter.kwargs = kwargs

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def map_dataset_classes_to_indices(
    label: NdarrayOrTensor,
    classes: list[NdarrayOrTensor] | None = None,
    image: NdarrayOrTensor | None = None,
    image_threshold: float = 0.0,
    max_samples_per_class: int | int = 50000,
    patch_size: list[NdarrayOrTensor] | None = None,
) -> list[NdarrayOrTensor]:
    """
    Filter out indices of the classes contained in this dataset of the input label data,
    return the indices after flattening.
    It can only handle One-Hot format label

    Args:
        label: use the label data to get the indices of every class.
        classes: classes contained in this dataset.
        image: if image is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        max_samples_per_class: maximum length of indices in each class to reduce memory consumption.
            Default is None, no subsampling.

    """
    check_non_lazy_pending_ops(label, name="map_classes_to_indices")
    img_flat: NdarrayOrTensor | None = None
    new_patch = torch.tensor(patch_size) // 4
    if image is not None:
        check_non_lazy_pending_ops(image, name="map_classes_to_indices")
        img_flat = ravel((image > image_threshold).any(0))

    # assuming the first dimension is channel
    classes = [0] + classes  # add background to the options
    channels = NUM_CLASS + 1  # with background

    indices: list[NdarrayOrTensor] = [[]] * channels
    ratios = [0] * channels
    for c in classes:
        binary = label == c
        if binary.sum() == 0:
            continue
        surroundings = torch.nonzero(binary)
        min_c = torch.maximum(
            torch.min(surroundings, dim=0)[0][1:] - new_patch, torch.tensor((0, 0, 0))
        )
        max_c = torch.minimum(
            torch.max(surroundings, dim=0)[0][1:] + new_patch,
            torch.tensor(binary.shape[1:]),
        )
        binary[:, min_c[0] : max_c[0], min_c[1] : max_c[1], min_c[2] : max_c[2]] = True
        label_flat = ravel(binary.any(0))

        if img_flat is not None:
            label_flat = img_flat & label_flat
        # no need to save the indices in GPU, otherwise, still need to move to CPU at runtime when crop by indices
        output_type = torch.Tensor if isinstance(label, monai.data.MetaTensor) else None
        cls_indices: NdarrayOrTensor = convert_data_type(
            nonzero(label_flat), output_type=output_type, device=torch.device("cpu")
        )[0]

        if len(cls_indices) > max_samples_per_class:
            sample_id = np.round(
                np.linspace(0, len(cls_indices) - 1, max_samples_per_class)
            ).astype(int)
            indices[c] = cls_indices[sample_id]
        else:
            indices[c] = cls_indices

        if len(cls_indices) > 10000:
            ratios[c] = 2
        elif len(cls_indices) > 5000:
            ratios[c] = 5
        elif len(cls_indices) > 0:
            ratios[c] = 10

    ratios[0] = 1

    return indices, ratios


class BatchWholeAndCrops(
    Randomizable, TraceableTransform, LazyTransform, MultiSampleTrait
):
    """
    Crop random fixed sized regions with the center being a class based on the specified ratios of every class.
    The label data can be One-Hot format array or Argmax data. And will return a list of arrays for all the
    cropped images.

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding every class, if None, must set at `self.__call__`.
        num_samples: number of samples (crop regions) to take in each list.
        image: if image is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        indices: if provided pre-computed indices of every class, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, expect to be 1 dim array
            of spatial indices after flattening. a typical usage is to call `ClassesToIndices` transform first
            and cache the results for better performance.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will remain
            unchanged.
        warn: if `True` prints a warning if a class is not present in the label.
        max_samples_per_class: maximum length of indices to sample in each class to reduce memory consumption.
            Default is None, no subsampling.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        template: dict[int] | None = None,
        label: torch.Tensor | None = None,
        num_samples: int = 1,
        image: torch.Tensor | None = None,
        image_threshold: float = 0.0,
        whole: bool = False,
        mode: str | None = None,
        allow_smaller: bool = False,
        warn: bool = True,
        max_samples_per_class: int | int = 50000,
        mask: int = -1,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy)
        self.spatial_size = spatial_size
        self.template = template
        self.label = label
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: tuple[tuple] | None = None
        self.allow_smaller = allow_smaller
        self.warn = warn
        self.max_samples_per_class = max_samples_per_class
        self.whole = whole
        self.mask = mask

        self.get_centers = (
            self.get_centers_normal if not whole else self.get_centers_whole
        )
        whole_idxs_x = [-spatial_size[0] // 2, 0, spatial_size[0] // 2]
        whole_idxs_y = [-spatial_size[1] // 2, 0, spatial_size[1] // 2]
        whole_idxs_z = [-spatial_size[2] // 2, 0, spatial_size[2] // 2]

        self.whole_idx_options = np.array(
            np.meshgrid(whole_idxs_x, whole_idxs_y, whole_idxs_z)
        ).T.reshape(-1, 3)

    def get_centers_whole(self, label, image, _shape, name, class_id):
        # reduce valid region to avoid adding noise with padding
        # TODO: fix sampling class_id in small images (mask)
        patch_size = np.array(self.spatial_size)
        new_patch = torch.tensor(patch_size) // 4
        p1, p2, p3 = np.minimum(patch_size, (np.array(image.shape)[1:] // 2) - 1)
        mask = torch.zeros_like(label)
        mask[:, p1:-p1, p2:-p2, p3:-p3] = 1
        image = image * mask

        if class_id is not None:
            binary = label == class_id
            if binary.sum() == 0:
                binary = label > 0
            else:
                surroundings = torch.nonzero(label)
                min_c = torch.maximum(
                    torch.min(surroundings, dim=0)[0][1:] - new_patch,
                    torch.tensor((0, 0, 0)),
                )
                max_c = torch.minimum(
                    torch.max(surroundings, dim=0)[0][1:] + new_patch,
                    torch.tensor(label.shape[1:]),
                )
                binary[
                    :, min_c[0] : max_c[0], min_c[1] : max_c[1], min_c[2] : max_c[2]
                ] = True
            label_whole = ravel(binary.any(0))
        else:
            label[label == self.mask] = 0
            label_whole = ravel((label > 0).any(0))

        img_flat = ravel((image > 0).any(0))
        label_whole = img_flat & label_whole
        if label_whole.sum() == 0:
            label_whole = img_flat

        center = np.unravel_index(np.random.choice(nonzero(label_whole)), _shape)
        self.whole_center = center

        patches = np.random.choice(
            np.arange(len(self.whole_idx_options)),
            size=self.num_samples - 1,
            replace=False,
        )
        self.centers = []
        for i in patches:
            shift = self.whole_idx_options[i]
            self.centers.append(tuple(np.asarray(center) + shift))
            # print(shift)

    def get_centers_normal(self, label, image, _shape, name, class_id):
        # reduce valid region to avoid adding noise with padding
        p1, p2, p3 = np.minimum(
            np.array(self.spatial_size) // 2, (np.array(image.shape)[1:] // 2) - 1
        )
        mask = torch.zeros_like(label)
        mask[:, p1:-p1, p2:-p2, p3:-p3] = 1
        image = image * mask

        key = get_key(name)
        indices_, ratios = map_dataset_classes_to_indices(
            label,
            self.template[key] if class_id is None else [class_id],
            image,
            self.image_threshold,
            self.max_samples_per_class,
            np.array(self.spatial_size),
        )

        self.centers = generate_label_classes_crop_centers(
            self.spatial_size,
            self.num_samples,
            _shape,
            indices_,
            ratios,
            self.R,
            self.allow_smaller,
            self.warn,
        )

    def randomize(
        self,
        label: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        name: str | None = None,
        class_id: int | None = None,
    ) -> None:
        _shape = None
        if label is not None:
            _shape = (
                label.peek_pending_shape()
                if isinstance(label, MetaTensor)
                else label.shape[1:]
            )
        elif image is not None:
            _shape = (
                image.peek_pending_shape()
                if isinstance(image, MetaTensor)
                else image.shape[1:]
            )
        if _shape is None:
            raise ValueError(
                "label or image must be provided to infer the output spatial shape."
            )

        self.get_centers(label, image, _shape, name, class_id)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val

    @property
    def requires_current_data(self):
        return False

    def __call__(
        self,
        img: torch.Tensor,
        label: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        mode: str | None = None,
        indices: list[NdarrayOrTensor] | None = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            img: input data to crop samples from based on the ratios of every class, assumes `img` is a
                channel-first array.
            label: the label image that is used for finding indices of every class, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``image > image_threshold`` to select the centers only in valid region. if None, use `self.image`.
            indices: list of indices for every class in the image, used to randomly select crop centers.
            randomize: whether to execute the random operations, default to `True`.
            lazy: a flag to override the lazy behaviour for this call, if set. Defaults to None.
        """
        if image is None:
            image = self.image
        if randomize:
            if label is None:
                label = self.label
            self.randomize(label, indices, image)
        results: list[torch.Tensor] = []
        if self.centers is not None:
            img_shape = (
                img.peek_pending_shape()
                if isinstance(img, MetaTensor)
                else img.shape[1:]
            )
            roi_size = fall_back_tuple(self.spatial_size, default=img_shape)
            lazy_ = self.lazy if lazy is None else lazy

            if self.whole:
                roi_size2 = [i * 2 for i in self.spatial_size]
                cropper = SpatialCrop(
                    roi_center=tuple(self.whole_center), roi_size=roi_size2, lazy=lazy_
                )
                cropped = cropper(img)

                cropped = resize(
                    cropped,
                    self.spatial_size,
                    mode,
                    lazy=lazy_,
                    align_corners=None,
                    dtype=None,
                    anti_aliasing=False,
                    anti_aliasing_sigma=None,
                    input_ndim=len(img_shape),
                    transform_info=self.get_transform_info(),
                )
                if get_track_meta():
                    ret_: MetaTensor = cropped  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = self.whole_center
                    self.push_transform(ret_, replace=True, lazy=lazy_)
                results.append([cropped, self.whole_center])

            for i, center in enumerate(self.centers):
                cropper = SpatialCrop(
                    roi_center=tuple(center), roi_size=roi_size, lazy=lazy_
                )
                cropped = cropper(img)
                if get_track_meta():
                    ret_: MetaTensor = cropped  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = center
                    self.push_transform(ret_, replace=True, lazy=lazy_)
                results.append([cropped, center])

        return results


class BatchWholeAndCropsd(Randomizable, MapTransform, LazyTransform, MultiSampleTrait):
    """
    Dictionary-based version :py:class:`BatchWholeAndCrops`.
    Crop random fixed sized regions with the center being a class based on the specified ratios of every class.
    The label data can be One-Hot format array or Argmax data. And will return a list of arrays for all the
    cropped images.

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding indices of every class.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image_key`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        indices_key: if provided pre-computed indices of every class, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, expect to be 1 dim array
            of spatial indices after flattening. a typical usage is to call `ClassesToIndices` transform first
            and cache the results for better performance.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will remain
            unchanged.
        allow_missing_keys: don't raise exception if key is missing.
        warn: if `True` prints a warning if a class is not present in the label.
        max_samples_per_class: maximum length of indices in each class to reduce memory consumption.
            Default is None, no subsampling.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = BatchWholeAndCrops.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        name_key: str,
        spatial_size: Sequence[int] | int,
        template: dict[int] | None = None,
        mode: list[NdarrayOrTensor] | None = None,
        mask: int | int = -1,
        num_samples: int = 1,
        whole: bool = False,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        warn: bool = True,
        max_samples_per_class: int | int = 50000,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.label_key = label_key
        self.name_key = name_key
        self.image_key = image_key
        self.mode = mode
        self.mask = mask
        self.cropper = BatchWholeAndCrops(
            spatial_size=spatial_size,
            template=template,
            num_samples=num_samples,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            warn=warn,
            max_samples_per_class=max_samples_per_class,
            lazy=lazy,
            whole=whole,
            mask=mask,
        )
        assert len(keys) == len(mode), "Need one mode per key"

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> BatchWholeAndCropsd:
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(
        self,
        label: torch.Tensor,
        image: torch.Tensor | None = None,
        name: dict[int] | None = None,
        class_id: int | None = None,
    ) -> None:
        self.cropper.randomize(label=label, image=image, name=name, class_id=class_id)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(
        self, data: Mapping[Hashable, Any], lazy: bool | None = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        self.randomize(
            d.get(self.label_key),
            d.get(self.image_key),
            d.get(self.name_key),
            d.get("class_id"),
        )  # type: ignore

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy
        for idx, key in enumerate(self.key_iterator(d)):
            for i, (im, center) in enumerate(
                self.cropper(
                    d[key],
                    randomize=False,
                    lazy=lazy_,
                    mode=self.mode[idx],
                )
            ):
                ret[i][key] = im
                ret[i]["crop_center"] = torch.tensor(center)
        return ret
