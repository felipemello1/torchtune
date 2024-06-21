# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from collections import defaultdict
from typing import List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def get_canvas_best_fit(
    image: torch.Tensor,
    possible_resolutions: torch.Tensor,
) -> Tuple[int, int]:
    """
    Args:
        image (torch.Tensor): Tensor of shape (..., H, W)
        possible_resolutions (torch.Tensor): Tensor with possible resolutions to choose from.
            e.g. possible_resolutions = torch.tensor([[224, 672], [672, 224], [224, 448]])
    Returns:
        image tiles (torch.Tensor):
    """
    height, width = image.shape[-2:]
    image_size = (height, width)

    # Check if the image can be fit to the canvas without downsampling and distortion
    best_resolution = _get_smallest_upscaling_possibility(
        image_size, possible_resolutions
    )

    if best_resolution is None:
        # If we did not find a canvas, find the closest aspect ratio to downsample the image to
        best_resolution = _find_closest_resolution(image_size, possible_resolutions)

    return best_resolution


def _find_closest_resolution(
    image_size: Tuple[int, int], possible_resolutions: torch.Tensor
) -> Tuple[int, int]:
    """
    Finds the closest resolution from a list that best matches the resolution of a given image size.

    Args:
        image_size (Tuple[int, int]): A tuple containing the height and width of the image.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each row
            represents a possible resolution (height, width).

    Returns:
        Tuple[int, int]: The resolution (height, width) from possible_resolutions that is closest in aspect ratio to the image.

    Example:
        >>> image_size = (800, 600)
        >>> possible_resolutions = torch.tensor([
        ...     [224, 896],
        ...     [448, 448],
        ...     [224, 224],
        ...     [896, 224],
        ...     [224, 672],
        ...     [672, 224],
        ...     [224, 448],
        ...     [448, 224]])
        >>> _find_closest_resolution(image_size, possible_resolutions)
        (448, 448)
        We have:
            target_aspect_ratio = width/height = 600/800 = 0.75
            possible_aspect_ratios = tensor([4.0000, 1.0000, 1.0000, 0.2500, 3.0000, 0.3333, 2.0000, 0.5000])
        Ratios are filtered and closest one is selected:
            valid_ratios = tensor([4., 1., 1., 3., 2.])
            closest_aspect_ratio = tensor(1.)
        Notice how there are two valid_ratios == 1, mapping to [[448, 448], [224, 224]].
        We pick the one with largest height, since its portrait mode (h > w):
            selected_pair = tensor([448, 448])
    """

    image_height, image_width = image_size
    target_aspect_ratio = image_width / image_height

    # Calculate aspect ratios: widths/heights
    possible_aspect_ratios = possible_resolutions[:, 1] / possible_resolutions[:, 0]

    # We choose to filter out resolutions that will accentuate the landscape/portrait aspect ratio.
    # eg: target_aspect_ratio = 2, possible_aspect_ratios = [0.5, 1, 2, 3], valid_ratios = [0.5, 1, 2]
    if target_aspect_ratio >= 1:  # landscape
        valid_ratios = possible_aspect_ratios[
            possible_aspect_ratios <= target_aspect_ratio
        ]
    else:  # portrait
        valid_ratios = possible_aspect_ratios[
            possible_aspect_ratios >= target_aspect_ratio
        ]

    # Find closest_aspect_ratio
    differences = torch.abs(valid_ratios - target_aspect_ratio)
    min_index = torch.argmin(differences)
    closest_aspect_ratio = valid_ratios[min_index]

    # Filter the resolutions with the closest aspect ratio. There can be multiple:
    # E.g. 800x400 and 400x200 are both 2:1 aspect ratio
    closest_indices = torch.where(possible_aspect_ratios == closest_aspect_ratio)[0]
    closest_resolutions = possible_resolutions[closest_indices]

    if target_aspect_ratio >= 1:  # landscape
        # Landscape target. Select the resolution with the maximum width.
        selected_index = torch.argmax(closest_resolutions[:, 1])
    else:
        # Portrait target. Select the resolution with the maximum height
        selected_index = torch.argmax(closest_resolutions[:, 0])

    selected_pair = closest_resolutions[selected_index]
    return tuple(selected_pair.tolist())


def _get_smallest_upscaling_possibility(
    image_size: Tuple[int, int], possible_resolutions: torch.Tensor
) -> Optional[Tuple[int, int]]:
    """
    Determines the smallest upscaling possibility from a list of possible resolutions,
    without distortion, for a given image size

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.

    Then, picks the resolution that allows the smallest upscaling.
    If no upscaling is possible, i.e., all scaling factors are less than 1, the function returns None.

    Args:
        image_size (Tuple[int, int]): A tuple containing the height and width of the image.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
            row represents a possible resolution (height, width).

    Returns:
        Optional[List[int]]: The best upscaling resolution [height, width] from possible_resolutions
            that allows for the smallest upscaling. Returns None if no upscaling is possible.

    Example:
        >>> image_size = (200, 300)
        >>> possible_resolutions = torch.tensor([[224, 672],
        ...                                     [672, 224],
        ...                                     [224, 448],
        ...                                     [448, 224],
        ...                                     [224, 224]])
        >>> _get_smallest_upscaling_possibility(image_size, possible_resolutions)
        [224, 448]
        We have:
            scale_w = tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
            scale_h = tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
            scales = tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])
        Only one of the resolutions allows for upscaling:
            upscaling_possible = tensor([1.1200, 1.1200])
            smallest_rescale = tensor(1.1200)
        So we pick the resolution with the smallest smallest area:
            areas = tensor([150528, 100352]) # [672, 224], [224, 448]
            optimal_canvas = tensor([224, 448])
    """

    original_height, original_width = image_size
    target_heights, target_widths = (
        possible_resolutions[:, 0],
        possible_resolutions[:, 1],
    )

    scale_w = target_widths / original_width
    scale_h = target_heights / original_height

    # get the min scale between width and height (limiting side)
    scales = torch.where(scale_w > scale_h, scale_h, scale_w)

    # keep only scales that allow upscaling
    upscaling_possible = scales[scales > 1]

    if len(upscaling_possible) == 0:
        return None

    # get the one that requires the least upscaling
    min_scale_idx = torch.argmin(upscaling_possible)
    smallest_rescale = upscaling_possible[min_scale_idx]

    # if there are multiple resolutions with the same max scale,
    # we pick the one with the lowest area
    max_scale_idx = torch.where(scales == smallest_rescale)[0]
    if len(max_scale_idx) > 1:
        areas = (
            possible_resolutions[max_scale_idx, 0]
            * possible_resolutions[max_scale_idx, 1]
        )
        optimal_idx = torch.argmin(areas)
        optimal_canvas = possible_resolutions[max_scale_idx[optimal_idx]]
    else:
        optimal_canvas = possible_resolutions[max_scale_idx[0]]

    return tuple(optimal_canvas.tolist())


def _find_supported_resolutions(
    max_num_tiles: int, tile_size: int
) -> List[Tuple[int, int]]:
    """
    Computes all of the allowed resoltuions for a fixed number of tiles
    and tile_size. Useful for when dividing an image into tiles.

    Args:
        max_num_tiles (int): Maximum number of tiles for processing.
        tile_size (int): Size of the side of the tile.

    Returns:
        List[Tuple[int, int]]: List of possible resolutions as tuples (height, width).

    Example:
        >>> max_num_tiles = 5
        >>> tile_size = 224
        >>> find_supported_resolutions(max_num_tiles, tile_size)
        [(224, 896), (448, 448), (224, 224), (896, 224), (224, 672), (672, 224), (224, 448), (448, 224)]

        Given max_num_tiles=4, tile_size=224, it will create a dictionary:
        {
        0.25: [(1, 4)],
        1.0: [(2, 2), (1, 1)],
        4.0: [(4, 1)],
        0.33: [(1, 3)],
        3.0: [(3, 1)],
        0.5: [(1, 2)],
        2.0: [(2, 1)]
        }

        and return the resolutions multiplied by the tile_size:
        [(1*224, 4*224), (2*224, 2*224), ..., (2*224, 1*224)]
    """
    asp_dict = defaultdict(list)
    for tile_size in range(max_num_tiles, 0, -1):
        factors = sorted(_get_factors(tile_size))
        asp_ratios = [(factor, tile_size // factor) for factor in factors]
        for height, width in asp_ratios:
            ratio_float = height / width
            asp_dict[ratio_float].append((height, width))

    # get the resolutions multiplied by the tile_size
    possible_resolutions = []
    for key, value in asp_dict.items():
        for height, depth in value:
            possible_resolutions.append((height * tile_size, depth * tile_size))

    return possible_resolutions


def _get_factors(n: int) -> Set[int]:
    """
    Calculate all factors of a given number, i.e. a dividor that leaves no remainder.
    For example, if n=12, it will return {1, 2, 3, 4, 6, 12}.

    Args:
        n (int): The number to find factors for.

    Returns:
        set: A set containing all factors of the number.
    """
    factors_set = set()

    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors_set.add(i)
            factors_set.add(n // i)
    return factors_set
