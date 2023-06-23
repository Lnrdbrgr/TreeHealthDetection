"""Module for utility and helper functions during the pre-processing
of the image data.
"""

def calculate_area(x_min: int, x_max: int, y_min: int, y_max: int) -> int:
    """
    Calculate the area in pixels given the coordinates of a bounding box.

    Args:
        x_min (int):
            Minimum x-coordinate of the bounding box.
        x_max (int):
            Maximum x-coordinate of the bounding box.
        y_min (int):
            Minimum y-coordinate of the bounding box.
        y_max (int):
            Maximum y-coordinate of the bounding box.

    Returns:
        int: The calculated area in pixels.
    """
    area = (x_max - x_min) * (y_max - y_min)
    return area

