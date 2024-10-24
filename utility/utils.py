# Functions and other utility items for trash_detector.

def calculate_distance(known_width, focal_length, per_width):
    """
    Calculate the distance to the object using the formula:
    Distance = (Known Width * Focal Length) / Perceived Width
    """
    return (known_width * focal_length) / per_width
