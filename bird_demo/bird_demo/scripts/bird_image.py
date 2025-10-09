import numpy as np
from PIL import Image

def bird_image() -> Image:
    """Create a simple pixel art bird image and generate metacontext for it."""
    # Define the size of the image
    width, height = 20, 20

    # Create a blank white canvas
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define bird colors. This bird is mean to be a chestnut sided warbler
    bird_body = [255, 200, 0]    # Yellow body
    bird_beak = [255, 100, 0]    # Orange beak
    bird_eye = [0, 0, 0]         # Black eye
    bird_wing = [200, 150, 0]    # Darker yellow wing

    # Define radius for body shape
    body_radius_squared = 25  # For checking if pixel is within body circle

    # Draw the bird body (simple oval shape)
    for y in range(8, 15):
        for x in range(5, 15):
            if (x - 10)**2 + (y - 12)**2 < body_radius_squared:
                img_array[y, x] = bird_body

    # Draw the beak
    for y in range(11, 13):
        for x in range(3, 6):
            img_array[y, x] = bird_beak

    # Draw the eye
    img_array[10, 7] = bird_eye

    # Draw the wing
    for y in range(10, 13):
        for x in range(12, 16):
            img_array[y, x] = bird_wing

    # Create the image from the numpy array
    return Image.fromarray(img_array)
