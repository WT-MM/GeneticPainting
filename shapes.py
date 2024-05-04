import random
from PIL import Image, ImageDraw

def generate_random_mask(width, height, num_shapes):
    # Create a new image with a transparent background
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # RGBA: Red, Green, Blue, Alpha
    draw = ImageDraw.Draw(image)

    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'circle'])
        # Randomize coordinates and size
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(x1, width)
        y2 = random.randint(y1, height)
        color = (0, 0, 0, 255)  # Black with full opacity

        # Draw the shapes
        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == 'circle':
            radius = min(x2 - x1, y2 - y1) // 2
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=color)

    return image

def main():
    width, height = 800, 600
    num_shapes = 3  # Number of shapes to draw
    for i in range(10):
        image = generate_random_mask(width, height, num_shapes)
        #image.show()
        image.save("shapes/"+str(i)+".png")


if __name__ == "__main__":
    main()
