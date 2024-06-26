import random
from PIL import Image, ImageDraw

def generate_random_mask(width, height, num_shapes):
    # Create a new image with a transparent background
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # RGBA: Red, Green, Blue, Alpha
    draw = ImageDraw.Draw(image)

    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'circle'])
        # Randomize coordinates and size
        x1 = random.randint(100, 200)
        y1 = random.randint(100, 200)
        x2 = random.randint(x1, 200)
        y2 = random.randint(y1, 200)
        color = (0, 0, 0, 255)  # Black with full opacity

        # Draw the shapes
        if shape_type == 'rectangle':
            draw.rectangle([100+x1, y1+100, 100+x2, 100+y2], fill=color)
        elif shape_type == 'ellipse':
            draw.ellipse([100+x1, 100+y1, 100+x2, 100+y2], fill=color)
        elif shape_type == 'circle':
            radius = min(x2 - x1, y2 - y1) // 2
            center_x, center_y = (100+x1 +100+ x2) // 2, (100+y1+ 100+ y2) // 2
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=color)

    return image

def main():
    width, height = 400, 400
    num_shapes = 2  # Number of shapes to draw
    for i in range(10):
        image = generate_random_mask(width, height, num_shapes)
        #image.show()
        image.save("shapes/"+str(i)+".png")


if __name__ == "__main__":
    main()
