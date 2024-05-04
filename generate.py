import numpy as np
import cv2
import glob
import random

class GeneticPainting:
    def __init__(self, img, shapes, seed=0):
        self.reference = cv2.imread(img)
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.imgHistory = [{"image" : np.zeros((self.reference.shape[0], self.reference.shape[1], 4), np.uint8), "score" : np.inf}]
        tempShapes = [cv2.imread(shape, cv2.IMREAD_UNCHANGED) for shape in shapes]
        #grayscale the shapes
        self.shapes = [cv2.cvtColor(shape, cv2.COLOR_BGR2BGRA) for shape in tempShapes]
        for i in range(len(self.shapes)):
            self.shapes[i][self.shapes[i][:, :, 3] != 0] = (255, 255, 255, 255)


    def generate(self, population_size=100, generations=100):
        population = self.initialize_population(population_size)
        for i in range(generations):
            #sort the population by fitness
            best_candidate = [population[0], self.fitness(self.create_applied_image(population[0]))]
            population = sorted(population, key=lambda x: self.fitness(self.create_applied_image(x)))
            print("Generation", i, "Fitness:" , best_candidate[1])
            #Use the top 20% of the population to create the next generation
            new_population = []
            for j in range(population_size//5):
                for _ in range(5):
                    new_population.append(self.mutate(population[j]))
            if(best_candidate[1] < self.imgHistory[-1]["score"]):
                self.imgHistory.append({"image" : self.create_applied_image(best_candidate[0]), "score" : best_candidate[1]})
                cv2.imwrite(f"output/output{len(self.imgHistory)}.jpg", self.imgHistory[-1]["image"])
            population = new_population                
    #Creates the initial population with the following random parameters: shape, position, size, color, rotation
    def initialize_population(self, population_size):
        population = []
        for i in range(population_size):
            shape = random.choice(self.shapes)
            x = np.random.randint(0, self.reference.shape[1] - shape.shape[1])
            y = np.random.randint(0, self.reference.shape[0] - shape.shape[0])
            scale = np.random.uniform(0.5, 1.5)
            angle = np.random.randint(0, 360)
            
            #color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            #set color to the average color in a 10x10 square in reference at the position
            color = np.mean(self.reference[y:y+10, x:x+10], axis=(0, 1)).astype(np.uint8)
            #make the color opaque
            color = np.append(color, 255)
            
            M = cv2.getRotationMatrix2D((shape.shape[1]//2, shape.shape[0]//2), angle, 1)
            shape = cv2.warpAffine(shape, M, (shape.shape[1], shape.shape[0]))
            shape = cv2.cvtColor(shape, cv2.COLOR_BGR2BGRA)
            shape[shape[:, :, 3] != 0] = color

            population.append({"shape" : shape, "x" : x, "y" : y, "scale":scale})
            
        return population
    
    def create_applied_image(self, candidate):
        img = self.imgHistory[-1]["image"].copy()
        shape = candidate["shape"]
        shape_height, shape_width, _ = shape.shape

        x = candidate["x"]
        y = candidate["y"]
        scale = candidate["scale"]

        shape = cv2.resize(shape, (int(shape_width * scale), int(shape_height * scale)))

        # Determine the region where the shape will be pasted
        x_min = max(0, x)
        x_max = min(img.shape[1], x + shape.shape[1])
        y_min = max(0, y)
        y_max = min(img.shape[0], y + shape.shape[0])

        # Paste the shape onto the image
        img[y_min:y_max, x_min:x_max] = shape[
            y_min - y : y_max - y, x_min - x : x_max - x
        ]

        return img
    
    def fitness(self, candidate):
        return np.sum(np.abs(self.reference[..., :3] - candidate[..., :3]))    
    
    def mutate(self, candidate):
        shape = candidate["shape"]
        x = candidate["x"]
        y = candidate["y"]

        #randomly change the position of the candidate
        x += np.random.randint(-60, 60)
        y += np.random.randint(-60, 60)

        #randomly change the color of the candidate by a bit (non transparent parts)
        #color = shape[shape[:, :, 3] != 0]
        #color = color + np.random.randint(-20, 20, color.shape)
        #color = np.clip(color, 0, 255)

        #if the 10x10 square is outside the image, find a different 10x10 square close by
        while y < 0 or y+10 > self.reference.shape[0] or x < 0 or x+10 > self.reference.shape[1]:
            x = np.random.randint(0, self.reference.shape[1] - shape.shape[1])
            y = np.random.randint(0, self.reference.shape[0] - shape.shape[0])
        color = np.mean(self.reference[y:y+10, x:x+10], axis=(0, 1)).astype(np.uint8)
        color = np.append(color, 255)

        shape[shape[:, :, 3] != 0] = color
        
        #randomly change the size of the candidate
        scale = candidate["scale"] + np.random.uniform(-0.1, 0.1)
        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((shape.shape[1]//2, shape.shape[0]//2), angle, scale)
        shape = cv2.warpAffine(shape, M, (shape.shape[1], shape.shape[0]))

        return {"shape" : shape, "x" : x, "y" : y, "scale":scale}
        
    def test(self):
        
        #test the genetic painting
        self.generate(population_size=100, generations=3000)
        for i, img in enumerate(self.imgHistory):
            #cv2.imwrite(f"output/output_{i}.jpg", img["image"])
            cv2.imshow("Genetic Painting", img["image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    #read in all pngs from the shapes folder
    shapes = glob.glob("shapes/*.png")
    gp = GeneticPainting("test.jpg", shapes)
    gp.test()
