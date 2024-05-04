import numpy as np
import cv2
import glob
import random

class GeneticPainting:
    def __init__(self, img, shapes, seed=0):
        self.reference = cv2.imread(img)
        self.seed = seed
        random.seed(self.seed)

        self.curImg = [np.zeros((self.reference.shape[0], self.reference.shape[1], 4), np.uint8)]
        tempShapes = [cv2.imread(shape, cv2.IMREAD_UNCHANGED) for shape in shapes]
        #grayscale the shapes
        self.shapes = [cv2.cvtColor(shape, cv2.COLOR_BGR2BGRA) for shape in tempShapes]
        for i in range(len(self.shapes)):
            self.shapes[i][self.shapes[i][:, :, 3] != 0] = (255, 255, 255, 255)


    #Creates the initial population with the following random parameters: shape, position, size, color, rotation
    def initialize_population(self, population_size):
        population = []
        for i in range(population_size):
            shape = random.choice(self.shapes)
            x = np.random.randint(0, self.reference.shape[1] - shape.shape[1])
            y = np.random.randint(0, self.reference.shape[0] - shape.shape[0])
            scale = np.random.uniform(0.1, 3)
            angle = np.random.randint(0, 360)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            M = cv2.getRotationMatrix2D((shape.shape[1]//2, shape.shape[0]//2), angle, scale)
            shape = cv2.warpAffine(shape, M, (shape.shape[1], shape.shape[0]))
            shape = cv2.cvtColor(shape, cv2.COLOR_BGR2BGRA)
            shape[shape[:, :, 3] != 0] = color
            
            #make sure the total shape size is the same as the reference image (padding if necessary)
            if x + shape.shape[1] > self.reference.shape[1]:
                shape = np.pad(shape, ((0, 0), (0, x + shape.shape[1] - self.reference.shape[1]), (0, 0)), mode='constant', constant_values=0)
            if y + shape.shape[0] > self.reference.shape[0]:
                shape = np.pad(shape, ((0, y + shape.shape[0] - self.reference.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
            population.append(shape)
    
        return population
    
    def test(self):
        #Test initialize_population by showing each candidate
        population = self.initialize_population(10)
        for i in range(len(population)):
            cv2.imshow("Candidate " + str(i), population[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    #read in all pngs from the shapes folder
    shapes = glob.glob("shapes/*.png")
    gp = GeneticPainting("test.jpg", shapes)
    gp.test()
