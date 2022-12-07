import os
import numpy as np



if __name__ == "__main__":
    for item in os.listdir("dataset/"):
        print(
            np.load("dataset/" + item)
        )
        input()