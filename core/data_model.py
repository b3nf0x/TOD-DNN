import numpy as np
import secrets
import os


class SynData:

    def __init__(self, mass, cf, static_ambient_temp, time_of_death, rectal_temp):
        self.mass = mass
        self.correction_factor = cf
        self.static_ambient_temp = static_ambient_temp
        self.time_of_death = time_of_death
        self.rectal_temp = rectal_temp
        self.file_name = secrets.token_urlsafe(15)

    
    def to_json(self):
        return self.__dict__

    def to_numpy_array(self):
        return np.array([
            self.mass, 
            self.correction_factor, 
            self.static_ambient_temp, 
            self.time_of_death, 
            self.rectal_temp
        ])


    def save_as_npy(self, path):
        np.save(
            os.path.join(path, self.file_name), 
            self.to_numpy_array()
        )

    @staticmethod
    def load_from_file(path):
        npy_data = np.load(path)
        return SynData(
            mass=npy_data[0],
            cf=npy_data[1],
            static_ambient_temp=npy_data[2],
            time_of_death=npy_data[3],
            rectal_temp=npy_data[4]
        )