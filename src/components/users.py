import numpy as np

class Users:
    def __init__(self, num_groups=1, num_users=4, x_min=50, y_min=50,\
                 z_min=1.5, x_max=100, y_max=100, z_max=2):
        self.num_groups = num_groups
        self.num_users = num_users
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.generate_users()
    
    def generate_users(self):
        self.locations = np.random.rand(self.K, 3)
        self.locations[:, 0] = self.x_min + self.locations[:, 0] * (self.x_max - self.x_min)
        self.locations[:, 1] = self.y_min + self.locations[:, 1] * (self.y_max - self.y_min)
        self.locations[:, 2] = self.z_min + self.locations[:, 2] * (self.z_max - self.z_min)

    def set_locations(self, new_locations):
        if new_locations.shape != (self.num_users, 3):
            raise ValueError(f"Expected new_locations with shape ({self.K}, 3), got {new_locations.shape}")
        self.locations = new_locations
        