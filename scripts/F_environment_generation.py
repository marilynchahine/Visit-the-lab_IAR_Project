import numpy as np


class Lab_structure:

    def __init__(self):

        print("I AM USED AFTERALL Lab_structure")
        
        # General structure
        self.total_width = np.random.randint(20, 21, dtype=np.int32)
        self.total_height = np.random.randint(20, 21, dtype=np.int32)
        self.number_rooms_up = np.random.randint(4, 5, dtype=np.int32)
        self.number_rooms_down = np.random.randint(4, 5, dtype=np.int32)
        self.door_width = 2

        self.height_each_room = self.list_with_variability(
            [3, 2, 3], self.total_height)
        self.width_rooms_up = self.list_with_variability(
            [2] * self.number_rooms_up, self.total_width)
        self.width_rooms_down = self.list_with_variability(
            [2] * self.number_rooms_down, self.total_width)

        self.doors_up = self.add_horizontal_doors(self.width_rooms_up)
        self.doors_down = self.add_horizontal_doors(self.width_rooms_down)
        self.doors_vertical_up = self.add_vertical_doors(
            self.height_each_room[0], 0.2, self.number_rooms_up)
        self.doors_vertical_down = self.add_vertical_doors(
            self.height_each_room[2], 0.2, self.number_rooms_down)

        self.height_each_room = np.cumsum(self.height_each_room, dtype=np.float32)
        self.width_rooms_up = np.cumsum(self.width_rooms_up, dtype=np.float32)
        self.width_rooms_down = np.cumsum(self.width_rooms_down, dtype=np.float32)

        # Walls and doors
        self.walls = np.zeros((self.total_height, self.total_width, 4), dtype=np.float32)
        self.get_horizontal_walls()  # Add horizontal walls on self.walls
        self.get_horizontal_doors()
        self.get_vertical_walls()
        self.get_vertical_doors()

        # Room labels
        self.labels = np.zeros((self.total_height, self.total_width), dtype=np.float32)

        self.labels_up = np.searchsorted(
            self.width_rooms_up, np.arange(
                1, self.total_width + 1, dtype=np.float32)) + 1
        self.labels_down = np.searchsorted(
            self.width_rooms_down, np.arange(
                1, self.total_width + 1, dtype=np.float32)) + self.number_rooms_up + 1

        self.labels[:self.height_each_room[0]:, :] += self.labels_up
        self.labels[self.height_each_room[1]:, :] += self.labels_down

        # Cartesian into 1D positions
        self.walls_1D = self.walls.reshape(
            self.total_height * self.total_width, 4)
        self.labels_1D = self.labels.flatten()

    def list_with_variability(self, original_list, number_to_reach):
        '''For a given list of ints of sum n and a number m>=n, randomly 
        add one to the list until the sum of the list is m. Change the list
        in place.'''
        number_to_add = number_to_reach - sum(original_list)

        if number_to_add > 0:
            for i in range(number_to_add):
                index_bonus = np.random.randint(len(original_list), dtype=np.int32)
                original_list[index_bonus] += 1

        return original_list

    def add_horizontal_doors(self, list_of_intersection):
        counter, doors = 0, []
        for room_length in list_of_intersection:
            random_number = np.random.randint(
                room_length - self.door_width + 1, dtype=np.int32)
            doors.append(random_number + counter)
            counter += room_length
        return doors

    def add_vertical_doors(self, height_level, threshold, number_rooms):
        all_doors = []
        for wall_number in range(number_rooms - 1):
            cond1 = np.random.random() < (threshold * height_level)
            cond2 = (height_level > self.door_width)
            is_there_a_door = int(cond1 and cond2)
            door_position = np.random.randint(
                height_level - self.door_width + 1, dtype=np.int32) + 1
            all_doors.append(door_position * is_there_a_door - 1)
        return all_doors

    def get_horizontal_walls(self):
        self.walls[0, :, 0] = 1
        for index, height in enumerate(self.height_each_room):
            if index == 2:  # bottom wall
                self.walls[height - 1, :, 1] = 1
            else:
                self.walls[height - 1, :, 1] = 1
                self.walls[height, :, 0] = 1

    def get_horizontal_doors(self):  # door_width parameter added ! :-D
        for position in self.doors_up:
            # Open the path from top to bottom (DOWN) so the robot can cross
            # the door
            self.walls[self.height_each_room[0] - 1,
                       position:position + self.door_width, 1] = 0
            # Open the path from bottom to top (UP) so the robot can cross the door
            self.walls[self.height_each_room[0],
                       position:position + self.door_width, 0] = 0
        for position in self.doors_down:
            self.walls[self.height_each_room[1] - 1,
                       position:position + self.door_width, 1] = 0
            self.walls[self.height_each_room[1],
                       position:position + self.door_width, 0] = 0

    def get_vertical_walls(self):
        self.walls[:, 0, 2] = 1  # left wall
        self.walls[:, self.total_width - 1, 3] = 1  # right wall
        for width in self.width_rooms_up[:-1]:
            # Adding walls only on the upper part
            for height in range(self.height_each_room[0]):
                self.walls[height, width - 1, 3] = 1
                self.walls[height, width, 2] = 1
        for width in self.width_rooms_down[:-1]:
            # Adding walls only on the bottom part
            for height in range(self.height_each_room[1],
                                self.height_each_room[2]):
                self.walls[height, width - 1, 3] = 1
                self.walls[height, width, 2] = 1

    def get_vertical_doors(self):
        for width, height in zip(self.width_rooms_up, self.doors_vertical_up):
            if height != -1:
                self.walls[height:height + self.door_width, width - 1, 3] = 0
                self.walls[height:height + self.door_width, width, 2] = 0
        for width, height in zip(self.width_rooms_down, self.doors_vertical_down):
            if height != -1:

                self.walls[self.total_height - height - self.door_width:
                           self.total_height - height, width, 2] = 0
                self.walls[self.total_height - height - self.door_width:
                           self.total_height - height, width - 1, 3] = 0
