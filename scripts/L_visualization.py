import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arrow
import os
import imageio
import re


class Visualization:

    def __init__(self, environment, mode="HRI"):

        self.environment = environment
        self.height = environment.height
        self.width = environment.width
        self.walls = np.reshape(
            environment.walls, (self.height, self.width, 4))

        self.robot_start = environment.pos
        self.human_start = environment.human_pos

        self.robot_previous_loc = self.robot_start
        self.human_previous_loc = self.human_start

        self.mode = mode  # either social, navigation or HRI

        self.linewidth = 3
        self.side = 1
        self.radius_circle = 0.3

        self.x_legend = 1
        self.y_legend = 0.9
        
        self.x_writings = self.x_legend+0.02
        
        self.x_description = self.x_writings
        self.y_description = 0.05
        
        self.x_states = self.x_writings
        self.y_states = self.y_legend-0.20

        self.x_action = self.x_legend + 0.04
        self.y_action = self.y_legend - 0.40

        self.x_req_dir = 4.75 * self.width / 4
        self.y_req_dir = 3*self.height / 4

        self.x_text_direction = self.x_writings
        self.y_text_direction = self.y_legend-0.50

        self.x_text_action = self.x_writings
        self.y_text_action = self.y_action+0.1

        self.arrow_components = 1.5
        self.arrow_length = np.sqrt(2 * self.arrow_components ** 2, dtype=np.float32)

        self.human_color = "blue"
        self.robot_color = "red"

        # map from agent orientation to increment dx, dy of the arrow displayed
        self.orientation_map = ((self.arrow_length, 0),
                                (self.arrow_components, -self.arrow_components),
                                (0, -self.arrow_length),
                                (-self.arrow_components, -self.arrow_components),
                                (-self.arrow_length, 0),
                                (-self.arrow_components, self.arrow_components),
                                (0, self.arrow_length),
                                (self.arrow_components, self.arrow_components))
        # action number to action name
        self.action_names = {0: "Up",
                             1: "Down",
                             2: "Left",
                             3: "Right",
                             4: "Up x2",
                             5: "Down x2",
                             6: "Left x2",
                             7: "Right x2",
                             8: "Up x3",
                             9: "Down x3",
                             10: "Left x3",
                             11: "Right x3",
                             12: "Up-Left ",
                             13: "Up-Right",
                             14: "Down-Left",
                             15: "Down-Right",
                             16: "Up-Left x2",
                             17: "Up-Right x2",
                             18: "Down-Left x2",
                             19: "Down-Right x2",
                             20: "Up-Left x3",
                             21: "Up-Right x3",
                             22: "Down-Left x3",
                             23: "Down-Right x3",
                             24: "Stay",
                             25: "Look Human",
                             26: "Hello!",
                             27: "Come!",
                             28: "Pointing",
                             29: "Go to Human"}

    def one_to_two_D(self, position):
        return (position // self.width, position % self.width)

    """ def two_to_one_D(self, position_2D):
        return position_2D[0] * self.width + position_2D[1]"""

    def get_background(self):
        self.rewards = np.reshape(
            self.environment.rewards, (self.height, self.width))
        fig = plt.figure(dpi=100)
        ax = plt.axes(xlim=(0, self.width), ylim=(self.height, 0))
        ax.axes.set_aspect('equal')
        ax.xaxis.tick_top()
        height_increment = 0
        first_cell = True

        for h in range(self.height):
            row = self.walls[h, :]
            width_increment = 0
            for w in range(self.width):
                state_cell = row[w, :]
                x, y = width_increment, height_increment
                if self.rewards[h, w] and self.mode != "social":
                    hatch_type = '///'
                    if first_cell:
                        label = 'Goal area'
                        first_cell = False
                    else:
                        label = None
                else:
                    hatch_type = None
                    label = None
                rect = Rectangle((x, y), self.side, self.side,
                                 fill=None, edgecolor="lightblue",
                                 hatch=hatch_type, label=label)
                ax.add_patch(rect)

                direction_to_walls = {
                    0: ([x, x + self.side], [y, y]),
                    1: ([x, x + self.side], [y + self.side, y + self.side]),
                    2: ([x, x], [y, y + self.side]),
                    3: ([x + self.side, x + self.side], [y, y + self.side])}

                for direction, wall in direction_to_walls.items():
                    if state_cell[direction] == 1:
                        x_wall, y_wall = wall[0], wall[1]
                        plt.plot(x_wall, y_wall, color='black',
                                 linewidth=self.linewidth)

                width_increment += self.side
            height_increment += self.side

        return fig

    def update_scenario(self, figure, environment, action, trial, step):
        ax = figure.gca()

        human_position = environment.human_pos
        robot_position = environment.pos

        robot_x, robot_y = self.get_x_and_y(robot_position)
        human_x, human_y = self.get_x_and_y(human_position)

        previous_robot_x, previous_robot_y = self.get_x_and_y(
            self.robot_previous_loc)
        previous_human_x, previous_human_y = self.get_x_and_y(
            self.human_previous_loc)

        robot_circle = self.plot_circle(
            robot_position, self.robot_color, label='Robot')
        human_circle = self.plot_circle(
            human_position, self.human_color, label='Human')

        plt.plot([previous_robot_x, robot_x], [previous_robot_y, robot_y],
                 linestyle='dashed', color=self.robot_color,
                 alpha=0.3, linewidth=0.7)

        previous_robot_circle = self.plot_circle(
            self.robot_previous_loc, self.robot_color, 0.3)
        previous_human_circle = self.plot_circle(
            self.human_previous_loc, self.human_color, 0.3)

        ax.add_artist(robot_circle)
        # ax.add_artist(human_circle)
        ax.add_artist(previous_robot_circle)
        # ax.add_artist(previous_human_circle)

        self.robot_previous_loc = robot_position
        self.human_previous_loc = human_position

        action_text = figure.text(self.x_action, self.y_action,
                                  self.action_names[action],
                                  transform=ax.transAxes,
                                  va="center", fontsize=9.25,
                                  bbox=dict(facecolor='none',
                                            edgecolor=self.robot_color,
                                            boxstyle='round,pad=0.5'))

        description = figure.text(self.x_description, self.y_description,
                                  " Trial: " + str(trial) +
                                  "\n Step: " + str(step),
                                  transform=ax.transAxes, va="center")
        action_intro_text = figure.text(self.x_text_action, self.y_text_action,
                                        "Robot action:", va="center",
                                        transform=ax.transAxes)
        action_intro_text.set(clip_on=False), ax.add_artist(action_intro_text)

        if self.mode != "navigation":
            direction_text = figure.text(self.x_text_direction,
                                         self.y_text_direction,
                                         "Goal direction:", va="center",
                                         transform=ax.transAxes)
            direction_text.set(clip_on=False), ax.add_artist(direction_text)

            direction_text = figure.text(self.x_text_direction, 
                                         self.y_text_direction,
                                         "Goal direction:", va="center",
                                         transform=ax.transAxes)
            direction_text.set(clip_on=False), ax.add_artist(direction_text)

            plt.plot([previous_human_x, human_x], [previous_human_y, human_y],
                     linestyle='dashed', color=self.human_color,
                     alpha=0.3, linewidth=0.7)
            ax.add_artist(human_circle)
            ax.add_artist(previous_human_circle)

            robot_orientation = environment.orientation
            human_orientation = environment.human_orientation
            required_direction = environment.required_direction
            distance = environment.distance_for_the_robot
            interaction_state = environment.interaction_state
            robot_arrow = self.plot_arrow(robot_position, robot_orientation)
            human_arrow = self.plot_arrow(human_position, human_orientation)

            dy_req_dir, dx_req_dir = self.orientation_to_increment(
                required_direction)
            required_direction = Arrow(self.x_req_dir, self.y_req_dir,
                                       dx_req_dir, dy_req_dir,
                                       width=1, edgecolor="black", 
                                       facecolor="grey")
            required_direction.set(clip_on=False)
            ax.add_artist(required_direction)

            ax.add_patch(robot_arrow)
            ax.add_patch(human_arrow)
            state_text = figure.text(self.x_states, self.y_states,
                                     " Interaction: " + str(interaction_state) +
                                     "\n Distance: " + str(distance),
                                     transform=ax.transAxes, va="center")

        ax.legend(loc='center left', bbox_to_anchor=(
            self.x_legend, self.y_legend))
        ax.axis('off')

        # ADDED create directory to save results in
        save_dir = './all_imgs/tmp-2D/'
        os.makedirs(save_dir, exist_ok=True) 

        plt.savefig("./all_imgs/tmp-2D/" + "Trial" + str(trial) + "_Step" +
                    str(step) + ".png", bbox_inches='tight')
        plt.close()

        description.set_visible(False)
        robot_circle.set_visible(False)
        human_circle.set_visible(False)
        previous_robot_circle.set_visible(False)
        previous_human_circle.set_visible(False)
        action_text.set_visible(False)

        if self.mode != "navigation":
            robot_arrow.set_visible(False)
            human_arrow.set_visible(False)
            required_direction.set_visible(False)
            state_text.set_visible(False)

    def get_x_and_y(self, position):
        y, x = self.one_to_two_D(position)
        x += self.side / 2
        y += self.side / 2
        return x, y

    def plot_circle(self, position, color, alpha=1, label=None):
        y, x = self.one_to_two_D(position)
        x += self.side / 2
        y += self.side / 2
        circle = Circle((x, y), self.radius_circle, edgecolor="black",
                        facecolor=color, alpha=alpha, label=label)
        return circle

    def plot_arrow(self, position, orientation):
        y, x = self.one_to_two_D(position)
        x += self.side / 2
        y += self.side / 2
        dy, dx = self.orientation_to_increment(orientation)
        arrow_orientation = Arrow(x, y, dx, dy, width=1)
        return arrow_orientation

    def orientation_to_increment(self, orientation):
        dx, dy = self.orientation_map[orientation]
        return dy, dx

    def clean_folder(self, path="./all_imgs/tmp-2D"):
        '''Clean the folder were the temporary 2D images are stored'''
        if os.listdir(path):
            for file_name in os.listdir(path):
                ext = os.path.splitext(file_name)[-1].lower()
                if ext == '.pdf' or ext == '.png':
                    file_path = os.path.join(path, file_name)
                    os.remove(file_path)


def save_gif(images_path="./all_imgs/tmp-2D",
             gif_path="./all_imgs/tmp-gif",
             gif_name="visualization.gif"):
    '''Create a gif out of a folder containing .png images'''
    images = []
    for image_file in sorted(os.listdir(images_path),
                             key=lambda x: tuple(map(int, re.findall('\d+', x)))):
        if image_file.endswith(".png"):
            file_path = os.path.join(images_path, image_file)
            images.append(imageio.imread(file_path))
    gif_folder = os.path.join(os.path.dirname(images_path), gif_path)
    gif_path = os.path.join(gif_folder, gif_name)
    imageio.mimsave(gif_path, images, duration=1)


def save_mp4(gif_name, gif_path="./all_imgs/tmp-gif",
             mp4_path="./all_imgs/tmp-mp4",
             mp4_name="visualization.mp4"):
    '''Create an mp4 video from a gif'''

    gif_file_path = os.path.join(gif_path, gif_name)
    mp4_file_path = os.path.join(mp4_path, mp4_name)
    with imageio.get_reader(gif_file_path) as reader:
        with imageio.get_writer(mp4_file_path, fps=1) as writer:
            for frame in reader:
                writer.append_data(frame)
