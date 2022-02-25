import cv2 as cv
import matplotlib.pyplot as pp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import trange


class PoisonBlender:
    MAX_INTENSITY = 255

    def __init__(self, source, target, mask, delta):
        self.source = source
        self.CHANNEL_SIZE = self.source.shape[2]

        self.target = target
        self.height, self.width, _ = self.target.shape

        self.mask = mask
        self.delta_x, self.delta_y = delta

    @staticmethod
    def get_helper_matrix(height, width):
        block = sp.lil_matrix((width, width))
        PoisonBlender.laplacian_taylor_approximation_block(block)
        A = sp.block_diag([block] * height).tolil()
        PoisonBlender.set_semi_main_diameter(A, width)
        return A

    def get_blended_channel(self, A, channel, f_laplacian, flatten_mask):
        flatten_target = self.target[:self.height, :self.width, channel].flatten()
        flatten_source = self.source[:self.height, :self.width, channel].flatten()
        b = self.calculate_b_vector(flatten_mask, f_laplacian, flatten_source, flatten_target)
        f = spsolve(A, b).reshape((self.height, self.width))
        f = PoisonBlender.outlier_intensities_correction(f).astype('uint8')
        return f

    @staticmethod
    def flatten2rectangular(matrix, height, width):
        return matrix.reshape((height, width))

    def translate(self, input, x, y):
        translation_matrix = np.float32([[1, 0, x],
                                         [0, 1, y]])

        return cv.warpAffine(input, translation_matrix, (self.width, self.height))

    @staticmethod
    def calculate_b_vector(flatten_mask, laplacian, source_flat, target_flat):
        b = laplacian.dot(source_flat)
        b[flatten_mask == 0] = target_flat[flatten_mask == 0]
        return b

    @staticmethod
    def outlier_intensities_correction(matrix):
        matrix[matrix <= 0] = 0
        matrix[matrix >= PoisonBlender.MAX_INTENSITY] = PoisonBlender.MAX_INTENSITY
        return matrix

    @staticmethod
    def set_semi_main_diameter(A, width):
        A.setdiag(-1, width)
        A.setdiag(-1, -width)

    @staticmethod
    def laplacian_taylor_approximation_block(block):
        # laplacian in x,y coordinate = 4f(x,y)-f(x-1,y)-f(x+1,y)-f(x,y-1)-f(x,y+1)
        block.setdiag(-1, -1)
        block.setdiag(-1, 1)
        block.setdiag(4)

    def set_out_pixel(self, coefficient_matrix, counter):
        coefficient_matrix[counter, counter + self.width] = 0
        coefficient_matrix[counter, counter - self.width] = 0
        coefficient_matrix[counter, counter + 1] = 0
        coefficient_matrix[counter, counter - 1] = 0
        # set to identity (out of mask region)
        coefficient_matrix[counter, counter] = 1

    def blend(self):
        # translate source image
        self.source = self.translate(self.source, self.delta_x, self.delta_y)
        # binary mask
        self.mask[:self.height, :self.width][self.mask != 0] = 1
        # initiate coefficient matrix in Af=b
        A = self.get_helper_matrix(self.height, self.width)
        f_l = A.tocsc()

        self.set_out_submatrix_to_identity(A)
        A = A.tocsc()

        # calculate f from Af=b based on flatten matrices
        flatten_mask = self.mask.flatten()
        for channel in trange(self.CHANNEL_SIZE):
            print('channel ' + str(channel) + ' blended')
            self.target[:self.height, :self.width, channel] = self.get_blended_channel(A,
                                                                                       channel,
                                                                                       f_l,
                                                                                       flatten_mask)

        return self.target

    def set_out_submatrix_to_identity(self, coefficient_matrix):
        # set to identity submatrix in out of mask region
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if self.mask[row, col] == 0:
                    counter = col + row * self.width
                    # set to zero/one
                    self.set_out_pixel(coefficient_matrix, counter)


class PolygonMaker:
    def __init__(self, points, mask_shape):
        self.points = points
        h, w, _ = mask_shape
        self.mask = np.zeros((h, w), np.uint8)

    class ClickHandler:
        image = None
        POINTS_SIZE = 0

        def __init__(self, image, window_name):
            self.image = image.copy()
            self.window_name = window_name
            cv.imshow(self.window_name, image)

            h, w, _ = self.image.shape
            self.counter = 0
            self.points = []
            print('clicked vertices of polygon:')

        def get_points(self):
            return np.array([[x, y] for x, y in self.points], np.int)

        def click_event(self, event, clicked_x, clicked_y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                print(clicked_x, clicked_y)
                point = np.array([clicked_x, clicked_y])
                cv.imshow(self.window_name, self.image)
                self.points.append(point)

    def get_filled_polygon(self):
        return cv.fillConvexPoly(self.mask, self.points, 255)


class PolygonMover:
    MAX_INTENSITY = 255

    def __init__(self, target, polygon, window_name):
        self.window_name = window_name
        self.target = target
        self.target_copy = self.target.copy()

        h, w, _ = self.target.shape
        self.polygon = np.zeros((h, w), np.uint8)
        self.polygon[np.where(polygon != 0)] = 255

        self.polygon_copy = self.polygon.copy()

        self.set_initial_parameters()

    @staticmethod
    def translate(input, delta_x, delta_y):
        translation_matrix = np.float32([[1, 0, delta_x],
                                         [0, 1, delta_y]])

        h, w = input.shape
        return cv.warpAffine(input, translation_matrix, (w, h))

    def mouse_moving_handler(self, event, current_x, current_y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            self.left_button_released()

        elif event == cv.EVENT_LBUTTONDOWN:
            self.left_button_pressed(current_x, current_y)

        elif event == cv.EVENT_MOUSEMOVE:
            self.mouse_moving(current_x, current_y)

    def set_initial_parameters(self):
        self.mouse_is_moving = False
        self.is_first_click = True
        self.x_before_move, self.y_before_move = 0, 0
        self.x_last_move, self.y_last_move = 0, 0

    def left_button_released(self):
        self.mouse_is_moving = False

    def mouse_moving(self, current_x, current_y):
        if self.mouse_is_moving:
            delta_x = current_x - self.x_last_move
            delta_y = current_y - self.y_last_move
            self.polygon_copy = PolygonMover.translate(self.polygon_copy, delta_x, delta_y)
            self.x_last_move, self.y_last_move = current_x, current_y

    def left_button_pressed(self, current_x, current_y):
        self.mouse_is_moving = True
        self.x_last_move, self.y_last_move = current_x, current_y
        if self.is_first_click:
            self.x_before_move, self.y_before_move = current_x, current_y
            self.is_first_click = False

    def get_current_frame(self):
        current_frame = self.target.copy()
        current_frame[self.polygon_copy != 0] = 255
        return current_frame

    def get_moved_polygon(self):
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name,
                            self.mouse_moving_handler)

        while True:
            current_frame = self.get_current_frame()
            cv.imshow(self.window_name, current_frame)

            pressed_key = cv.waitKey(1)
            if pressed_key == ord('c'):
                break

        cv.destroyAllWindows()

        total_delta_x, total_delta_y = self.x_last_move - self.x_before_move, self.y_last_move - self.y_before_move
        return (total_delta_x, total_delta_y), self.polygon_copy


def scale():
    # scale image
    path = 'source8.jpg'
    source = cv.imread(path)
    source = cv.resize(source, (0, 0), fx=.4, fy=.4)
    cv.imwrite(path, source)


def main():
    source_path = 'source8.jpg'
    target_path = 'target1.jpg'
    result_path = 'res8.jpg'

    # read source and target image
    source = cv.imread(source_path)
    target = cv.imread(target_path)

    # initiate polygon maker (mouse click handler)
    first_window_name = 'Create polygon (press any key to continue)'
    polygon_maker = PolygonMaker.ClickHandler(source, first_window_name)
    cv.setMouseCallback(first_window_name, polygon_maker.click_event)
    cv.waitKey(0)
    cv.destroyWindow(first_window_name)
    print('polygon created')

    # initiate polygon mover (mouse move handler)
    filled_polygon = PolygonMaker(polygon_maker.get_points(), source.shape).get_filled_polygon()
    second_window_name = 'Move polygon to create mask (press c to continue)'
    delta, moved_polygon = PolygonMover(target, filled_polygon, second_window_name).get_moved_polygon()
    print('polygon moved')
    
    # blend moved polygon and target
    print('blending started:')
    poisson_blend_result = PoisonBlender(source,
                                         target,
                                         moved_polygon,
                                         delta).blend()

    # save
    cv.imwrite(result_path, poisson_blend_result)


if __name__ == '__main__':
    main()
