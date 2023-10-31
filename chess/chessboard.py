import cv2
import numpy as np
import image_processing as ip
import utilites as util

KEEP_FOR_LATER = {
    'A1': ['Rook', 'White'], 'B1': ['Knight', 'White'], 'C1': ['Bishop', 'White'],
    'D1': ['Queen', 'White'], 'E1': ['King', 'White'], 'F1': ['Bishop', 'White'],
    'G1': ['Knight', 'White'], 'H1': ['Rook', 'White'],
    'A2': ['Pawn', 'White'], 'B2': ['Pawn', 'White'], 'C2': ['Pawn', 'White'],
    'D2': ['Pawn', 'White'], 'E2': ['Pawn', 'White'], 'F2': ['Pawn', 'White'],
    'G2': ['Pawn', 'White'], 'H2': ['Pawn', 'White'],
    
    'A7': ['Pawn', 'Black'], 'B7': ['Pawn', 'Black'], 'C7': ['Pawn', 'Black'],
    'D7': ['Pawn', 'Black'], 'E7': ['Pawn', 'Black'], 'F7': ['Pawn', 'Black'],
    'G7': ['Pawn', 'Black'], 'H7': ['Pawn', 'Black'],
    'A8': ['Rook', 'Black'], 'B8': ['Knight', 'Black'], 'C8': ['Bishop', 'Black'],
    'D8': ['Queen', 'Black'], 'E8': ['King', 'Black'], 'F8': ['Bishop', 'Black'],
    'G8': ['Knight', 'Black'], 'H8': ['Rook', 'Black']
}

class Chessboard:
    def __init__(self,frame, image_processor):
        self.squares = {}  # List of square coordinates
        self.squares_histograms = {}  # Histogram or pixel values for each square
        self.empty_squares_histogram = {} # Histogram or pixel values for each empty square
        self.current_pieces = {}  # Representation of the current pieces on the board
        self.previous_pieces = {} # Representation of the previous pieces on the board
        self.detection_threshold = 0.12 # The change threshold to consider different piece
        self.empty_threshold = 0.1 # The theshold to consider a square is empty
        self.ip = image_processor
        self.warped_img = None
        self.stability_counter = 0
        self.stability_threshold = 5
        self.stability_hist = None 
        
        self.ip.initial_detect() # Define Board Wrapping Parameters

        # Defines corners for squares construction
        self.warped_img = self.ip.get_warped()
        
        corners = get_chessboard_corners(self.warped_img)
        outter_coreners = extrapolate_outer_corners_direct(corners)
        edge_corners = add_four_outermost_corners(corners)
        corners = np.vstack((corners, outter_coreners, edge_corners))
        corners = ip.y_based_order_corners(corners)
        self.squares = construct_squares_array(corners)

        for square_name in self.squares:
            self.current_pieces[square_name] = None

        self.define_initial_position() ## defines the initial position of the game.

        self.previous_pieces = self.current_pieces.copy()


    def define_initial_position(self):
        """
        Define the initial position of pieces on the chessboard.
        """
        # Set all squares to be initially empty
        for square in self.current_pieces.keys():
            self.current_pieces[square] = [False, None, None]

        # Define the classic initial positions
        initial_positions = {
        'A1': ['Rook', 'White'], 'B1': ['Knight', 'White'], 'C1': ['Bishop', 'White'],
        'D1': ['Queen', 'White'], 'E1': ['King', 'White'], 'F1': ['Bishop', 'White'],
        'G1': ['Knight', 'White'], 'H1': ['Rook', 'White'],
        'A2': ['Pawn', 'White'], 'B2': ['Pawn', 'White'], 'C2': ['Pawn', 'White'],
        'D2': ['Pawn', 'White'], 'E2': ['Pawn', 'White'], 'F2': ['Pawn', 'White'],
        'G2': ['Pawn', 'White'], 'H2': ['Pawn', 'White'],
        
        'A7': ['Pawn', 'Black'], 'B7': ['Pawn', 'Black'], 'C7': ['Pawn', 'Black'],
        'D7': ['Pawn', 'Black'], 'E7': ['Pawn', 'Black'], 'F7': ['Pawn', 'Black'],
        'G7': ['Pawn', 'Black'], 'H7': ['Pawn', 'Black'],
        'A8': ['Rook', 'Black'], 'B8': ['Knight', 'Black'], 'C8': ['Bishop', 'Black'],
        'D8': ['Queen', 'Black'], 'E8': ['King', 'Black'], 'F8': ['Bishop', 'Black'],
    'G8': ['Knight', 'Black'], 'H8': ['Rook', 'Black']
        }

        for square, (piece, color) in initial_positions.items():
            self.current_pieces[square] = [True, color, piece]



    def capture_empty_board(self):
        # Captures the state of the empty board
        # Initializes squares_histograms with the histograms of each square
        self.empty_squares_histogram = ip.detect_circles(self.ip, self.squares, self.warped_img)
        self.squares_histograms = ip.detect_circles(self.ip, self.squares, self.warped_img)

    def detect_changes(self,robot_turn = False):
        """
        Compares the current state of the board with the last known state using circle detection and color determination.
        """

        new_img = self.ip.get_warped()
        new_histogram = ip.detect_circles(self.ip, self.squares, new_img)
        new_moves = []  # Returns a list of squares that have changed
        moved_to = None
        moved_from = None
        color_changed_square = None
        
        print(self.stability_counter)

        if self.stability_hist is None:
            self.stability_hist = new_histogram
            return new_moves

        for key in self.stability_hist.keys():
    
            if self.stability_hist[key][0] != new_histogram[key][0]:
                self.stability_counter = 0
                self.stability_hist = new_histogram
                return new_moves
            
        if self.stability_counter < self.stability_threshold:
            self.stability_counter += 1
            self.stability_hist = new_histogram
            return new_moves

        self.stability_counter = 0
        self.stability_hist = new_histogram        
        print("I AM STABLE AS FUCK BOI")
        empty_squares_diff = [square for square, state in new_histogram.items() if not state[0] and self.squares_histograms[square][0]]
        occupied_squares_diff = [square for square, state in new_histogram.items() if state[0] and not self.squares_histograms[square][0]]

        # Detect piece capture scenario
        if len(empty_squares_diff) == 1 and not occupied_squares_diff:
            # Check for color change in squares that have pieces in both states
            for square, curr_state in new_histogram.items():
                prev_state = self.squares_histograms.get(square, [False, None, None])
                if curr_state[0] and prev_state[0] and curr_state[1] != prev_state[1] and curr_state[1] == self.squares_histograms[empty_squares_diff[0]][1]:
                    color_changed_square = square
                    break

        for square, curr_state in new_histogram.items():
            prev_state = self.squares_histograms.get(square, [False, None, None])
            if curr_state != prev_state:
                if not curr_state[0] and prev_state[0]:
                    moved_from = square
                elif curr_state[0] and (not prev_state[0] or square == color_changed_square):
                    moved_to = square

        if (moved_to is None and moved_from is not None) or (moved_from is None and moved_to is not None):
            moved_to, moved_from = None, None

        if moved_from:
            new_moves.append(moved_from)
        if moved_to:
            new_moves.append(moved_to)
          
        self.squares_histograms = new_histogram.copy()

        return new_moves



    def recognize_square(self, piece):
        # Recognizes the piece on a given square based on its histogram
        # Returns the identified piece (e.g., 'white pawn')
        for square,my_piece in self.current_pieces:
            if my_piece == piece:
                return square
        return None

    def update_board_state(self, moves):
        # Updates the board state based on the provided moves
        if len(moves) != 2:
            return

        moving_piece_color = self.current_pieces[moves[0].capitalize()][1]
        moving_piece_type = self.current_pieces[moves[0].capitalize()][2]

        self.previous_pieces = self.current_pieces.copy()

        # Update the moved piece's new position
        self.current_pieces[moves[1]] = [True, moving_piece_color, moving_piece_type]
        # Set the old position to empty
        self.current_pieces[moves[0]] = [False, None, None]

        print(f"{moving_piece_color} {moving_piece_type} at {moves[0]} moved to {moves[1]}.")



    def get_pieces_positions(self):
        # Returns a dict with the pieces on the board and their positions
        pieces = {}
        for square,piece in self.current_pieces:
            if piece is None:
                continue
            pieces[piece] = square
        return pieces
    
    def get_states(self):
        return self.current_pieces.copy(), self.previous_pieces.copy()



# Function for chessboard corner detection
def get_chessboard_corners(image, pattern_size=(7,7)):
    gray = image if image is not None and len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        return corners
    return None

def extrapolate_outer_corners_direct(corners):
    horizontal_dist = corners[1][0] - corners[0][0]
    vertical_dist = corners[7][0] - corners[0][0]
    
    def non_negative(point):
        return [max(0, point[0]), max(0, point[1])]
    
    top_row = [non_negative(corner[0] - [0, vertical_dist[1]]) for corner in corners[0:7]]
    bottom_row = [non_negative(corner[0] + [0, vertical_dist[1]* 0.9]) for corner in corners[42:49]]
    left_col = [non_negative(corner[0] - [horizontal_dist[0], 0]) for corner in corners[0:49:7]]
    right_col = [non_negative(corner[0] + [horizontal_dist[0], 0]) for corner in corners[6:49:7]]
    
    outer_corners = top_row + bottom_row + left_col + right_col
    return np.array(outer_corners).reshape(-1,1,2)

def add_four_outermost_corners(corners):
    horizontal_dist = corners[1][0] - corners[0][0]
    vertical_dist = corners[7][0] - corners[0][0]
    # Helper function to ensure non-negative coordinates
    def non_negative(point):
        return [max(0, point[0]), max(0, point[1])]
    
    top_left_corner = non_negative(corners[0][0] - [horizontal_dist[0], vertical_dist[1]])
    top_right_corner = non_negative(corners[6][0] + [horizontal_dist[0], -vertical_dist[1]])
    bottom_left_corner = non_negative(corners[42][0] - [horizontal_dist[0], -vertical_dist[1]* 0.9])
    bottom_right_corner = non_negative(corners[48][0] + [horizontal_dist[0], vertical_dist[1]* 0.9])
    
    return np.array([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]).reshape(-1,1,2)

def construct_squares_array(corners):
    squares = {}
    labels = ["H", "G", "F", "E", "D", "C", "B", "A"]


    # Iterate over rows and columns of the chessboard
    for row in range(8):
        for col in range(8):
            # Each square is defined by 4 consecutive corners
            top_left = corners[row * 9 + col]
            top_right = corners[row * 9 + col + 1]
            bottom_left = corners[(row + 1) * 9 + col]
            bottom_right = corners[(row + 1) * 9 + col + 1]
            
            square_label = labels[col] + str(row + 1)
            square_corners = [top_left, top_right, bottom_left, bottom_right]
            squares[square_label] = square_corners

    return squares

