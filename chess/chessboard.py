import cv2
import numpy as np
import image_processing as ip
import utilites as util
import chess_logic as logic
from GUI import GUI as ui
import pickle

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

filepath = "/Users/ahmadghanayem/Desktop/Technion/Project 2/utils/"

class Chessboard:
    def __init__(self,frame, image_processor, UI):
        self.squares = {}  # List of square coordinates
        self.squares_histograms = {}  # Histogram or pixel values for each square
        self.current_pieces = {}  # Representation of the current pieces on the board
        self.previous_pieces = {} # Representation of the previous pieces on the board
        self.UI = UI

        self.ip = image_processor
        self.warped_img = None

        self.stability_counter = 0
        self.stability_threshold = 3
        self.stability_hist = None 
        
        self.ip.initial_detect() # Define Board Wrapping Parameters

        # Defines corners for squares construction
        self.warped_img = self.ip.get_warped()
        self.load_squares()
        
        if len(self.squares) != 0:
            for square_name in self.squares:
                self.current_pieces[square_name] = None

            self.define_initial_position() ## defines the initial position of the game.
            self.previous_pieces = self.current_pieces.copy()
            
        elif self.warped_img is not None and len(self.squares) == 0:
            corners = get_chessboard_corners(self.warped_img)
            outter_coreners = extrapolate_outer_corners_direct(corners)
            edge_corners = add_four_outermost_corners(corners)
            corners = np.vstack((corners, outter_coreners, edge_corners))
            corners = ip.y_based_order_corners(corners)
            self.squares = construct_squares_array(corners)
            self.save_squares()
            
            for square_name in self.squares:
                self.current_pieces[square_name] = None

            self.define_initial_position() ## defines the initial position of the game.

            self.previous_pieces = self.current_pieces.copy()

    def reinitilize_chessboard(self):
        self.warped_img = self.ip.get_warped()
        
        if self.warped_img is not None:               
            corners = get_chessboard_corners(self.warped_img)
            outter_coreners = extrapolate_outer_corners_direct(corners)
            edge_corners = add_four_outermost_corners(corners)
            corners = np.vstack((corners, outter_coreners, edge_corners))
            corners = ip.y_based_order_corners(corners)
            self.squares = construct_squares_array(corners)
            self.save_squares()

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
        self.squares_histograms = ip.detect_circles(self.ip, self.squares, self.warped_img)

    def detect_changes(self):
        """
        Compares the current state of the board with the last known state using circle detection and color determination.
        """

        new_img = self.ip.get_warped()
        org_img = new_img.copy()
        new_histogram = ip.detect_circles(self.ip, self.squares, new_img)
        new_moves = []  # Returns a list of squares that have changed
        moved_to = None
        moved_from = None
        color_changed_square = None
        promote_to = None
        stab_flag = False

        if self.stability_hist is None:
            self.stability_hist = new_histogram
            return new_moves,promote_to,stab_flag

        for key in self.stability_hist.keys():
    
            if self.stability_hist[key] != new_histogram[key]:
                self.stability_counter = 0
                self.stability_hist = new_histogram
                return new_moves,promote_to,stab_flag
            
        if self.stability_counter < self.stability_threshold:
            self.stability_counter += 1
            self.stability_hist = new_histogram
            return new_moves,promote_to,stab_flag

        self.stability_counter = 0
        self.stability_hist = new_histogram
        stab_flag = True

        empty_squares_diff = [square for square, state in new_histogram.items() if not state and self.squares_histograms[square]]
        occupied_squares_diff = [square for square, state in new_histogram.items() if state and not self.squares_histograms[square]]

        # Check for En Passant move
        if len(empty_squares_diff) == 2 and len(occupied_squares_diff) == 1:
            if self.current_pieces[empty_squares_diff[0]][2] == "Pawn" and \
               self.current_pieces[empty_squares_diff[1]][2] == "Pawn":
                captured_piece = empty_squares_diff[1] if empty_squares_diff[1][0] == occupied_squares_diff[0][0] else empty_squares_diff[0]
                empty_squares_diff.remove(captured_piece)
                self.squares_histograms[captured_piece] = False
                self.current_pieces[captured_piece] = [False,None,None]
                self.UI.add_message(f"En Passant capture occured at {captured_piece} by {self.current_pieces[empty_squares_diff[0]][1]}")

        # Detect piece capture scenario
        if len(empty_squares_diff) == 1 and not occupied_squares_diff:
            # Check for color change in squares that have pieces in both states
            possible_captured_squares = logic.capture_move_squares_for_piece(self.current_pieces,empty_squares_diff[0])
            if len(possible_captured_squares) == 0:
                return new_moves,promote_to,stab_flag
            squares_cords = {}
            for square in possible_captured_squares:
                squares_cords[square] = self.squares[square]
            
            new_hist = ip.extract_squares(org_img,squares_cords)
            old_hist = ip.extract_squares(self.ip.prev_warped,squares_cords)       
   
            differences = []
            for square in possible_captured_squares:
                if square not in new_hist or square not in old_hist:
                    import pdb; pdb.set_trace()
                diff = util.compare_histograms(new_hist[square],old_hist[square])
                differences.append([square,diff])
            max_diff = 0
            max_square = []
            for change in differences:
                if change[1] > max_diff:
                   max_diff = change[1]
                   max_square = change
            self.UI.add_message(f"{empty_squares_diff[0]} captured {max_square[0]} with {max_square[1]}")
            color_changed_square = max_square[0]
                                 

        # Check for castling moves
        if len(empty_squares_diff) == 2 and len(occupied_squares_diff) == 2:
            # White kingside castling
            if 'E1' in empty_squares_diff and 'H1' in empty_squares_diff and \
               'G1' in occupied_squares_diff and 'F1' in occupied_squares_diff:
                new_moves = new_moves + ["E1", "G1", "H1", "F1"]

            # White queenside castling
            elif 'E1' in empty_squares_diff and 'A1' in empty_squares_diff and \
                 'C1' in occupied_squares_diff and 'D1' in occupied_squares_diff:
                new_moves = new_moves + ["E1", "C1", "A1", "D1"]

            # Black kingside castling
            elif 'E8' in empty_squares_diff and 'H8' in empty_squares_diff and \
                 'G8' in occupied_squares_diff and 'F8' in occupied_squares_diff:
                new_moves = new_moves + ["E8", "G8", "H8", "F8"]

            # Black queenside castling
            elif 'E8' in empty_squares_diff and 'A8' in empty_squares_diff and \
                 'C8' in occupied_squares_diff and 'D8' in occupied_squares_diff:
                new_moves = new_moves + ["E8", "C8", "A8", "D8"]

            self.squares_histograms = new_histogram.copy()
            self.ip.prev_warped = org_img
            return new_moves,promote_to,stab_flag

        for square, curr_state in new_histogram.items():
            prev_state = self.squares_histograms[square]
            
            if not curr_state and prev_state:
                moved_from = square
            elif curr_state and (not prev_state or square == color_changed_square):
                moved_to = square

        if (moved_to is None and moved_from is not None) or (moved_from is None and moved_to is not None):
            moved_to, moved_from = None, None

        if moved_from:
            new_moves.append(moved_from)
        if moved_to:
            new_moves.append(moved_to)
          
        if len(new_moves) == 2 and self.current_pieces[new_moves[0]][2] == "Pawn" \
           and (new_moves[1][1] == "1" or new_moves[1][1] == "8"):
            import pdb; pdb.set_trace()
            usr_input = self.UI.promote_pawn()
            if usr_input == "q":
                promote_to = 5
                self.current_pieces[new_moves[0]][2] = "Queen"
            elif usr_input == "r":
                promote_to = 4
                self.current_pieces[new_moves[0]][2] = "Rook"
            elif usr_input == "b":
                promote_to = 3
                self.current_pieces[new_moves[0]][2] = "Bishop"
            elif usr_input == "n":
                promote_to = 2 
                self.current_pieces[new_moves[0]][2] = "Knight"           

        self.squares_histograms = new_histogram.copy()
        self.ip.prev_warped = org_img
        self.UI.display_chessboard_image(util.generate_pieces_string(self.current_pieces))
        return new_moves,promote_to,stab_flag



    def recognize_square(self, piece):
        # Recognizes the piece on a given square based on its histogram
        # Returns the identified piece (e.g., 'white pawn')
        for square,my_piece in self.current_pieces:
            if my_piece == piece:
                return square
        return None

    def update_board_state(self, moves):
        # Updates the board state based on the provided moves
        if len(moves) != 2 and len(moves) != 3 and len(moves) != 4:
            return

        self.previous_pieces = self.current_pieces.copy()
        moving_piece_color = self.current_pieces[moves[0].capitalize()][1]
        moving_piece_type = self.current_pieces[moves[0].capitalize()][2]

        if len(moves) == 2:
            # Update the moved piece's new position
            self.current_pieces[moves[1]] = [True, moving_piece_color, moving_piece_type]
            # Set the old position to empty
            self.current_pieces[moves[0]] = [False, None, None]

            self.UI.add_message(util.convert_move_to_string(self,moves))

        elif len(moves) == 4:
            castling_side = "Kingside" if moves[1] == "G1" or moves[1] == "G8" else "Queenside"
            self.current_pieces[moves[1]] = [True, moving_piece_color, "King"]
            self.current_pieces[moves[0]] = [False, None, None]
            self.current_pieces[moves[3]] = [True, moving_piece_color , "Rook"]
            self.current_pieces[moves[2]] = [False, None, None]
            self.UI.add_message(util.convert_move_to_string(None,moves))
            
        self.UI.display_chessboard_image(util.generate_pieces_string(self.current_pieces))
            


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
        
    def save_squares(self):
        with open(filepath + "_squares.pkl", 'wb') as file:
            pickle.dump(self.squares, file)

    def load_squares(self):
        try:
            with open(filepath + "_squares.pkl", 'rb') as file:
                self.squares = pickle.load(file)
        except FileNotFoundError:
            self.squares = {}

# Function for chessboard corner detection
def get_chessboard_corners(image, pattern_size=(7,7)):
    if image is None: 
        return None
    gray = image if image is not None and len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        return corners
    return None

def extrapolate_outer_corners_direct(corners):

    if corners is None:
        return None
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
    if corners is None:
        return None
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

    if corners is not None:
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

