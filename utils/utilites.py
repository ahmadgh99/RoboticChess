import random
import numpy as np
import cv2


# Compare normalized histograms
def compare_histograms(hist1, hist2):
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)

    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    if hist1.shape != hist2.shape:
        print("Histogram shapes do not match:", hist1.shape, "vs.", hist2.shape)
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    
def compute_middle_points(squares_dict):
    
    for name, square in squares_dict.items():
        top_left, top_right, bottom_left, bottom_right = square
        top_left, _, _, bottom_right = tuple(map(int, top_left[0])), tuple(map(int, top_right[0])), tuple(map(int, bottom_left[0])), tuple(map(int, bottom_right[0]))
        mid_x = (top_left[0] + bottom_right[0]) / 2
        mid_y = (top_left[1] + bottom_right[1]) / 2
        middle_points[name] = (mid_x, mid_y)

    return middle_points


def convert_move_to_string(ChessBoard,move):
    if len(move) == 4:
        castling_string = ""
        if move == ["E1", "G1", "H1", "F1"]:
           castling_string = "Kingside castling for White player occured!"
        elif move == ["E1", "C1", "A1", "D1"]:
           castling_string = "Queenside castling for White player occured!"
        elif move == ["E8", "G8", "H8", "F8"]:
            castling_string = "Kingside castling for Black player occured!"
        elif move == ["E8", "C8", "A8", "D8"]:
            castling_string = "Queenside castling for Black player occured!"
        return castling_string
 
    moved_from = move[0]
    moved_to = move[1]
    current_state, prev_state = ChessBoard.get_states()
    moving_piece_type = prev_state[moved_from][2]
    moving_piece_color = prev_state[moved_from][1]
    if moving_piece_type is None or moving_piece_color is None:
        print("ONE OF TWO WAS NONE")
        import pdb;pdb.set_trace();
    return moving_piece_color + " " + moving_piece_type + " at " + moved_from + " moved to " + moved_to

def are_images_same(image1, image2):
    if image1.shape != image2.shape:
        return False
    difference = np.subtract(image1, image2)
    return not np.any(difference)


def modify_corners(custom_ordered_corners, shift_value=10):
    # Enlarge the polygon by shifting each corner diagonally by shift_value pixels
    custom_ordered_corners[0][0] -= shift_value  # Top-left x
    custom_ordered_corners[0][1] -= shift_value  # Top-left y

    custom_ordered_corners[1][0] += shift_value  # Top-right x
    custom_ordered_corners[1][1] -= shift_value  # Top-right y

    custom_ordered_corners[2][0] += shift_value  # Bottom-right x
    custom_ordered_corners[2][1] += shift_value  # Bottom-right y

    custom_ordered_corners[3][0] -= shift_value  # Bottom-left x
    custom_ordered_corners[3][1] += shift_value  # Bottom-left y
    
    return custom_ordered_corners


def get_input(message):
    user_input = input(f"{message}\n").strip().lower()
    return user_input


def generate_pieces_string(board_dict):
    pieces_string = ""
    for square, data in board_dict.items():
        has_piece, piece_color, piece_name = data
        if has_piece:
            # Assuming piece_color and piece_name are not None when has_piece is True
            pieces_string += f"{piece_color}-{piece_name} {square} "
    return pieces_string.strip()
    
def generate_action_string(square_list, capture_square=None):
    if len(square_list) % 2 != 0:
        raise ValueError("List must contain an even number of elements (pairs of 'from' and 'to' positions)")

    action_strings = []
    
    if capture_square is not None:
        action = f"{capture_square} -> X"
        action_strings.append(action)
        
    for i in range(0, len(square_list), 2):
        from_square = square_list[i]
        to_square = square_list[i + 1]

        action = f"{from_square} -> {to_square}"
        action_strings.append(action)

    return "\n".join(action_strings)


