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
    #print(ChessBoard.current_pieces)
    print(move)
    moved_from = move[0]
    moved_to = move[1]
    current_state, prev_state = ChessBoard.get_states()
    moving_piece_type = prev_state[moved_from][2]
    moving_piece_color = prev_state[moved_from][1]
    if moving_piece_type is None or moving_piece_color is None:
        print("ONE OF TWO WAS NONE")
        import pdb;pdb.set_trace();
    return moving_piece_color + " " + moving_piece_type + " at " + moved_from + " moved to " + moved_to


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



    
