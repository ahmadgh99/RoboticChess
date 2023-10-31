from chessboard import Chessboard
from chess_logic import ChessLogic
import time
import chess
import utilites as uts

class ChessGame:
    def __init__(self,frame,image_processor):
        self.physical_board = Chessboard(frame,image_processor)
        self.logic = ChessLogic()
        self.robot_turn = True
        self.robot_color = "White"
        self.logic.board = self.logic.board_from_dict(self.physical_board.current_pieces)

    def initialize_empty_board(self):
        self.physical_board.capture_empty_board()

    def setup_pieces(self,robot_start_first = True):
        # Recognizes the initial placement of pieces
        # This involves comparing the current state of the board with the empty state
        if robot_start_first:
            self.robot_turn == True
            self.robot_color == "White"
        else:
            self.robot_turn == False
            self.robot_color == "Black"

    def detect_player_move(self):
        #print(self.robot_turn)
        # Checks the current state of the board against the last known state
        # to determine the move made by the player

        if self.robot_turn:
            self.play_turn()
           
        detected_move = self.physical_board.detect_changes()
        if len(detected_move) == 0:
            return
        
        print(detected_move)
        move_string = uts.convert_move_to_string(self.physical_board,detected_move)
        piece_color = move_string.split()[0]

        if (self.logic.board.turn == chess.WHITE and piece_color != "White") or (self.logic.board.turn == chess.BLACK and piece_color != "Black"):
            print(f"its not {piece_color}'s turn!!")
            import pdb;pdb.set_trace();
            return
        current_board_state,previous_board_state = self.physical_board.get_states()

        self.logic.is_move_legal(move_string,current_board_state)
        #self.logic.get_last_move(current_board_state,previous_board_state)
        self.update_game_state(detected_move)
        from_square = chess.SQUARE_NAMES.index(detected_move[0].lower())
        to_square = chess.SQUARE_NAMES.index(detected_move[1].lower())
        move = chess.Move(from_square,to_square)
        self.logic.board.push(move)
        self.switch_turn()
        self.logic.get_game_status()

    def update_game_state(self, move):
        # Updates both the Chessboard and GameState with a given move
        self.physical_board.update_board_state(move)

    
    def play_turn(self):

        if not self.robot_turn:
            return
        
        board_states = self.physical_board.get_states()
        result = self.logic.get_best_move(board_states[0],board_states[1],self.robot_color)
        from_square = result.split()[-3].capitalize()
        to_square = result.split()[-1].capitalize()
        move = [from_square,to_square]
        print(f"best move is: {from_square} to {to_square}, MOVE QUICKLY!")
        #time.sleep(5) ## should be changed by wait until the robot finishes moving and placing the piece



    def switch_turn(self):

        if self.robot_turn:
            self.robot_turn = False
        else:
            self.robot_turn = True

        #self.logic.board.turn = chess.BLACK if self.logic.board.turn == chess.WHITE else chess.WHITE
        print(self.logic.board.turn)
        
