from chessboard import Chessboard
import chess_logic as cl
import time
import chess
import utilites as uts

class ChessGame:
    def __init__(self,frame,image_processor,UI):
        self.physical_board = Chessboard(frame,image_processor,UI)
        self.logic = cl.ChessLogic(UI)
        self.robot_turn = False
        self.UI = UI
        self.first_move = True
        self.robot_color = "White"
        self.logic.board = cl.board_from_dict(self.physical_board.current_pieces)
        self.illegal_occured = [False,None,None]

    def initialize_empty_board(self):
        self.physical_board.capture_empty_board()

    def reinitilize_chessboard(self):
        self.physical_board.reinitilize_chessboard()

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
        # Checks the current state of the board against the last known state
        # to determine the move made by the player
        
        arm_inst = ""

        detected_move,promotion,stab = self.physical_board.detect_changes()
        
        if self.first_move and stab:
            arm_inst = self.switch_turn()
            self.first_move = False
            
        if len(detected_move) == 0:
            return arm_inst
        
        current_board_state,previous_board_state = self.physical_board.get_states()
        try:
            self.logic.is_move_legal(detected_move,promotion)
        except Exception as e:
            if self.illegal_occured[0] == False:
                temp_state = current_board_state.copy()
                temp_state[detected_move[1]] = temp_state[detected_move[0]]
                temp_state[detected_move[0]] = [False, None, None]
                self.illegal_occured = [True,detected_move[0],detected_move[1]]
                self.UI.display_wrong_move(detected_move[0],detected_move[1],uts
                .generate_pieces_string(temp_state))
            else:
                if detected_move == self.illegal_occured[1:3][::-1]:
                    self.illegal_occured = [False,None,None]
                    self.UI.display_chessboard_image(uts.generate_pieces_string(current_board_state))
                else:
                    self.UI.add_message("Please fix this and previous illegal moves")
                
        else:
            self.update_game_state(detected_move)
            from_square = chess.SQUARE_NAMES.index(detected_move[0].lower())
            to_square = chess.SQUARE_NAMES.index(detected_move[1].lower())
            move = chess.Move(from_square,to_square,promotion)
            self.logic.soundHandler(detected_move)
            self.logic.board.push(move)
            arm_inst = self.switch_turn()
        finally:
            return arm_inst

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
        cap = self.logic.is_capture(move)
        if move[0] == 'E1' and move[1] in ["C1","G1"] and self.physical_board.current_pieces['E1'][2] == "King":
            if move[1] == 'G1':
                move.append('H1')
                move.append('F1')
            else:
                move.append('A1')
                move.append('D1')
            
        self.UI.add_message(f"best move is: {from_square} to {to_square}, MOVE QUICKLY!")
        return uts.generate_action_string(move,cap)

    def switch_turn(self):

        if self.robot_turn:
            self.robot_turn = False
            return ""
        else:
            self.robot_turn = True
            return self.play_turn()

    def change_difficulty(self,diff):
        self.logic.set_engine_diffculty(diff)
        
    def engine_off(self):
        self.logic.engine.quit()
        
