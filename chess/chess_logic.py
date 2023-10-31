
import chess
import chess.engine

class ChessLogic:

    def __init__(self, engine_path="/app/stockfish/stockfish"):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.board = None
    
    def board_from_dict(self, board_dict):
        board = chess.Board()
        board.clear()
        
        # Mapping the pieces from the provided structure to the python-chess symbols
        piece_mapping = {
            ('White', 'Pawn'): 'P',
            ('White', 'Rook'): 'R',
            ('White', 'Knight'): 'N',
            ('White', 'Bishop'): 'B',
            ('White', 'Queen'): 'Q',
            ('White', 'King'): 'K',
            ('Black', 'Pawn'): 'p',
            ('Black', 'Rook'): 'r',
            ('Black', 'Knight'): 'n',
            ('Black', 'Bishop'): 'b',
            ('Black', 'Queen'): 'q',
            ('Black', 'King'): 'k'
        }
        
        for square, piece_info in board_dict.items():
            if piece_info and piece_info[0]:  # If the square is occupied
                color = piece_info[1]
                piece_type = piece_info[2]
                piece_symbol = piece_mapping[(color, piece_type)]
                board.set_piece_at(chess.SQUARE_NAMES.index(square.lower()), chess.Piece.from_symbol(piece_symbol))

        return board

    def get_best_move(self, current_state, previous_state, turn):
        self.board.turn = chess.WHITE if turn == "White" else chess.BLACK
        
        result = self.engine.play(self.board_from_dict(current_state), chess.engine.Limit(time=2.0))
        move = result.move
        
        is_capture = "capture" if self.board.is_capture(move) else "move"

        return turn  + " " + self.board.piece_at(move.from_square).symbol().upper() + " " + is_capture.capitalize() + "From " + chess.SQUARE_NAMES[move.from_square]  + " to " + chess.SQUARE_NAMES[move.to_square]
    
    def get_last_move(self, current_state, previous_state):
        current_board = self.board_from_dict(current_state)
        prev_board = self.board_from_dict(previous_state)
        
        move = None
        for m in prev_board.legal_moves:
            temp_board = prev_board.copy()
            temp_board.push(m)
            if temp_board == current_board:
                move = m
                break
        
        if move:
            piece = prev_board.piece_at(move.from_square)
            return f"{piece.color.capitalize()} {piece.symbol().upper()} Moved From {chess.SQUARE_NAMES[move.from_square]} to {chess.SQUARE_NAMES[move.to_square]}"
        else:
            return "No move detected"
    
    def is_move_legal(self, move_str, current_state):

        parts = move_str.split()
        color = parts[0].lower()
        piece_symbol = parts[1].lower()
        from_square = chess.SQUARE_NAMES.index(parts[3].lower())
        to_square = chess.SQUARE_NAMES.index(parts[-1].lower())
        
        move = chess.Move(from_square, to_square)

        if move not in self.board.legal_moves:
            print("Python-chess board representation:")
            print(self.board)
            print("Current Turn (True for white, False for black):", self.board.turn)
            print("All legal moves for current turn:", list(self.board.legal_moves))

            raise self.IllegalMoveDone(parts[3],parts[-1],[color,piece_symbol])
        
    
    def get_game_status(self):
        if self.board.is_checkmate():
            return "Checkmate"
        elif self.board.is_check():
            print("Check")
        elif self.board.is_stalemate():
            return "Stalemate"
        elif self.board.is_insufficient_material():
            return "Draw due to insufficient material"
        elif self.board.is_seventyfive_moves():
            return "Draw due to seventy-five moves rule"
        elif self.board.is_fivefold_repetition():
            return "Draw due to fivefold repetition"
        else:
            return "Ongoing"
    
    def board_to_fen(self, board_dict):
        return self.board.fen()

    class IllegalMoveDone(Exception):
        def __init__(self,moved_from,moved_to,piece):
            self.moved_from = moved_from
            self.moved_to = moved_to
            self.piece = piece

        def print_error():
            return "Illegal Move : {self.piece[1]} {self.piece[0]} moved from {self.moved_from.upper()} to {self.moved_to.upper()}"

