
import chess
import chess.engine

class ChessLogic:

    def __init__(self,UI, engine_path="/opt/homebrew/Cellar/stockfish/16/bin/stockfish"):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.UI = UI
        self.board = None
        self.thinktime = 2.0

    def get_best_move(self, current_state, previous_state, turn):
        self.board.turn = chess.WHITE if turn == "White" else chess.BLACK

        result = self.engine.play(self.board, chess.engine.Limit(time=self.thinktime))
        move = result.move

        is_capture = "capture" if self.board.is_capture(move) else "move"
        if self.board.piece_at(move.from_square) is None:
            self.board = board_from_dict(current_state)
        return turn  + " " + self.board.piece_at(move.from_square).symbol().upper() + " " + is_capture.capitalize() + "From " + chess.SQUARE_NAMES[move.from_square]  + " to " + chess.SQUARE_NAMES[move.to_square]
        
    def en_passant_capture_square(self, move):
        # Convert move from UCI format if necessary
        if isinstance(move, str):
            move = chess.Move.from_uci(move)

        # Check if the move is legal and is an en passant capture
        if move in self.board.legal_moves and self.board.is_en_passant(move):
            # The captured pawn's square is the "to" square of the move, shifted one rank
            captured_square = move.to_square + 8 if self.board.turn == chess.BLACK else move.to_square - 8
            return chess.square_name(captured_square)

        return None
        
    
    def get_en_passant_move_if_available(self, current_state, turn):
        self.board.turn = chess.WHITE if turn == "White" else chess.BLACK

        # Check for en passant moves among legal moves
        for move in self.board.legal_moves:
            if self.board.is_en_passant(move):
                return self.format_move(move, turn)

        # If no en passant move is available, use Stockfish to get the best move
        result = self.engine.play(self.board, chess.engine.Limit(time=2.0))
        return self.format_move(result.move, turn)

    def format_move(self, move, turn):
        is_capture = "capture" if self.board.is_capture(move) else "move"
        piece = self.board.piece_at(move.from_square)
        piece_symbol = piece.symbol().upper() if piece is not None else "Unknown"
        return turn + " " + piece_symbol + " " + is_capture.capitalize() + " from " + chess.SQUARE_NAMES[move.from_square] + " to " + chess.SQUARE_NAMES[move.to_square]
        
        
    def get_castling_move_if_available(self, current_state, turn):
        self.board = board_from_dict(current_state)
        self.board.turn = chess.WHITE if turn == "White" else chess.BLACK

        # Check for castling moves among legal moves
        for move in self.board.legal_moves:
            if self.board.is_castling(move):
                return self.format_move(move, turn)

        # If no castling move is available, use Stockfish to get the best move
        result = self.engine.play(self.board, chess.engine.Limit(time=2.0))
        return self.format_move(result.move, turn)

    def get_last_move(self, current_state, previous_state):
        current_board = board_from_dict(current_state)
        prev_board = board_from_dict(previous_state)
        
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
    

    def is_move_legal(self, moves_list,promotion):
        if not moves_list:
            raise ValueError("Empty moves list")

        # Extract the first move
        from_square_str, to_square_str = moves_list[0:2]
        from_square = chess.SQUARE_NAMES.index(from_square_str.lower())
        to_square = chess.SQUARE_NAMES.index(to_square_str.lower())
        move = chess.Move(from_square, to_square,promotion)

        # Check if the move is legal
        if move not in self.board.legal_moves:
            print("Python-chess board representation:")
            print(self.board)
            print("Current Turn (True for white, False for black):", self.board.turn)
            print("All legal moves for current turn:", list(self.board.legal_moves))
            print(self.board_to_fen())
            raise self.IllegalMoveDone(from_square_str, to_square_str,[])

        return move  # or True, depending on what you want the function to return

    
    def is_capture(self, move):
        from_square, to_square = move
        # Convert square names to chess.Square indices
        from_square_index = chess.SQUARE_NAMES.index(from_square.lower())
        to_square_index = chess.SQUARE_NAMES.index(to_square.lower())

        # Create a move object
        chess_move = chess.Move(from_square_index, to_square_index)

        # Check if the move is legal and is a capture
        if chess_move in self.board.legal_moves and self.board.is_capture(chess_move):
            if not self.board.is_en_passant(chess_move):
                return to_square  # Return the name of the captured square
            else:
                return to_square[0]+str((int(to_square[1])-1))
        return None
        
    def get_game_status(self):
        if self.board.is_checkmate():
            self.UI.add_message("Checkmate")
        elif self.board.is_check():
            self.UI.add_message("Check")
        elif self.board.is_stalemate():
            self.UI.add_message("Stalemate")
        elif self.board.is_insufficient_material():
            self.UI.add_message("Draw due to insufficient material")
        elif self.board.is_seventyfive_moves():
            self.UI.add_message("Draw due to seventy-five moves rule")
        elif self.board.is_fivefold_repetition():
            self.UI.add_message("Draw due to fivefold repetition")
        else:
            return "Ongoing"
    

    def board_to_fen(self):
        return self.board.fen()

    
    def set_engine_diffculty(self,diffculty):
        if diffculty == 1:
            self.engine.configure({"Skill Level": 0})
            self.thinktime = 0.1
        elif diffculty == 2:
            self.engine.configure({"Skill Level": 5})
            self.thinktime = 0.5
        elif diffculty == 3:
            self.engine.configure({"Skill Level": 10})
            self.thinktime = 1.0
        elif diffculty == 4:
            self.engine.configure({"Skill Level": 20})
            self.thinktime = 2.0


    class IllegalMoveDone(Exception):
        def __init__(self,moved_from,moved_to,piece):
            self.moved_from = moved_from
            self.moved_to = moved_to
            self.piece = piece

        def print_error():
            return "Illegal Move : {self.piece[1]} {self.piece[0]} moved from {self.moved_from.upper()} to {self.moved_to.upper()}"


def capture_move_squares_for_piece(chessboard,square):
        square_index = chess.SQUARE_NAMES.index(square.lower())
        board = board_from_dict(chessboard)
        piece = board.piece_at(square_index)
        
        if piece:
            if piece.color == chess.BLACK:
                board.push(chess.Move.null())
        # Filter moves that start from the specified square and are captures
        capture_squares = [chess.square_name(move.to_square).upper() for move in board.legal_moves if move.from_square == square_index and board.is_capture(move)]
        
        return capture_squares if capture_squares else []


def board_from_dict(board_dict):
    board = chess.Board()
    board.clear()

    castling_rights = "KQkq"
    board.set_castling_fen(castling_rights)
    
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

