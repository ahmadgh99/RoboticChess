import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def draw_chess_game(state_string, image_folder):
    # Open the tile images
    tile_a = Image.open(os.path.join(image_folder, "Tile-A.png"))
    tile_b = Image.open(os.path.join(image_folder, "Tile-B.png"))

    # Define the size of a square (assumed to be square)
    square_size = tile_a.size[0]  # assume all tiles are the same size

    # Create a new image for the board with extra space for the border
    board_size = 10 * square_size
    board = Image.new('RGBA', (board_size, board_size))

    # Build the board
    tiles = [tile_a, tile_b]
    for i in range(8):
        for j in range(8):
            tile = tiles[(i+j)%2]
            board.paste(tile, ((i+1)*square_size, (j+1)*square_size))

    # Split the state string into individual pieces
    pieces = state_string.split()

    for i in range(0, len(pieces), 2):
        # Open the piece image
        piece = Image.open(os.path.join(image_folder, pieces[i] + ".png"))
        # Resize the piece to fit the square
        piece = piece.resize((square_size, square_size))

        # Get the position of the piece
        position = pieces[i+1]
        # Convert the position to a 0-based index and adjust for the border
        x = (8 - (ord(position[0].upper()) - ord('A'))) * square_size
        y = (int(position[1])) * square_size

        # Paste the piece onto the board
        board.paste(piece, (x, y), piece)
        
    # Draw the border
    for i in range(1, 9):
        # Top border
        half_tile_a = tile_a.crop((0, square_size/2, square_size, square_size)).rotate(180)
        board.paste(half_tile_a, (i*square_size, int(square_size/2)))  # Adjusted position
        # Bottom border
        half_tile_a = tile_a.crop((0, 0, square_size, square_size/2))
        board.paste(half_tile_a, (i*square_size, board_size-square_size))
        # Left border
        half_tile_a = tile_a.crop((square_size/2, 0, square_size, square_size))
        board.paste(half_tile_a, (int(square_size/2), i*square_size))  # Adjusted position
        # Right border
        half_tile_a = tile_a.crop((0, 0, square_size/2, square_size)).rotate(180)
        board.paste(half_tile_a, (board_size-square_size, i*square_size))

    # Corner tiles
    quarter_tile_a = tile_a.crop((square_size/2, 0, square_size, square_size/2)).rotate(180)
    board.paste(quarter_tile_a, (int(square_size/2), int(square_size/2)))  # Top left adjusted position
    quarter_tile_a = tile_a.crop((0, 0, square_size/2, square_size/2)).rotate(180)
    board.paste(quarter_tile_a, (board_size-square_size, int(square_size/2)))  # Top right adjusted position
    quarter_tile_a = tile_a.crop((square_size/2, square_size/2, square_size, square_size))
    board.paste(quarter_tile_a, (int(square_size/2), board_size-square_size))  # Bottom left adjusted position
    quarter_tile_a = tile_a.crop((0, square_size/2, square_size/2, square_size))
    board.paste(quarter_tile_a, (board_size-square_size, board_size-square_size))  # Bottom right

    # Draw a bold black border around the 8x8 board
    draw = ImageDraw.Draw(board)
    draw.rectangle([(square_size, square_size), (board_size - square_size, board_size - square_size)], outline="black", width=int(square_size/10))

    # Draw labels on the board
    font = ImageFont.load_default()

    # Draw numbers in the left border
    for i in range(8):
        label = str(i+1)
        draw.text((int(square_size) - int(square_size/4), (i+1) * square_size + square_size / 2 - int(square_size * 0.125)), label, fill="black", font=font, anchor="mm")
        draw.text((board_size - int(square_size) + int(square_size/8), (i+1) * square_size + square_size / 2 - int(square_size * 0.125)), label, fill="black", font=font, anchor="mm")

    # Draw letters in the top border
    for i in range(8):
        label = chr(ord('A') + 7 - i)
        draw.text(((i+1) * square_size + square_size / 2 - int(square_size * 0.125), int(square_size) - int(square_size/4)), label, fill="black", font=font, anchor="mm")
        draw.text(((i+1) * square_size + square_size / 2 - int(square_size * 0.125), board_size - int(square_size)), label, fill="black", font=font, anchor="mm")

    # Display the board
    return board
    #board.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw a chess game.')
    parser.add_argument('--state', type=str, required=True, help='The state of the chess game.')
    parser.add_argument('--folder', type=str, required=True, help='The folder containing the images.')

    args = parser.parse_args()

    draw_chess_game(args.state, args.folder)
