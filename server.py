import socket
import time
import numpy
import pygame

# Board class
class Board:
    def __init__(self, game, size):
        self.size = size
        self.game = game
        self.pass_count = 0
        self.board = numpy.zeros((size, size), dtype="int32")
        self.initialize_board()

    def initialize_board(self):
        if self.game == "A":
            # Agent 1 is 1Red
            # Agent 2 is -1Blue
            self.board[0, 0] = 1
            self.board[-1, -1] = 1
            self.board[0, -1] = -1
            self.board[-1, 0] = -1
        elif self.game == "G": pass
            # Agent 1 is 2Black
            # Agent 2 is 1White

    def update_board(self, data, player):
        # For Attaxx: "MOVE source[0],source[1],dest[0],dest[1]"
        # For Go: "MOVE row,col"  or  pass
        if self.game == "A":
            source = (int(data[5]),int(data[7]))
            dest = (int(data[9]),int(data[11]))
            distance = max(abs(source[0] - dest[0]), abs(source[1] - dest[1]))

            # Spread Move
            if distance == 1:
                self.board[dest[0]][dest[1]] = player
            # Jump Move
            if distance == 2:
                self.board[source[0]][source[1]] = 0
                self.board[dest[0]][dest[1]] = player

            # Update the board
            rows, cols = self.board.shape
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            for direction in directions:
                neighbor_x, neighbor_y = dest[0] + direction[0], dest[1] + direction[1]
                if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                    if self.board[neighbor_x, neighbor_y] == -player:
                        self.board[neighbor_x, neighbor_y] = player

        elif self.game == "G":
            if data == "pass":
                self.pass_count += 1
            else:
                row = (int(data[5]))
                col = (int(data[7]))
                self.board[row][col] = player
                self.pass_count = 0
                to_remove = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.board[nr][nc] not in [0, player]:
                            enemy_group, enemy_liberties = self.find_group(nr, nc)
                            if not enemy_liberties:
                                to_remove.append(enemy_group)
                for group in to_remove:
                    # Remove the stones of the specified group from the board
                    counter = 0 
                    for row, col in group:
                        self.board[row][col] = 0
                        counter += 1

    def find_group(self, row, col):
        # This function will find all connected stones of the same color and their liberties
        color = self.board[row][col]
        group = set()
        liberties = set()
        queue = [(row, col)]

        while queue:
            r, c = queue.pop()
            if (r, c) in group:
                continue
            group.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == 0:
                        liberties.add((nr, nc))
                    elif self.board[nr][nc] == color and (nr, nc) not in group:
                        queue.append((nr, nc))
        return group, liberties

            

            ###################
            ###################
            ###################
            # Falta as coisas de comer se estiver à volta do outro e tal

    def display_board(self):
        # Size of the board
        rows, cols = self.size, self.size
        # Size of each cell in the Pygame window
        cell_size = 50

        # Initialize Pygame window
        screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))
        pygame.display.set_caption("Server Board")

        screen.fill((211, 211, 211))  # Fill the background with white

        for row in range(rows):
            for col in range(cols):
                # Calculate the position of the cell on the screen
                x = col * cell_size
                y = row * cell_size

                # Draw the cell based on the value in the board matrix
                if self.game == "A":
                    if self.board[row][col] == 1:
                        pygame.draw.rect(screen, (255, 0, 0), (x, y, cell_size, cell_size))
                    elif self.board[row][col] == -1:
                        pygame.draw.rect(screen, (0, 0, 255), (x, y, cell_size, cell_size))
                elif self.game == "G":
                    if self.board[row][col] == 2:
                        pygame.draw.circle(screen, (0, 0, 0), (x + cell_size // 2, y + cell_size // 2), cell_size // 2)
                    elif self.board[row][col] == 1:
                        pygame.draw.circle(screen, (255, 255, 255), (x + cell_size // 2, y + cell_size // 2),
                                           cell_size // 2)

        # Update the display
        pygame.display.flip()



# Starts the board depending on the game (just used for visual purposes)
def start_board(Game):
    if "A" in Game:
        size = int(Game[1])
        return Board("A", size)
    elif "G" in Game:
        size = int(Game[1])
        return Board("G", size)
    else:
        raise ValueError("Invalid game type")

# Main code
def start_server(host='localhost', port=12345):
    # Choose the game to play
    options = [
        "A4x4",
        "A5x5",
        "A6x6",
        "G7x7",
        "G9x9",
    ]
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)
    Game = options[choice]

    # Start server's board
    game_board = start_board(Game)

    # Create server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(2)

    # Wait for agents to connect
    print("Waiting for two agents to connect...")
    agent1, addr1 = server_socket.accept()
    print("Agent 1 connected from", addr1)
    bs=b'AG1 '+Game.encode()
    agent1.sendall(bs)
    agent2, addr2 = server_socket.accept()
    print("Agent 2 connected from", addr2)
    bs=b'AG2 '+Game.encode()
    agent2.sendall(bs)    
    agents = [agent1, agent2]
    current_agent = 0
    jog=0
    
    # Game loop
    while True:
        try:
            # Process the move
            # For Attaxx: "MOVE source[0],source[1],dest[0],dest[1]"
            # For Go: "MOVE row,col"  or  pass
            data = agents[current_agent].recv(1024).decode()
            if not data: break
            print(current_agent, " -> ",data)
            jog = jog+1
            
            # Visual Board
            if "A" in Game:
                if current_agent == 0 : player = 1
                else : player = -1
            if "G" in Game:
                if current_agent == 0 : player = 2
                else : player = 1
            game_board.update_board(data,player)
            game_board.display_board()

            # Stop if game takes too long
            if jog==200: 
                agents[current_agent].sendall(b'END 0 10 10')
                agents[1-current_agent].sendall(b'END 0 10 10')
                break
            
            # Verify if move is valid
            if is_valid_move(data):
                agents[current_agent].sendall(b'VALID')
                agents[1-current_agent].sendall(data.encode())
            else:
                agents[current_agent].sendall(b'INVALID')

            # Switch to the other agent
            current_agent = 1-current_agent
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break

    # End of the game
    print("\n-----------------\nGAME END\n-----------------\n")
    time.sleep(1)
    agent1.close()
    agent2.close()
    server_socket.close()

# Verify if move is valid
def is_valid_move(move):
    # Implement the logic to check if the move is valid
    # G6's agents do not need this since they can only play legal moves
    return True

if __name__ == "__main__":
    start_server()
