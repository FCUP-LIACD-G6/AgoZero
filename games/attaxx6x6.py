import datetime
import pathlib
import numpy
import random
import torch
RAY_DEDUP_LOGS=0
from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        # 3 channels (empty,Red,Blue)
        self.observation_shape = (3, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        board_size = 6
        # All the "possible" moves
        action_space = [((i, j), (x, y)) for i in range(board_size) for j in range(board_size)
                        for x in range(max(0, i-2), min(board_size, i+3))
                        for y in range(max(0, j-2), min(board_size, j+3))
                        if (i, j) != (x, y)]
        board_len = len(action_space)
        self.action_space = list(range(board_len))
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):

    # Init Attaxx env
    def __init__(self, seed=None):
        self.env = Attaxx()

    # Apply action to the game.
    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    # Current player
    def to_play(self):
        return self.env.to_play()
    
    # All the legal actions for the current player
    def legal_actions(self):
        legal_actions=self.env.legal_actions(self.to_play())
        return legal_actions
    
    def legal_actions_source(self, player):
        return self.env.legal_actions_source(player)
    
    def legal_actions_dest(self, source):
        return self.env.legal_actions_dest(source)

    # Reset the game for a new game
    def reset(self):
        return self.env.reset()

    # Text UI
    def render(self):
        self.env.render()
        #input("Press enter to take a step ")

    # From input (a1) to action (source,dest)
    def human_to_action(self):
        player = self.to_play()
        if player == -1: pl = 'B'
        if player == 1: pl = 'R'

        while True:
            source = input(f"Enter the source cell (e.g: a1) for the player {pl}: ")
            col = ord(source[0].lower()) - ord('a')
            row = int(source[1:]) - 1
            source = (row, col)
            destination = input(f"Enter the destination cell: ")
            col = ord(destination[0].lower()) - ord('a')
            row = int(destination[1:]) - 1
            destination = (row, col)

            source = tuple(source)
            destination = tuple(destination)

            if (source in self.legal_actions_source(player) and destination in self.legal_actions_dest(source)):
                action = (source,destination)
                # Encode action (source,dest) to move range(540)
                return self.env.encode(action)
            else:
                print("Invalid move. Please try again.")

    # Multiplayer bot (doesn't influence training)
    def expert_agent(self):
        return self.env.expert_action()

    # Display move range(540) in action (source,dest) form
    def action_to_string(self, action_int):
        action = self.env.decode(action_int)
        source = action[0]
        dest = action[1]
        cellsource = f"{chr(source[1] + ord('a'))}{source[0] + 1}"
        celldest = f"{chr(dest[1] + ord('a'))}{dest[0] + 1}"
        return f"Move from {cellsource} to {celldest}"



class Attaxx:
    # Init the board and players
    def __init__(self):
        self.size = 6
        self.board = numpy.zeros((6, 6), dtype="int32")
        self.board[0, 0] = 1
        self.board[-1, -1] = 1
        self.board[0, -1] = -1
        self.board[-1, 0] = -1
        self.player = -1

    # Reset the env
    def reset(self):
        self.size = 6
        self.board = numpy.zeros((6, 6), dtype="int32")
        self.board[0, 0] = 1
        self.board[-1, -1] = 1
        self.board[0, -1] = -1
        self.board[-1, 0] = -1
        self.player = -1
        return self.get_observation()

    # Current player
    def to_play(self):
        return 1 if self.player == 1 else -1

    # Game iteration
    def step(self, action_int):
        # Decode from move range(540) to action (source,dest)
        action = self.decode(action_int)

        if not action or not isinstance(action, tuple) or len(action) != 2:
            print("Warning: Invalid action received:",action)
            return self.get_observation(), 0, False
        
        source = action[0]
        dest = action[1]
        distance = max(abs(source[0] - dest[0]), abs(source[1] - dest[1]))

        # Spread Move
        if distance == 1:
            self.board[dest[0]][dest[1]] = self.player
            self.update_board(dest)
        # Jump Move
        if distance == 2:
            self.board[source[0]][source[1]] = 0
            self.board[dest[0]][dest[1]] = self.player
            self.update_board(dest)
        
        done = self.game_over()
        reward = 1 if self.have_winner() else 0
        self.player *= -1
        return self.get_observation(), reward, done
    
    # Encodes action (source,destination) into move range(540)
    def encode(self, action):
        board_size = 6
        action_space=[((i, j), (x, y)) for i in range(board_size) for j in range(board_size)
                        for x in range(max(0, i-2), min(board_size, i+3))
                        for y in range(max(0, j-2), min(board_size, j+3))
                        if (i, j) != (x, y)]
        return action_space.index(action)
    
    # Decodes move range(540) into action (source,destination)
    def decode(self, action_int):
        board_size = 6
        action_space=[((i, j), (x, y)) for i in range(board_size) for j in range(board_size)
                        for x in range(max(0, i-2), min(board_size, i+3))
                        for y in range(max(0, j-2), min(board_size, j+3))
                        if (i, j) != (x, y)]
        return action_space[action_int]
    
    # Updates the surrounding pieces of the destination spot
    def update_board(self, dest):
        player = self.player
        rows, cols = self.board.shape

        # Define the eight possible directions for surrounding pieces
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for direction in directions:
            neighbor_x, neighbor_y = dest[0] + direction[0], dest[1] + direction[1]
            if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                if self.board[neighbor_x, neighbor_y] == -player:
                    self.board[neighbor_x, neighbor_y] = player

    # Verifies if the game has ended
    def game_over(self):
        if self.legal_actions(1) == None or self.legal_actions(-1) == None: return True
        if self.empty_squares() == 0: return True
        if self.player_pieces(1) == 0 or self.player_pieces(-1) == 0: return True
        if self.player_spots(1) == 0 or self.player_pieces(-1) == 0: return True
        return False
    
    # Number of empty squares
    def empty_squares(self):
        rows, cols = self.board.shape
        return len([(i, j) for i in range(rows) for j in range(cols) if self.board[i, j] == 0])
    
    # Number of player pieces
    def player_pieces(self, player):
        rows, cols = self.board.shape
        return len([(i, j) for i in range(rows) for j in range(cols) if self.board[i, j] == player])
    
    # Number of spots player can move
    def player_spots(self, player):
        rows, cols = self.board.shape
        player_spots_count = 0

        for i in range(rows):
            for j in range(cols):
                if self.board[i, j] == player:
                    # Check spots in range 1 or 2 from the player piece
                    for x in range(max(0, i - 2), min(rows, i + 3)):
                        for y in range(max(0, j - 2), min(cols, j + 3)):
                            if self.board[x, y] == 0:
                                player_spots_count += 1
        return player_spots_count

    # 3D channel of the game observations
    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_empty = numpy.where(self.board == 0, 1.0, 0.0)
        return numpy.array([board_player1, board_player2, board_empty])
    
    # Gets all the legal actions (source,dest) for a player
    def legal_actions(self, player):
        actions = []
        sources = self.legal_actions_source(player)

        for source in sources:
            destinations = self.legal_actions_dest(source)
            for destination in destinations:
                action = (source,destination)
                # Encode action (source,dest) into move range(540)
                actions.append(self.encode(action))
        return actions

    # Gets player's legal source spots
    def legal_actions_source(self, player):
        rows, cols = self.board.shape
        return [(i, j) for i in range(rows) for j in range(cols) if self.board[i, j] == player]
    
    # Gets the spots to where the source piece can be moved
    def legal_actions_dest(self, source):
        rows, cols = self.board.shape
        legal_destinations = []

        for i in range(rows):
            for j in range(cols):
                if self.board[i, j] == 0 and self.is_valid_distance(source, (i, j)):
                    legal_destinations.append((i, j))
        return legal_destinations

    # True if destination and source spots are dist <=2
    def is_valid_distance(self, source, destination):
        distance = max(abs(source[0] - destination[0]), abs(source[1] - destination[1]))
        return distance <= 2

    # Checks if game has a winner
    def have_winner(self):
        if self.game_over() and self.player_pieces(self.player) > self.player_pieces(-self.player):
            return True
        return False

    # Bot action    
    def expert_action(self):
        # Initialize with random source and destination as fallbacks
        source = tuple(random.choice(self.legal_actions_source(self.player)))
        legal_moves = self.legal_actions_dest(source)
        
        # Retry if no legal moves found for the initial source
        while not legal_moves:
            source = tuple(random.choice(self.legal_actions_source(self.player)))
            legal_moves = self.legal_actions_dest(source)

        dest = tuple(random.choice(legal_moves))
        action = (source,dest)
        return self.encode(action)
    
    # Text UI
    def render(self):
        # Print column indices
        print('   ', end='')
        for i in range(self.size):
            print(chr(ord('a') + i), end=' ')
        print()
        # Print rows with row indices and board content
        for i in range(self.size):
            print(f'{i + 1:2d} ', end='')
            for j in range(self.size):
                stone = self.board[i][j]
                if stone == 1: stone = 'R'
                if stone == -1: stone = 'B'
                if stone == 0: stone = ' '
                print(f'{stone}', end=' ')
            print()
