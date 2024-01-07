import datetime
import pathlib
RAY_DEDUP_LOGS=0
import numpy
import torch
from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game ### 7x7
        self.observation_shape = (3, 7, 7)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(50))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length - players 0 and 1
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 1  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 80  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1500000  # Total number of training steps (ie weights update according to a batch)
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
        self.num_unroll_steps = 80  # Number of game moves to keep for every batch element
        self.td_steps = 80  # Number of steps in the future to take into account for calculating the target value
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
    # init go game
    def __init__(self, seed=None):
        self.env = Go()

    # Apply action to the game.
    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def player_to_string(self):
        if self.env.current_player == 1:
            player = 'W'
        if self.env.current_player == 2:
            player = 'B'
        return player

    def to_play(self):
        #Return the current player.
        return self.env.to_play()

    def legal_actions(self):
        #Returns the legal actions at each turn.
        return self.env.legal_actions()

    def reset(self):
        #Reset the game for a new game.
        return self.env.reset()

    def render(self):
        #Display the game observation.
        self.env.render()
        #input("Press enter to take a step ")


        ######## mudar
    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action.
        """
        player = self.env.current_player
        if player == 1: player = 'W'
        if player == 2: player = 'B'
        print(f"Current player: {player}")
        move_str = input("Enter your move (e.g., 'a1'), or type 'pass' to pass: ")

        # Encode the move a1 to range(0,81)
        if move_str.lower() == 'pass':
            choice = 49
        else:
            col = ord(move_str[0].lower()) - ord('a')
            row = int(move_str[1:]) - 1
            move = (row, col)
            choice = self.env.encode(move)

        '''while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")'''
        
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action to a string representing the action.
        """ 
        if(action_number == 49):
            return f"pass"
        else:
            action_cords = self.env.decode(action_number)
            row, col = action_cords
            return f'{chr(col + ord("a"))}{row + 1}'


class Go:
    def __init__(self):
        self.size = 7
        self.board = numpy.zeros((self.size, self.size), dtype="int32")
        self.current_player = 2 ### 1W 2B       (black começa)
        self.pass_count = 0
        self.komi = 5.5
        self.ended = False
        self.blackcaptured = 0  #peças que o black capturou
        self.whitecaptured = 0  #peças que o white capturou

    def change_play(self):
        self.current_player = 1 if self.current_player == 2 else 2
        
    def to_play(self):
        return 0 if self.current_player == 2 else 1
    
    def reset(self):
        self.size = 7
        self.board = numpy.zeros((self.size, self.size), dtype="int32")
        self.current_player = 2
        self.pass_count = 0
        self.komi = 5.5
        self.ended = False
        self.blackcaptured = 0
        self.whitecaptured = 0
        return self.get_observation()

    def get_liberties(self, row, col):
        # This function will calculate the liberties of a stone at position (row, col)
        # Liberties are empty spaces adjacent (not diagonally) to a stone or group
        liberties = 0
        
        # Check for empty spaces around the stone
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == 0:
                    liberties += 1
        return liberties
    
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

    def capture_stones(self, group):
        # Remove the stones of the specified group from the board
        counter = 0 
        for row, col in group:
            self.board[row][col] = 0
            counter += 1

        if self.current_player == 2:
            self.blackcaptured += counter
        else: self.whitecaptured += counter
        

    def step(self, action):
        '''
        Input: Action range(0,81)
        Step function
        '''

        move = self.decode(action) # move type (x,y)/pass

        # Pass move
        if move == "pass":
            self.pass_move()
            done = self.ended

        # Normal move
        elif self.is_valid_move(move):      # SE NAO FOR pass
            row, col = move
            self.board[row][col] = self.current_player
            to_remove = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] not in [0, self.current_player]:
                        enemy_group, enemy_liberties = self.find_group(nr, nc)
                        if not enemy_liberties:
                            to_remove.append(enemy_group)
            for group in to_remove:
                self.capture_stones(group)
            # Check if the move is suicidal and revert it if so.
            current_group, current_group_liberties = self.find_group(row, col)
            if not current_group_liberties:
                self.board[row][col] = 0  # Revert move
                print("Suicidal move. Try again.")
                return
            self.pass_count = 0  # Reset pass count to zero
            self.change_play()  # Switch players only after the move is finalized
        else:
            print("Invalid move. Try again.")

        done = self.ended or len(self.legal_actions()) == 1

        if done:
            reward = 1 if self.have_winner() else 0
            #score1,score2 = self.calculate_score()      #score 1 = black, score 2 = white
            #if score1 > score2:
                #print("Black won with score:",score1," vs White with score:",score2)
            #else : print("White won with score:",score2," vs Black with score:",score1)
            #print("White has captured,",self.whitecaptured," vs Black has captured:",self.blackcaptured)
        else: reward = 0

        return self.get_observation(), reward, done

    def is_valid_move(self, move):      #checka se está dentro do dos possiveis
        if move == 'pass':
            return True
        row, col = move
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0:
            return True
        return False
    
    def pass_move(self):
        # Player passes, switch to the other player
        self.change_play()
        # Track the number of consecutive passes
        self.pass_count += 1
        if self.pass_count >= 2:
            self.ended = True
        else: self.ended = False

    '''def check_end(self): #check, if for each action possible its all suicidal move
        
        # Check if the move is suicidal and revert it if so.
        current_group, current_group_liberties = self.find_group(row, col)
        if not current_group_liberties:
            self.board[row][col] = ' '  # Revert move'''
#########################

    def calculate_score(self, scoring_system="territory"):
        black_score, white_score = 0, 0

        if scoring_system == "area":
            # Area scoring: counts the number of points a player's stones occupy and surround
            for row in range(self.size):
                for col in range(self.size):
                    if self.board[row][col] == 1:
                        black_score += 1
                    elif self.board[row][col] == 2:
                        white_score += 1
                     #Check empty intersections surrounded by stones
                    if self.board[row][col] == 0 and self.is_surrounded((row, col), 1):
                        black_score += 1
                    elif self.board[row][col] == 0 and self.is_surrounded((row, col), 2):
                        white_score += 1

        elif scoring_system == "territory":
            # Territory scoring: counts the number of empty points a player's stones surround, together with the number of stones the player captured
            black_score, white_score = self.count_territory_and_prisoners()

        # Add komi to white_score
        white_score += self.komi

        return black_score, white_score

    def count_territory_and_prisoners(self):
        black_territory, white_territory = 0, 0
        black_prisoners, white_prisoners = 0, 0

        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    # Check empty points enclosed by a player's stones
                    if self.is_surrounded((row, col), 1):
                        black_territory += 1
                    elif self.is_surrounded((row, col), 2):
                        white_territory += 1


        return black_territory - self.whitecaptured, white_territory - self.blackcaptured

    def is_surrounded(self, position, player):
        # Helper function to check if a group of stones is surrounded
        visited = set()
        queue = [position]

        while queue:
            row, col = queue.pop()
            if (row, col) in visited:
                continue
            visited.add((row, col))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == 0:
                        return False  # Group is not completely surrounded
                    elif self.board[nr][nc] == player and (nr, nc) not in visited:
                        queue.append((nr, nc))

        return True  # Group is completely surrounded


    def print_score(self, black_score, white_score):
        print(f"Final score - Black: {black_score}, White: {white_score}")

    
    def have_winner(self):
        # Calculate the score and determine the winner
        black_score, white_score = self.calculate_score()
        # Announce the winner
        if black_score > white_score:
            winner = 2
        elif white_score > black_score:
            winner = 1
        else:
            winner = -1
        if winner == self.current_player: return True
        else: return False
    

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == 2, 1, 0)
        board_empty = numpy.where(self.board == 0, 1, 0)
        #print("Observation:\n",numpy.array([board_player1, board_player2, board_empty]))
        return numpy.array([board_player1, board_player2, board_empty],dtype="int32")
    
    def decode(self, action):
        '''
        action range(0,49) to move (x,y)/pass
        '''
        if action == 49 :  # Special case for pass
            return "pass"
        else:
            row = int(action // 7)  # Use integer division to ensure integer result
            col = int(action % 7)
            move = (row, col)
            return move

    def encode(self,move):
        '''
        move (x,y) to action range(0,49)
        '''
        row, col = move
        if move == "pass":
            return 49 #ultima move?
        return row*7 + col      #(0 -> 8),(9->17)8*9 = 72 + 8 = 80

    def legal_actions(self):
        legal =[]
        legal.append(49)        #ação de dar pass
        for row in range(7):
            for col in range(7):
                move = (row, col)
        
                if self.is_valid_move(move):
                    # Check if the move is suicidal (not a legal move)
                    current_group, current_group_liberties = self.find_group(row, col)
                    if current_group_liberties:
                        encoding = self.encode(move)
                        legal.append(encoding)
        #print("açoes legal :)",legal)
        return legal

    def expert_action(self):
        legal_actions = self.legal_actions()
        
        # Filter out suicidal moves
        non_suicidal_actions = [action for action in legal_actions if self.is_non_suicidal(action)]
        
        # If there are non-suicidal actions, choose randomly from them; otherwise, choose randomly from all legal actions
        action = numpy.random.choice(non_suicidal_actions) if non_suicidal_actions else numpy.random.choice(legal_actions)
        
        return action

    def is_non_suicidal(self, action):
        if action == 49:  # Special case for pass
            return True
        
        row, col = self.decode(action)
        # Check if the move is suicidal (not a legal move)
        current_group, current_group_liberties = self.find_group(row, col)
        
        return bool(current_group_liberties)

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
                if stone == 1: stone = 'W'
                if stone == 2: stone = 'B'
                if stone == 0: stone = ' '
                print(f'{stone}', end=' ')
            print()
