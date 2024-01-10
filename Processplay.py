import pygame
import random as rand
import math
import numpy as np
import models
import torch
import pathlib

class Go:
    def __init__(self,size):        #defenir size do tabuleiro
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype="int32")
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
                self.change_play()  # Switch players only after the move is finalized
            self.pass_count = 0  # Reset pass count to zero
            self.change_play()  # Switch players only after the move is finalized
        else:
            print("Invalid move. Try again.")

        done = self.ended or len(self.legal_actions()) == 1

        if done:
            reward = 1 if self.have_winner() else 0
            score1,score2 = self.calculate_score()      #score 1 = black, score 2 = white
            if score1 > score2:
                print("Black won with score:",score1," vs White with score:",score2)
            else : print("White won with score:",score2," vs Black with score:",score1)
            print("White has captured,",self.whitecaptured," vs Black has captured:",self.blackcaptured)
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
                    if self.is_surrounded((row, col), 2):
                        black_territory += 1
                    elif self.is_surrounded((row, col), 1):
                        white_territory += 1


        return black_territory - self.whitecaptured, white_territory - self.blackcaptured

    def is_surrounded(self, position, player):
        # Helper function to check if a group of stones is surrounded
        visited = []
        queue = [position]

        while queue:
            row, col = queue.pop()
            if (row, col) in visited:
                continue
            visited.append((row, col))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] == 0:
                        queue.append((nr,nc))  # Group is not completely surrounded
                    elif self.board[nr][nc] == player and (nr, nc) not in visited:
                        pass     
                    else :
                        return False #está surrounded pelo outro player

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

class selffplay():
    def __init__(self, initial_checkpoint, seed,board,config):
        self.config = config
        self.board=board
        
        # Fix random generator seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cpu"))
        self.model.eval()
        
        
        
    def play_game(self, temperature, temperature_threshold,ai_player):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.board.get_observation()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        #game_history.to_play_history.append(self.game.to_play())

        done = False

        

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(np.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(np.array(observation).shape)} dimensionnal. Got observation of shape: {np.array(observation).shape}"
                assert (
                    np.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )
    
                # Choose the action
                
                root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.board.legal_actions(ai_player),
                        self.board.to_play(),
                        True,
                )
                action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                )
                done=True
                self.board.step(action,ai_player)
        return game_history
    
    def select_action(self,node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action
    
    
    
# Game independent



class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.observation_history[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    
        
