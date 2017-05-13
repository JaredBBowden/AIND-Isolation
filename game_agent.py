"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    # FIXME I think we might be able to look at score methods in the player classes
    # for more context on how this should be done.

    # FIXME Check the testing code -- it looks as though the titles for our players
    # could be different here...?

    # Cover the win/lose scenarios
    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    # Score optional moves. In this case, I'm just using a modified version of the
    # weighted my_move - opponent_moves evaluation function covered in lectures.
    else:

        my_moves = len(game.get_legal_moves(player))
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        try:
            return my_moves / opponent_moves
        except:
            return my_moves

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    # Score optional moves. In this case, I'm just using a modified version of the
    # weighted my_move - opponent_moves evaluation function covered in lectures.
    else:

        def center_classification(legal_moves):

            if (3,3) in legal_moves:
                return 5
            else:
                return 0

        my_moves = len(game.get_legal_moves(player))
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        my_center_bonus = center_classification(game.get_legal_moves(player))
        opponent_center_bonus = center_classification(game.get_legal_moves(player))

        return (my_moves + my_center_bonus) - (opponent_moves - opponent_center_bonus)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    # Score optional moves. In this case, I'm just using a modified version of the
    # weighted my_move - opponent_moves evaluation function covered in lectures.
    else:

        def center_classification(legal_moves):

            if (3,3) in legal_moves:
                return 10
            else:
                return 0

        my_moves = len(game.get_legal_moves(player))
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        my_center_bonus = center_classification(game.get_legal_moves(player))
        opponent_center_bonus = center_classification(game.get_legal_moves(player))

        return (my_moves + my_center_bonus) - (opponent_moves - opponent_center_bonus)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Top level moves to evaluate
        legal_moves = game.get_legal_moves()

        # Control flow for situations where we don't have any legal moves.
        if not legal_moves  or len(legal_moves) == 0:
             best_move =  (-1, -1)
             return best_move

        # Initalize some place-holder variables
        best_score = float("-inf")
        best_move = (-1, -1)

        for move in legal_moves:
            temp_score = self.min_value(game.forecast_move(move), depth - 1)

            if temp_score > best_score:

                # Update best scores and moves
                best_score = temp_score
                best_move = move

        return best_move

    def max_value(self, game, depth):
        """

        """
        # FIXME improve docstring here

        # Control flow for terminal condition / final depth
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return self.score(game, self)

        # Initialize a variable to hold our score
        best_score = float("-inf")

        for move in game.get_legal_moves():
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth - 1))

        return best_score

    def min_value(self, game, depth):
        """

        """
        # FIXME improve docstring here

        # TODO this terminal condition code is starting to get repetitious.
        # consider a refactor

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        best_score = float("inf")

        for move in game.get_legal_moves():
            best_score = min(best_score, self.max_value(game.forecast_move(move), depth - 1))

        return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # Timer
        self.time_left = time_left

        """
        # Get the current moves we have to work with
        legal_moves = game.get_legal_moves()

        # Add control flow to account for the absence of legal moves
        if not legal_moves:
            return (-1, -1)

        #################################
        # Is where we should consider our "opening book?
        # Incorporate our understanding of the best possible opening moves.
        # We could get much more complicated with this; at present, move to
        # the center square if it's one of our legal moves.

        # Find the center square. Account for the chance that the board
        # could be different sizes. And remember zero indexing!
        if (board.height % 2  == 0) and (board.height % 2  == 0):  # Quick check for symmetry

            center_square = ((board.width / 2), (board.height / 2))

            # If the center square is available as a legal move, return that
            if center_square in legal_moves:
                best_move = center_square
                return best_mov
        """

        best_move = (-1, -1)

        try:
            # Go as deep as we can with AlphaBeta search until we find the
            # best move or run our of time
            depth = 1

            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # FIXME The following functions are basically mirror images,
        # with the exception of min and max. I think we could work to make
        # these function names a little better.

        # TODO It strikes me that we're using this "terminal state" code in
        # several places. Might be cleaner to refactor and recycle.
        # FIXME this is not the first time I've commented this in here.

        # TODO Are there performance gains that could be realized here
        # by reordering values?

        # FIXME added out of habit, but do we really need this here? This is
        # going to be the first thing we hit -- no recursion.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return self.score(game, self)

        # Intialize empty variables
        best_move = (-1, -1)
        best_score = float("-inf")

        legal_moves = game.get_legal_moves()

        # Control flow to check for legal moves available
        # FIXME Again, I'm a little perplexe by the need to this additional
        # check.
        if not legal_moves or len(legal_moves)==0:
            return best_move

        # In the case where we have moves to work with, search the game tree
        # with alpha beta thresholds.
        for move in legal_moves:

            v = self.alpha_beta_min(game.forecast_move(move), depth - 1, alpha, beta)

            # FIXME clear up documentation in both of these level

            if v > best_score:
                best_score = v
                best_move = move

            if best_score >= beta:
                return move

            alpha = max(alpha, best_score)

        return best_move


    def alpha_beta_max(self, game, depth, alpha, beta):
        """
        Helper function, as framed in AIMA text

        Itterate through the game tree and compute an alpha threshold --
        this will provide

        """
        # FIXME add more detail to this doc string

        # Terminal state: timer
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Final depth
        if depth == 0:
            return self.score(game, self)

        # Intialize a variable to hold our max value
        v = float("-inf")

        # Iterate over legal moves and return values where our value (v)
        # updates to either the current value, or the values returned
        # by alpha_beta_min. Each iteration goes a level deeper.
        for move in game.get_legal_moves():

            # Forecast a level deeper
            v = max(v, self.alpha_beta_min(game.forecast_move(move), depth - 1, alpha, beta))

            if v >= beta:
                return v

            alpha = max(alpha, v)

        return v


    def alpha_beta_min(self, game, depth, alpha, beta):
        """
        Helper function, as framed in AIMA text.

        Itterate through the game tree and compute a beta threshold --
        the minimum UPPER bound.

        """
        # FIXME add more detail to this doc string

        # Terminal state: timer (as above)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Final depth
        if depth == 0:
            return self.score(game, self)

        # Intialize a variable to hold our max value
        v = float("inf")

        for move in game.get_legal_moves():

            # Forecast a level deeper
            v = min(v, self.alpha_beta_max(game.forecast_move(move), depth - 1, alpha, beta))

            if v <= alpha:
                return v

            beta = min(beta, v)

        return v


    # Call the helper functions and return a result
