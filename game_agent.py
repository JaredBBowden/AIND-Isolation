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
        # Adjust to change the importance of opponent moves
        weight = 2.0

        my_moves = len(game.get_legal_moves(player))
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        return my_moves - (opponent_moves * weight)


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
    # TODO: finish this function!
    raise NotImplementedError

    # FIXME This appear to just be another run of the eavlutation function,
    # so we can test multiple attempts at quantifying.

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
    # TODO: finish this function!
    raise NotImplementedError

    # FIXME As above, test multiple attempts at quantifying.

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
        # I think this is good opportunity to use recursion. While we still have
        # time on the clock, run as deep within the search tree, returning the
        # return the best move and score available.

        # Incremental steps
        # 1. Itterate through all current legal moves
        # 2. Forecast the next moves available to each level.

        # FIXME remove for now
        #if self.time_left() < self.TIMER_THRESHOLD:
        #    raise SearchTimeout()

        # FIXME Should there be a win/lose control flow here?


        # FIXME I'm no longer sure that this is going to be required
        #try:
        #    best_score
        #except NameError:
        #    best_score = float('-inf')
        #    best_move = (-1, -1)

        # All possible legal moves for the current board state

        # FIXME There needs to be some control flow to account for the absence
        # of leagal moves. What do we do in this case? What score do we return?

        legal_moves = game.get_legal_moves()

        # Add control flow to acknowledge the minimizing and maximizing
        # identity of the game state
        node_identity = max if game.active_player == self else min

        # When we get to depth, consider the score of all nodes, and the
        # identity of the node that will make a selection based on these
        # options.
        if depth == 1:
            for move in legal_moves:

                best_score, best_move = node_identity((score, move), (best_score, best_move))

                return best_move

        # Make a dicionary with original moves and bottom moves?

        # Itterate to depth
        for move in legal_moves:

            recursion_move = self.minimax(game.forecast_move(move), depth - 1)

            _, best_move = node_identity((score, move), (best_score, best_move))

        return recursion_move


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

        # Get the current moves we have to work with
        legal_moves = game.get_legal_moves()

        # Add control flow to account for the absence of legal moves
        if not legal_moves:
            return (-1, -1)

        # This is where we should consider our "opening book". Let's
        # incorporate our understanding of the best possible opening moves.
        # We could get much more complicated with this; at present, move to
        # the center square if it's one of our legal moves.

        # Find the center square. Account for the chance that the board
        # could be different sizes. And remember zero indexing!
        if (board.height % 2  == 0) and (board.height % 2  == 0):  # Quick check for symmetry
            center_square = ((board.width / 2), (board.height / 2))

        # If the center square is available as a legal move, return that
        if center_square in legal_moves:
            move = center_square
            return best_move

        # In the case where the center move wasn't available, we have to get
        # a little more complicated. Here, we're going to iterate as
        # deep as we can

        # Work within framework established through the minimax
        # method, above

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # FIXME the overall logic is fairly clear: as long as we have time,
        # and we're below the limit, we want to iterate as deep as we can.

        try:

            # Go as deep as we can with AlphaBeta search until we find the
            # best move or run our of time

            # FIXME I'm a little unclear on the use of depth here. Basically,
            # I'm not really sure if we want to be iterating up or down.
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
        ###############################################################


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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def alpha_beta_search():
            """
            Helper function as framed in AIMA text

            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # FIXME Call the max function find an array of fixed values.

            # return a value within the available state that has a value
            # within the above array.

            # So, I think what we're doing here is saying: from the legal
            # moves we have at our disposal, which of thse moves are
            # match within the array above.

            # I kinda need to understand what this array contains...


        # FIXME The following functions aare basically mirror images,
        # with the exception of min and max

        def alpha_beta_max():
            """
            Helper function, as framed in AIMA text

            """
            # FIXME add more detail to this doc string

            # Terminal state: timer
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Final depth
            if depth == 0:
                return self.score(game, self)

            # Intialize a variable to hold our max value
            v = float("inf")

            # Iterate over legal moves and return values where our value (v)
            # updates to either the current value, or the values returned
            # by alpha_beta_min. Each call goes a level deeper.
            for move in game.get_legal_moves():

                # Forecast a level deeper
                v = max(v, alpha_beta_min(game.forecast_move(move)), depth - 1, alpha, beta))

                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v


        def alpha_beta_min():
            """
            Helper function, as framed in AIMA text

            """
            # FIXME add more detail to this doc string

            # Terminal state: timer (as above)
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Final depth
            if depth == 0:
                return self.score(game, self)


            or move in game.get_legal_moves():

                # Forecast a level deeper
                v = max(v, alpha_beta_max(game.forecast_move(move)), depth - 1, alpha, beta))

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v
