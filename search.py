# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Stack = LIFO = DFS
    frontier = util.Stack()
    start_state = problem.getStartState()
    frontier.push((start_state, []))

    # Need to keep track of visited nodes for all searches
    visited = set()

    while not frontier.isEmpty():
        # Pop a node if there are any left
        state, path = frontier.pop()

        # Done?
        if problem.isGoalState(state):
            return path

        # Keep track of where we have been
        if state not in visited:
            visited.add(state)

            # Get succ of current state
            successors = problem.getSuccessors(state)

            for successor, action, _ in successors:
                # Push succ -> stack
                frontier.push((successor, path + [action]))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Same as DFS but w Q
    frontier = util.Queue()
    frontier.push((problem.getStartState(),[]))

    visited = set()

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            successors = problem.getSuccessors(state)

            for successor, action, _ in successors:
                frontier.push((successor, path + [action]))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # PQueue for the frontier (priority is cumm. cost)
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), []), 0)  # init cost to 0

    visited = set()

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            successors = problem.getSuccessors(state)

            for successor, action, stepCost in successors:
                # need to calc new cost
                new_cost = problem.getCostOfActions(path + [action])

                if successor not in visited:
                    # push succ onto the pqueue w cost
                    frontier.push((successor, path + [action]), new_cost)
                elif frontier.update((successor, path + [action]), new_cost):
                    # Update pqueue
                    frontier.push((successor, path + [action]), new_cost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Create a priority queue for the frontier with priority as cumulative cost
    frontier = PriorityQueue()
    start_state = problem.getStartState()

    # 0 is the starting cost, and we can also initialize the heuristic cost here
    frontier.push((start_state, []), heuristic(start_state, problem))

    # Again, keep track of visited nodes in a set
    visited = set()

    while not frontier.isEmpty():
        # Dequeue a node with the lowest cost (path + heuristic)
        state, path = frontier.pop()

        # If the state is the goal, return the path we took
        if problem.isGoalState(state):
            return path

        # If state has not been visited, mark it as such
        if state not in visited:
            visited.add(state)

            # For each successor of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    # Calculate new path and the cost from the start node to this successor
                    new_path = path + [action]
                    new_cost = problem.getCostOfActions(new_path) + heuristic(successor, problem)

                    # Add the new node to the frontier with the calculated cost
                    frontier.update((successor, new_path), new_cost)

    # If there is no solution, return an empty list
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
