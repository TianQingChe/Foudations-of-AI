# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #if all food are eaten, return the score
        totalScore=successorGameState.getScore()
        if successorGameState.getNumFood()==0:
            return totalScore       
     
        #find the closest ghost, if it is scared, more closer distance between it and the 
        #pacman is, greater score will be added to the total score, otherwise, smaller score
        #will be added to the total score
        closestGhost=None   
        min_g_dis=float('inf')
        for g in newGhostStates:
            tempDis=util.manhattanDistance(g.getPosition(), newPos)
            if tempDis<min_g_dis:
                min_g_dis=tempDis
                closestGhost=g
        #if next position is a ghost, decrement the score by 50, which can guarantee a 
        #small enough total score to avoid taking that action        
        if min_g_dis==0:
            totalScore-=50
        else:
            if closestGhost.scaredTimer>0:
                totalScore+=50/min_g_dis
            else:
                totalScore+=(-15/min_g_dis)        
        
        #find the closest food, more closer the distance between it and the pacman is,
        #greater score will be added to the total score
        food_distances=[util.manhattanDistance(newPos, f) for f in newFood.asList()]
        min_f_dis=min(food_distances)
        totalScore+=(10/min_f_dis)
        
        return totalScore
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pacmanActions=gameState.getLegalActions(0)
        max_action=pacmanActions[0]
        ghost_index=1
        initial_depth=0
        max_value=self.minValue(gameState.generateSuccessor(0, max_action),ghost_index,initial_depth)
        for action in pacmanActions:
            nextState=gameState.generateSuccessor(0, action)
            nextValue=self.minValue(nextState,ghost_index,initial_depth)
            if nextValue>max_value:
                max_action=action
                max_value=nextValue
        return max_action
    
    def maxValue(self,gameState,current_depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #if last layer reach the max depth, return evaluation value and
        #stop more exploration        
        if current_depth==self.depth:
            return self.evaluationFunction(gameState)
        values=[]
        for action in gameState.getLegalActions(0):
            nextState=gameState.generateSuccessor(0, action)
            values.append(self.minValue(nextState,1,current_depth))
        return max(values)
    
    #There are multiple ghosts, so to visit each ghost and get its value,
    #it is necessary to keep the agent index
    def minValue(self,gameState,agentIndex,current_depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        #if last layer reach the max depth, return evaluation value and
        #stop more exploration
        if current_depth==self.depth:
            return self.evaluationFunction(gameState)      
        value=float('inf')
        for action in gameState.getLegalActions(agentIndex):
            nextState=gameState.generateSuccessor(agentIndex, action)
            if agentIndex<gameState.getNumAgents()-1:
                value=min(value,self.minValue(nextState,agentIndex+1,current_depth))
            else:
                #When next layer is pacman, increment the depth by 1, which is the
                #depth of this layer
                value=min(value,self.maxValue(nextState,current_depth+1))
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
          Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"        
        def maxValue(gameState, depth, alpha, beta):     
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)               
            v=-float('inf')
            action=Directions.STOP
            result=[v,action]
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                nextValue = minValue(nextState, depth, 1, alpha, beta)                      
                if nextValue>result[0]:
                    result=[nextValue,action]
                if nextValue>beta:
                    return [nextValue,action]
                alpha=max(alpha,nextValue)
            return result
                
        def minValue(gameState, depth, agent, alpha, beta):
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)                
            v=float('inf')             
            for action in gameState.getLegalActions(agent):
                nextState = gameState.generateSuccessor(agent, action)
                if agent<gameState.getNumAgents()-1:
                    nextValue = minValue(nextState, depth, agent+1,alpha, beta)
                    if nextValue<v:
                        v=nextValue
                    if nextValue<alpha:
                        return nextValue
                    beta=min(beta,nextValue)
                #the agent is the last ghost, its next layer is max layer for pacman
                else:
                    nextValue= maxValue(nextState, depth+1,alpha, beta)
                    if type(nextValue) is list:
                        nextValue=nextValue[0]                         
                    if nextValue<v:
                        v=nextValue
                    if nextValue<alpha:
                        return nextValue
                    beta=min(beta,nextValue)                      
            return v
                 
        result = maxValue(gameState, 0, -float("inf"), float("inf"))
        if type(result) is list:
            return result[1]
        else:
            return result
                
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):     
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)               
            v=-float('inf')
            action=Directions.STOP
            result=[v,action]
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                nextValue = expValue(nextState, depth, 1)                      
                if nextValue>result[0]:
                    result=[nextValue,action]
            return result
                
        def expValue(gameState, depth, agent):
            if (depth==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)                
            v=0
            actions=gameState.getLegalActions(agent)
            #Each action has the same probability, in this way, ghost can act
            #in a random manner
            actionProb=1.0/len(actions)               
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                if agent<gameState.getNumAgents()-1:
                    nextValue = expValue(nextState, depth, agent+1)
                    v+=actionProb*nextValue
                else:
                    nextValue= maxValue(nextState, depth+1)
                    if type(nextValue) is list:
                        nextValue=nextValue[0]     
                    v+=actionProb*nextValue
            return v
                 
        result = maxValue(gameState, 0)
        if type(result) is list:
            return result[1]
        else:
            return result

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    totalScore=currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()   
    
    #if all food are eaten, return the score
    totalScore=currentGameState.getScore()
    if currentGameState.getNumFood()==0:
        return totalScore        
    
    #find the closest food, more closer the distance between it and the pacman is,
    #greater score will be added to the total score    
    food_distances=[util.manhattanDistance(pos, f) for f in food.asList()]
    min_f_dis=min(food_distances)
    if min_f_dis==0:
        min_f_dis=1
    totalScore+=(10/min_f_dis)
 
    #find the closest ghost, if it is scared, more closer distance between it and the 
    #pacman is, greater score will be added to the total score, otherwise, smaller score
    #will be added to the total score       
    closestGhost=None   
    min_g_dis=float('inf')
    for g in ghostStates:
        tempDis=util.manhattanDistance(g.getPosition(), pos)
        if tempDis<min_g_dis:
            min_g_dis=tempDis
            closestGhost=g

    #if the current position is a ghost, the game ends, so just return current score 
    if min_g_dis==0:
        return currentGameState.getScore()
    if closestGhost.scaredTimer>0:
        totalScore+=50/min_g_dis
    else:
        totalScore+=(-15/min_g_dis)
        
    return totalScore
    
    

# Abbreviation
better = betterEvaluationFunction

