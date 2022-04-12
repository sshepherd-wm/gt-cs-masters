######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################


import math
import numpy as np
import re
import copy

class DeliveryPlanner_PartA:

    """
    Required methods in this class are:
    
      plan_delivery(self, debug = False) which is stubbed out below.  
        You may not change the method signature as it will be called directly 
        by the autograder but you may modify the internals as needed.
    
      __init__: which is required to initialize the class.  Starter code is 
        provided that intializes class variables based on the definitions in 
        testing_suite_partA.py.  You may choose to use this starter code
        or modify and replace it based on your own solution
    
    The following methods are starter code you may use for part A.  
    However, they are not required and can be replaced with your
    own methods.
    
      _set_initial_state_from(self, warehouse): creates structures based on
          the warehouse and todo definitions and intializes the robot
          location in the warehouse
    
      _search(self, debug=False): Where the bulk of the A* search algorithm
          could reside.  It should find an optimal path from the robot
          location to a goal.  Hint:  you may want to structure this based
          on whether looking for a box or delivering a box.
  
    """

    ## Definitions taken from testing_suite_partA.py
    ORTHOGONAL_MOVE_COST = 2
    DIAGONAL_MOVE_COST = 3
    BOX_LIFT_COST = 4
    BOX_DOWN_COST = 2
    ILLEGAL_MOVE_PENALTY = 100

    def __init__(self, warehouse, todo):
        
        self.todo = todo
        self.boxes_delivered = []
        self.total_cost = 0
        self._set_initial_state_from(warehouse)

        self.delta = [[-1, 0 ], # north
                      [ 0,-1 ], # west
                      [ 1, 0 ], # south
                      [ 0, 1 ], # east
                      [-1,-1 ], # northwest (diag)
                      [-1, 1 ], # northeast (diag)
                      [ 1, 1 ], # southeast (diag)
                      [ 1,-1 ]] # southwest (diag)

        self.delta_directions = ["n","w","s","e","nw","ne","se","sw"]

        # Can use this for a visual debug
        self.delta_name = [ '^', '<', 'v', '>','\\','/','[',']' ]

        # Costs for each move
        self.delta_cost = [  self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST ]

    ## state parsing and initialization function from testing_suite_partA.py
    def _set_initial_state_from(self, warehouse):
        """Set initial state.

        Args:
            warehouse(list(list)): the warehouse map.
        """
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'

                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'

                elif this_square == '@':
                    self.warehouse_state[i][j] = '*'
                    self.dropzone = (i, j)

                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)

        self.robot_position = self.dropzone
        self.box_held = None
    
    def _search(self, goal, start, target, debug=False):
        """
        This method should be based on Udacity Quizes for A*, see Lesson 12, Section 10-12.
        The bulk of the search logic should reside here, should you choose to use this starter code.
        Please condition any printout on the debug flag provided in the argument.  
        You may change this function signature (i.e. add arguments) as 
        necessary, except for the debug argument which must remain with a default of False
        """

        # get a shortcut variable for the warehouse (note this is just a view no copying)
        grid = self.warehouse_state
        # for g in grid:
        #     print(g)
        # print(' ')
        ## Step 1: Create Tracker grid with shortest # of steps to goal for each cell

        ## Initialize Tracker from grid with simplified obstacles
        tracker = copy.deepcopy(grid)
        for r in range(len(tracker)):
            for c in range(len(tracker[0])):
                if tracker[r][c] not in ('.','@','*'):
                    tracker[r][c] = '#'
                else:
                    tracker[r][c] = '.'
                if r == goal[0] and c == goal[1]:
                    tracker[r][c] = 0 ## goal cost is 0

        ## Expand open adjacent cells starting at the goal and update with cost to get to the goal
        pos = goal
        adj = _adjacents(tracker, pos)
        #print(adj)
        cost = 0

        # iterate through adjancents
        while len(adj):
            cost += 1
            adj2 = []
            for a in adj:
                ## if not already explored
                if type(tracker[a[0][0]][a[0][1]]) != int:
                    #print(a, cost)
                    tracker[a[0][0]][a[0][1]] = cost
                    [adj2.append(aa) for aa in _adjacents(tracker, a[0])] ## add this adjacent's adjacents
            adj = adj2[:] ## new adjacents become the main adjacent list

        # for t in tracker:
        #     print([str(tt) for tt in t])
        # print(' ')

        ## Step 2: Search from the starting position taking the shortest step until we've reached the goal
        pos = start
        adj = _adjacents(tracker, pos, excl_ints=False)
        path = [(pos,tracker[pos[0]][pos[1]])] ## initialize search path list
        reached_goal = False
        count_iter = 0

        #print(pos)
        #print(adj)

        while not reached_goal and len(adj) and count_iter < 10000:
            #print(pos)
            min_cost = min([a[1] for a in adj]) ## find lowest cost of the possible steps
            next_step = [a for a in adj if a[1] == min_cost][0] ## take first step that equals the lowest cost
            path.append(next_step) ## add step to search path list
            pos = next_step[0] ## take this step by making the position
            if pos == goal:
                break
            adj = _adjacents(tracker, pos, excl_ints=False)
            count_iter += 1

        ## if not returned, throw exception
        if count_iter + 1 >= 1000:
            raise Exception("passed 10000 iterations with no path found")
            
        #return path, path[-2][0]
        
        ## Step 3: Generate plan instructions
        ## plan variables
        #target = grid[goal[0]][goal[1]]
        plan = []
        delts = []

        ## move to the target
        for i in range(len(path)):
            pos = path[i]
            nxt = path[i + 1]
            delt   = (np.matrix(nxt[0]) - np.matrix(pos[0])).tolist()[0]
            delts.append(delt)
            direct = self.delta_directions[ self.delta.index(delt) ]
            #direct = delta_directions[ delta.index(delt) ]

            ## check if next spot is goal
            if nxt[1] == 0: ## if next to dropzone
                if grid[nxt[0][0]][nxt[0][1]] != target: 
                    move = 'down ' + str(direct)
                else: ## if next to box
                    move = 'lift ' + str(target)
                    grid[nxt[0][0]][nxt[0][1]] = '.' ## mark location as free
                plan.append(move)
                break
            else:
                move = 'move ' + direct
                plan.append(move)
                
        ## return the plan, and the current robot location which will be the second to last position of the path to the goal
        #print(plan)
        return plan, path[-2][0]

  
    def plan_delivery(self, debug = False):
        """
        plan_delivery() is required and will be called by the autograder directly.  
        You may not change the function signature for it.
        Add logic here to find the moves.  You may use the starter code provided above
        in any way you choose, but please condition any printouts on the debug flag
        """

        # Find the moves - you may add arguments and change this logic but please leave
        # the debug flag in place and condition all printouts on it.

        # You may wish to break the task into one-way paths, like this:
        #
        #    moves_to_1   = self._search( ..., debug=debug )
        #    moves_from_1 = self._search( ..., debug=debug )
        #    moves_to_2   = self._search( ..., debug=debug )
        #    moves_from_2 = self._search( ..., debug=debug )
        #    moves        = moves_to_1 + moves_from_1 + moves_to_2 + moves_from_2
        #
        # If you use _search(), you may need to modify it to take some
        # additional arguments for starting location, goal location, and
        # whether to pick up or deliver a box.

        ## get the warehouse grid
        grid = self.warehouse_state

        ## identify the dropzone and goals in the grid
        dropzone, goals = _get_dropzone_goals(grid)

        ## position
        pos = dropzone

        ## moves to return
        moves = []

        ## iterate through target boxes
        for tgt in self.todo:
            goal = [g for g in goals if g[1] == tgt][0][0]
            #print('running for:', tgt, goal)
            ## plan to get to the target
            plan_depart, pos = self._search(goal    , pos, tgt, debug=debug)
            #print('DEPART:', plan_depart, pos)
            ## plan to get back to the dropzone
            plan_return, pos = self._search(dropzone, pos, tgt, debug=debug)
            #print('RETURN:', plan_return, pos)

            ## add moves to plan
            moves += plan_depart + plan_return

        return moves


class DeliveryPlanner_PartB:
    """
    Required methods in this class are:

        plan_delivery(self, debug = False) which is stubbed out below.
        You may not change the method signature as it will be called directly
        by the autograder but you may modify the internals as needed.

        __init__: required to initialize the class.  Starter code is
        provided that intializes class variables based on the definitions in
        testing_suite_partB.py.  You may choose to use this starter code
        or modify and replace it based on your own solution

    The following methods are starter code you may use for part B.
    However, they are not required and can be replaced with your
    own methods.

        _set_initial_state_from(self, warehouse): creates structures based on
            the warehouse and todo definitions and intializes the robot
            location in the warehouse

        _find_policy(self, debug=False): Where the bulk of the dynamic
            programming (DP) search algorithm could reside.  It should find
            an optimal path from the robot location to a goal.
            Hint:  you may want to structure this based
            on whether looking for a box or delivering a box.

    """

    # Definitions taken from testing_suite_partA.py
    ORTHOGONAL_MOVE_COST = 2
    DIAGONAL_MOVE_COST = 3
    BOX_LIFT_COST = 4
    BOX_DOWN_COST = 2
    ILLEGAL_MOVE_PENALTY = 100

    def __init__(self, warehouse, warehouse_cost, todo):

        self.todo = todo
        self.boxes_delivered = []
        self.total_cost = 0
        self._set_initial_state_from(warehouse)
        self.warehouse_cost = warehouse_cost

        self.delta = [[-1, 0],  # go up
                        [0, -1],  # go left
                        [1, 0],  # go down
                        [0, 1],  # go right
                        [-1, -1],  # up left (diag)
                        [-1, 1],  # up right (diag)
                        [1, 1],  # dn right (diag)
                        [1, -1]]  # dn left (diag)

        self.delta_directions = ["n", "w", "s", "e", "nw", "ne", "se", "sw"]

        # Use this for a visual debug
        self.delta_name = ['^', '<', 'v', '>', '\\', '/', '[', ']']

        # Costs for each move
        self.delta_cost = [self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST]

    # state parsing and initialization function from testing_suite_partA.py
    def _set_initial_state_from(self, warehouse):
        """Set initial state.

        Args:
            warehouse(list(list)): the warehouse map.
        """
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'

                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'

                elif this_square == '@':
                    self.warehouse_state[i][j] = '*'
                    self.dropzone = (i, j)

                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)


    def _search_b(self, goal, start, target, debug=True):
        """
        This method should be based on Udacity Quizes for A*, see Lesson 12, Section 10-12.
        The bulk of the search logic should reside here, should you choose to use this starter code.
        Please condition any printout on the debug flag provided in the argument.  
        You may change this function signature (i.e. add arguments) as 
        necessary, except for the debug argument which must remain with a default of False
        """

        # get a shortcut variable for the warehouse (note this is just a view no copying)
        grid = self.warehouse_state
        warehouse_cost = self.warehouse_cost
        ## Step 1: Create Tracker grid with shortest # of steps to goal for each cell

        ## Initialize Tracker from grid with simplified obstacles
        tracker = copy.deepcopy(grid)
        
        for r in range(len(tracker)):
            for c in range(len(tracker[0])):
                if tracker[r][c] not in ('.','@','*'):
                    tracker[r][c] = '#'
                else:
                    tracker[r][c] = '.'
                if r == goal[0] and c == goal[1]:
                    #tracker[r][c] = warehouse_cost[r][c]
                    tracker[r][c] = 0

        ## Expand open adjacent cells starting at the goal and update with cost to get to the goal
        pos = goal
        adj = _adjacents(tracker, pos)
        #print(adj)
        cost = int(warehouse_cost[pos[0]][pos[1]])

        # iterate through adjancents
        while len(adj):
            adj2 = []
            #print('____ gap ______')
            
            for a in adj:
                ## get the cost of the least expensive position adjacent to this adjacent
                min_cost_adj = _adjacents_mincost(tracker, a[0], excl_ints=False)
                cost = int(warehouse_cost[a[0][0]][a[0][1]]) + min_cost_adj ## cost of this spot + cheapest adjacent spot
                ## if not already explored, or has higher cost
                if (type(tracker[a[0][0]][a[0][1]]) != int) or (tracker[a[0][0]][a[0][1]] > cost):
                    #print(a, cost2)
                    tracker[a[0][0]][a[0][1]] = cost
                    [adj2.append(aa) for aa in _adjacents(tracker, a[0]) if aa not in adj2] ## add this adjacent's adjacents
            adj = adj2[:] ## new adjacents become the main adjacent list

        ## returns grid of minimum costs to the goal for any given spot in the grid
        return tracker

    def _generate_plan(self, tracker, goal, target_label, action, box_pos):
        '''
        Based on a grid of minimum costs to the goal for any given spot in the grid,
        generates movement commands for the robot for any given location
        '''

        ## initialize empty grid the same size as the warehouse
        directions = [[None for i in range(len(tracker[0]))] for r in range(len(tracker))]

        for r in range(len(tracker)):
            for c in range(len(tracker[0])):
                pos = (r,c) # position in the grid
                adj = _adjacents(tracker, pos, excl_obstacles=True, excl_ints=False) ## get valid adjacent positions to this one

                costs = [a[1] for a in adj] ## costs of the adjacent positions
                target = [a for a in adj if a[1] == min(costs)][0][0] ## target cell from current position should be the one with the minimum cost

                delt   = ( np.matrix(target) - np.matrix(pos) ).tolist()[0] ## position difference between current position and the target
                direct = 'move ' + self.delta_directions[ self.delta.index(delt) ] ## translate into directions for the robot

                ## if adjacent to goal, generate lift or put down directions
                for a in adj:
                    if a[0] == goal:
                        if action == 'retrieve':
                            direct = 'lift ' + target_label
                            break
                        if action == 'deliver':
                            direct = 'down ' + direct.replace('move','')

                ## add direction to the grid
                directions[pos[0]][pos[1]] = direct

                ## edit obstacle or box position cells to match value required by the instructions
                if tracker[r][c] == '#' and (r,c) != box_pos:
                    directions[r][c] = -1
                if (r,c) == goal and action == 'retrieve':
                    directions[r][c] = 'B'
        
        ## return robot instructions for an given position in the warehouse
        return directions                 


    def _find_policy(self, goal, pickup_box=True, debug=False):
        """
        This method should be based on Udacity Quizes for Dynamic Progamming,
        see Lesson 12, Section 14-20 and Problem Set 4, Question 5.  The bulk of
        the logic for finding the policy should reside here should you choose to
        use this starter code.  Please condition any printout on the debug flag
        provided in the argument. You may change this function signature
        (i.e. add arguments) as necessary, except for the debug argument which
        must remain with a default of False
        """

        ##############################################################################
        # insert code in this method if using the starter code we've provided
        ##############################################################################


        # get a shortcut variable for the warehouse (note this is just a view it does not make a copy)
        grid = self.warehouse_state
        grid_costs = self.warehouse_cost

        ## get dropzone and goals
        dropzone, goals = _get_dropzone_goals(grid)
        pos = dropzone
        goal = goals[0][0]
        target_label = goals[0][1]

        # You will need to fill in the algorithm here to find the policy
        # The following are what your algorithm should return for test case 1
        if pickup_box:
            # To box policy
            #                        in retrieval mode, the goal is the box and starting point for the cost search algorithm is the dropzone
            tracker = self._search_b(goal=goal, start=pos, target='1', debug=debug)
            policy  = self._generate_plan(tracker=tracker, goal=goal, target_label=target_label, action='retrieve', box_pos=goal)

        else:
            # Deliver policy
            #                        in delivery mode, the goal is really the dropzone and starting point for the cost search algorithm is from the box
            tracker = self._search_b(goal=pos, start=goal, target='1', debug=debug)
            policy  = self._generate_plan(tracker=tracker, goal=pos, target_label='1', action='deliver', box_pos=goal)

        return policy


    def plan_delivery(self, debug=False):
        """
        plan_delivery() is required and will be called by the autograder directly.  
        You may not change the function signature for it.
        Add logic here to find the policies:  First to the box from any grid position
        then to the dropzone, again from any grid position.  You may use the starter
        code provided above in any way you choose, but please condition any printouts
        on the debug flag
        """
        ###########################################################################
        # Following is an example of how one could structure the solution using
        # the starter code we've provided.
        ###########################################################################


        # Start by finding a policy to direct the robot to the box from any grid position
        # The last command(s) in this policy will be 'lift 1' (i.e. lift box 1)
        goal = self.boxes['1']
        to_box_policy = self._find_policy(goal, pickup_box=True, debug=debug)

        # Now that the robot has the box, transition to the deliver policy.  The
        # last command(s) in this policy will be 'down x' where x = the appropriate
        # direction to set the box into the dropzone
        goal = self.dropzone
        deliver_policy = self._find_policy(goal, pickup_box=False, debug=debug)

        if debug:
            print("\nTo Box Policy:")
            for i in range(len(to_box_policy)):
                print(to_box_policy[i])

            print("\nDeliver Policy:")
            for i in range(len(deliver_policy)):
                print(deliver_policy[i])

            print("\n\n")

        return (to_box_policy, deliver_policy)



def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith123).
    whoami = 'sshepherd35'
    return whoami


if __name__ == "__main__":
    """ 
    You may execute this file to develop and test the search algorithm prior to running 
    the delivery planner in the testing suite.  Copy any test cases from the
    testing suite or make up your own.
    Run command:  python warehouse.py
    """

    # Test code in here will not be called by the autograder

    # Testing for Part A
    # testcase 1
    warehouse = ['1#2',
                '.#.',
                '..@']

    todo =  ['1','2'] 

    partA = DeliveryPlanner_PartA(warehouse, todo)
    partA.plan_delivery(debug=True)

    # Testing for Part B
    # testcase 1
    warehouse = ['1..',
                 '.#.',
                 '..@']

    warehouse_cost = [[0, 5, 2],
                      [10, math.inf, 2],
                      [2, 10, 2]]

    todo = ['1']

    partB = DeliveryPlanner_PartB(warehouse, warehouse_cost, todo)
    partB.plan_delivery(debug=True)


### SBS functions
def _printgrid(grid):
    print('----------------------')
    for g in grid:
        print(g)
    print('----------------------')

def _get_dropzone_goals(grid):
    '''Searches grid to extract the position of the dropzone and any boxes'''
    dropzone = None
    goals = []

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '*':
                dropzone = (r,c)
            if len( re.sub(r'[^0-9a-zA-Z]', '', grid[r][c]) ):
                goals.append( ((r,c), grid[r][c]) )

    #goals.sort(key = lambda x: x[1])

    return dropzone, goals

def _adjacents(grid, cell, excl_obstacles=True, excl_ints=True):
    '''Returns adjacent cells to a given position'''
    adj = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            rdiff = np.abs(cell[0] - r)
            cdiff = np.abs(cell[1] - c)
            if rdiff in (0,1) and cdiff in (0,1) and rdiff + cdiff in (1,2):
                adj.append( ((r,c), grid[r][c]) )
                
                if grid[r][c] not in ('.','@','*') and excl_obstacles:
                    ignore = adj.pop()
                    if not excl_ints:
                        if type(grid[r][c]) == int:
                            adj.append( ((r,c), grid[r][c]) )
                
    return adj

def _adjacents_mincost(grid, cell, excl_obstacles=True, excl_ints=True):
    '''Searches adjacent cells for the minimum cost value'''
    adj = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            rdiff = np.abs(cell[0] - r)
            cdiff = np.abs(cell[1] - c)
            if rdiff in (0,1) and cdiff in (0,1) and rdiff + cdiff in (1,2):
                adj.append( ((r,c), grid[r][c]) )
                #print('adj: ', (r,c), grid[r][c])
                
                if grid[r][c] not in ('@','*') and excl_obstacles:
                    ignore = adj.pop()
                    if not excl_ints:
                        if type(grid[r][c]) == int:
                            adj.append( ((r,c), grid[r][c]) )
                            
    ## min cost of all adjacents
    costs = [int(grid[a[0][0]][a[0][1]]) for a in adj]

    return min(costs)