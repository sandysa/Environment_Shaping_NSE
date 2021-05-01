#####################################################################
# Author: Sandhya Saisubramanian
# Description: Implements helper functions for NSE calculation.
#####################################################################
import numpy as np

bp_nse_penalty = 5

def readmap(filename):
    index = 0
    width = 0
    height = 0
    with open(filename,'r') as f:
        for line in f:
            index += 1
            if index == 1:
                height = int(line.strip())
                
            elif index == 2:
                width = int(line.strip())
                break
    f.close()
    grid = np.zeros((height,width), dtype='U1')
    curr_row  = 0
    index = 0
    with open(filename,'r') as f:
        for line in f:
            index += 1
            if index <= 2:
                continue
            curr_col = 0
            for char in line.strip():
                grid[curr_row][curr_col] = char
                curr_col += 1
            curr_row += 1
    f.close()
    
    return grid


# Boxpushing support functions
def CarpetLoc_boxpushing(grid):
    carpet = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '@':
                carpet.append((r,c))
    return carpet

def VaseLocation(grid):
    vase = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'V':
                vase.append((r,c))
    return vase


def ParseGrid_boxpushing(grid):
    wall = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'S':
                s0 = (r,c,False)
            if grid[r][c] == "G":
                goal = (r,c)
                goal_state = (r,c,True)
            if grid[r][c] == "B":
                box = (r,c)
            if grid[r][c] == "x":
                wall.append((r,c))
    return s0, goal, goal_state, box, wall

def get_nse_penalty_bp(state,action,NSE_locations):
    if is_NSE_boxpushing(state,action,NSE_locations):
        return 5
    return 0

def is_NSE_boxpushing(state, action,NSE_locations):
    if action == 'pickup':
        return False

    if ((state[0], state[1])) in NSE_locations and state[2] == True:
        return True

    return False

def NSE_penalty_boxpushing(all_states, policy,NSE_locations,visitation_freq):
    nse_penalty = 0
    penalty = 5
    for s, state in enumerate(all_states):
        if s in policy:
            action = policy[s]
            if is_NSE_boxpushing(state,action,NSE_locations):
                nse_penalty += penalty * visitation_freq[s]

    return nse_penalty

def bp_state_to_feature(state):
        temp = []
        temp.append(state[0])
        temp.append(state[1])
        if state[2] == True:
            temp.append(1)
        else:
            temp.append(0)
        return temp


# Support function for driving domain
def ParseGrid_driving(grid):
    wall = []
    regions = []
    pothole = {}
    pothole['S'] = [] #Shallow potholes
    pothole['D'] = [] #Deep potholes
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == "x":
                wall.append((r,c))
            if grid[r][c] == "L":
                regions.append((r,c))
            if grid[r][c] == "@":
                pothole['S'].append((r,c))
            if grid[r][c] == "D":
                pothole['D'].append((r,c))
    return wall,pothole,regions, len(grid[0])-1, len(grid)-1


def WallLoc_driving(grid):
    wall = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == "x":
                wall.append((r,c))
    return wall

def Reduced_speed_regions(grid):
    regions = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == "L":
                regions.append((r,c))
    return regions

def PotholeLoc_driving(grid):
    pothole = {}
    pothole['S'] = [] #Shallow potholes
    pothole['D'] = [] #Deep potholes
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == "@":
                pothole['S'].append((r,c))
            if grid[r][c] == "D":
                pothole['D'].append((r,c))
    return pothole

def mild_NSE_driving(state,action,potholes):
    if "fast" in action and state in potholes['S']:
        return True 
    return False
   
def severe_NSE_driving(state,action,potholes):
    if "fast" in action and state in potholes['D']:
        return True
    return False

def get_nse_penalty_dr(state,action,potholes):
    if mild_NSE_driving(state,action,potholes):
        return 2
    elif severe_NSE_driving(state,action,potholes):
        return 5
    else:
        return 0

def NSE_penalty_driving(all_states,policy,NSE_locations,visitation_freq):
    nse_penalty = 0
    mild_NSE_penalty = 2
    severe_NSE_penalty = 5
    for s, state in enumerate(all_states):
        if s in policy:
            action = policy[s]
            if mild_NSE_driving(state,action,NSE_locations):
                nse_penalty += mild_NSE_penalty *visitation_freq[s]

            elif severe_NSE_driving(state,action,NSE_locations):
                nse_penalty += severe_NSE_penalty * visitation_freq[s]

    return nse_penalty

def get_modifications(domain_name):
    if domain_name == "driving":
        return ['null','reduce_speed_all','fill_potholes','fill_deep_potholes','reduce_speed_fill_deep',\
                'reduce_speed_zone1','reduce_speed_zone2', 'reduce_speed_zone3','reduce_speed_zone4']
    if domain_name == "boxpushing":
        return ['null', 'remove_rug', 'remove_rug_remove_vase', 'add_sheet', 'add_sheet_moveVase_bottomleft', 'add_sheet_moveVase_bottomright',\
                'add_sheet_moveVase_topleft','add_sheet_moveVase_topright', 'moveVase_bottomleft', 'moveVase_topleft', 'moveVase_topright',\
                 'moveVase_bottomright','add_sheet_remove_vase','block', 'block_rug', 'remove_vase', 'block_rug_moveVase_bottomleft',\
                 'block_rug_moveVase_bottomright','block_rug_moveVase_topright', 'block_rug_moveVase_topleft', 'remove_rug_moveVase_bottomleft',\
                'remove_rug_moveVase_bottomright', 'remove_rug_moveVase_topleft', 'remove_rug_moveVase_topright']
