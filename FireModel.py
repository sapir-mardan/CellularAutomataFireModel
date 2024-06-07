import numpy as np
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd
from tqdm import tqdm


###########################
#   Simulations Functions
###########################

EMPTY = 0
TREE = 1
FIRE = 2
FIRE_SPREADING = 3


def create_grid(grid_size):
    """
    This function creates a grid of integers, randomly set to be either  TREE or EMPTY.
    It will be used to start a system for simulating a simplified model of the propagation of a forest fire.
    """
    
    return rng.integers(low=0, high=2, size=grid_size)

def random_choice(p, vals, size):
    return np.random.choice(vals, p=[p, 1-p], size=size)

def compute_next_grid(grid, ps, fs, rs, t):
    """
    This function takes a grid (2D array) and computes its next state by a set of rules.
    The state of each cell can be EMPTY, TREE, FIRE(caused by probability), or FIRE_SPREADING.

    - Arguments:
        grid: the array.
        f (float): probability a tree turns into fire.
        p (float): probability an empty space fills with a tree(tree growth).
        r (float): probability that fire becomes a tree (fire put out by rain  - new custom rule). Default is 0 (no rain possible).

    - Returns:
        next_grid: array with the next cell states.
    """
    
    next_grid = grid.copy()

    is_any_type_fire_grid = (grid == FIRE) | (grid == FIRE_SPREADING)
    is_tree_grid = grid == TREE
    is_empty_grid = grid == EMPTY

    # rule 1 (burn-out) + new rule (rain)
    np.putmask(next_grid, is_any_type_fire_grid, rs[t])

    has_neighbor_on_fire_grid = compute_has_neighbors_on_fire(is_any_type_fire_grid)
    is_tree_with_neighbors_on_fire_grid = is_tree_grid & has_neighbor_on_fire_grid
    # rule 2 (fire spread)
    next_grid[is_tree_with_neighbors_on_fire_grid] = FIRE_SPREADING

    is_tree_without_neighbors_on_fire_grid = is_tree_grid & (~has_neighbor_on_fire_grid)
    # rule 3 (lightning strike)
    np.putmask(next_grid, is_tree_without_neighbors_on_fire_grid, fs[t])

    # rule 4 (tree growth)
    np.putmask(next_grid, is_empty_grid, ps[t])
    return next_grid

def compute_has_neighbors_on_fire(is_any_type_fire_grid):
    """
    This function checks if a cell in a grid has a neighbor that is in state FIRE or FIRE_SPREADING.

    - Arguments:
        grid: the array.
        row: the row of the cell.
        col: the column of the cell.

    - Returns:
        True if a neighbor is in state FIRE/FIRE_SPREADING else False.

    """
    down = np.pad(is_any_type_fire_grid, ((1,0), (0,0)))[:-1]
    up = np.pad(is_any_type_fire_grid, ((0,1), (0,0)))[1:]
    right = np.pad(is_any_type_fire_grid, ((0,0), (1,0)))[:, :-1]
    left = np.pad(is_any_type_fire_grid, ((0,0), (0,1)))[:, 1:]
    return down + up + right + left

def count_cell_state_in_grid_states(states, cell_state):
    return np.sum(states == cell_state, axis=(1, 2))

def compute_state_percentages(states):
  """
  This functions computes for an array of states the percentages of trees, total fires, and spreading fires (fires caused by spreading) in each state.

  -Parameters:
    - states: 3D array of grid states (array of grids from a simulation).

   -Returns:
      - percentages_arr: 2D array of shape 3x(number of grid states), where each of the 3 rows has the percentages of a different cell values:
        -   row 0: trees_percentages: array of numbers of total trees in each grid state.
        -   row 1: total_fires_percentages: array of numbers of total fires in each grid state.
        -   row 2: spreading_fires_percentages: array of numbers of spreading fires in each grid state.
  """
  # get counts
  trees_counts = count_cell_state_in_grid_states(states, TREE)
  fires_counts = count_cell_state_in_grid_states(states, FIRE)
  spreading_fires_counts = count_cell_state_in_grid_states(states, FIRE_SPREADING)
  total_fires_counts = fires_counts + spreading_fires_counts
  counts_arr = np.vstack([trees_counts, total_fires_counts, spreading_fires_counts])

  # convert to percentages
  total_num_cells_in_grid = states.shape[1] * states.shape[2]

  percentages_arr = 100 * counts_arr / total_num_cells_in_grid
  percentages_arr = percentages_arr.round(2)

  return percentages_arr

def run_simulation(grid_size, f, p, r=0.0, num_grid_states=200, visualise = False):
    """
    This function runs a simulation of propagation of a forest fire by set of rules.
    It creates a random grid with cells having values of either TREE or EMPTY,
    and computes all the next states of the grid.

    A state can be EMPTY, TREE, FIRE(caused by lightning-strike), FIRE_SPREADING.

    - Arguments:
        grid_size: tuple of size of the array.
        f (float): probability a tree turns into fire by a lightning-strike.
        p (float): probability an empty space fills with a tree (tree growth rate).
        r (float): probability that fire becomes a tree (fire put out by rain  - new custom rule). Default is 0 (no rain possible).
        num_grid_states (int): number of total grid states
        visualise: If True, the array will be saved for visualization in another script. Default is False.

    - Returns:
        states: history of all steps of a simulation as a 3D array.

    """
    states = np.empty((num_grid_states, grid_size[0], grid_size[1]))
    ps, fs, rs = random_choice(p, [TREE, EMPTY], states.shape), random_choice(f, [FIRE, TREE], states.shape), random_choice(r, [TREE, EMPTY], states.shape)

    grid = create_grid(grid_size)
    states[0] = grid

    for t in range(1, num_grid_states):
        grid = compute_next_grid(grid, ps, fs, rs, t)
        states[t] = grid
        
    if visualise:
      percentage_arr = compute_state_percentages(states)
      steady = is_steady(percentage_arr, 50, 20, f, 5)
      np.savez_compressed("results/states_simulation", state=states, perc=percentage_arr, grid=grid_size, f=f, p=p, r=r, steady=steady)
      
    return states

def run_simulations(grid_sizes, fs, ps, rs, num_states, num_simulations_per_params, num_last_iterations, pct_steady_range, above_fire_rate_margin):
    """
    This function run simulations for a combinations of parameters.

    -Parameters:
        - grid_sizes: all the grid sizes to test in the simulations
        - fs: all the values of probability f (fire) to test in the simulations
        - ps: all the values of probability p (tree growth) to test in the simulations
        - rs: all the values of probability r (rain) to test in the simulations
        - num_states: number of grid states per each simulation
        - num_simulations_per_params: number of simulations to run for each combination of (grid_size, f, p, r)
        - num_last_iterations: the number of last states that are used for determining steady state
        - pct_steady_range: the size of the range for the percentage to be determined as in a steady state
        - above_fire_rate_margin: The margins by which the rate of total fires exceeds the lightning strike rate.

    -Returns:
        results_df: a dataframe with each row containing the results for a set of parameters:
                    "grid_size": grid size,
                    "f": lightning probability,
                    "p": tree growth probability
                    "r": rain probability,
                    "num_states": number of states in each simulation,
                    "num_simulations": number of simulation runs for the parameter combination,
                    "is_steady": did simulation reach steady state,
                    "pct_trees": average precentage of trees in steady state, or np.nan if not steady
                    "pct_fires": average precentage of fires(all) in all steady state, or np.nan if not steady

    """
    
    all_simulations_results = []
    for r in tqdm(rs):
      for grid_size in grid_sizes:
        for f in fs:
          for p in ps:
            for _ in range(num_simulations_per_params):
              states = run_simulation(grid_size, f, p, r, num_grid_states=num_states)
              percentages_arr = compute_state_percentages(states)
              reached_steady_state = is_steady(percentages_arr, num_last_iterations, pct_steady_range, f, above_fire_rate_margin)
              if reached_steady_state:
                pct_means = percentages_arr[:2, -num_last_iterations:].mean(1)
                pct_trees, pct_fires = pct_means
              else:
                pct_trees, pct_fires = np.nan, np.nan
              simulation_results = {
                  "grid_size": grid_size,
                    "f": f,
                    "p": p,
                    "r": r,
                    "num_states": num_states,
                    "num_simulations": num_simulations_per_params,
                    "is_steady": reached_steady_state,
                    "pct_trees": pct_trees,
                    "pct_fires": pct_fires,
              }
              all_simulations_results.append(simulation_results)
    simulations_results_df = pd.DataFrame(data=all_simulations_results)
    return simulations_results_df

######################
# Functions for Stats
######################

from tqdm import tqdm

def is_steady(percentages_arr, num_last_iterations, pct_steady_range, f, above_fire_rate_margin):
    """
    This function checks if a simulation entered a steady state.
    A steady steady state defined if for all num_last_iterations:
    1. Are in range pct_steady_range.
    2. Percentage of trees and fires (percentages_arr) between 0-100.
    3. Percentage of total fires is bigger than f by above_fire_rate_margin.
    """
    last_percentages_arr = percentages_arr[:, -num_last_iterations:]
    percentage_minimums = last_percentages_arr.min(axis=1)
    # check that
    #   1) no percentage is zero: always had spreading fires and trees
    #   2) percentage of trees or total fires is never 100%
    has_state_with_0_or_100_percent = np.any((last_percentages_arr == 0) | (last_percentages_arr == 100))
    if has_state_with_0_or_100_percent:
        return False

    percent_fires_always_above_lightning_rate = np.all(last_percentages_arr[1] > (100*f) + above_fire_rate_margin)
    if not percent_fires_always_above_lightning_rate:
        return False

    percentage_maximums = last_percentages_arr.max(axis=1)
    percentage_ranges = percentage_maximums - percentage_minimums
    maximum_range = percentage_ranges[:2].max()
    return maximum_range <= pct_steady_range

def analyze_parameters(num_states, num_last_iterations, pct_steady_range, above_fire_rate_margin, num_extra_verification_iterations):
    """
    This functions used to assess the validity of simulation parameters.

    -Parameters:
      - num_states: Number of grids (iterations) in a simulation.
      - num_last_iterations: Number of last iterations to check for steady-state.
      - pct_steady_range: The maximum range within which the percentage of trees and fires can vary.
      - above_fire_rate_margins: The margins by which the rate of total fires exceeds the lightning strike rate.
      - num_extra_verification_iteration: Number of subsequent iterations above num_states to ensure a steady-state with given parameters, remains as such.

    Values implemented as variables:
    - grid_sizes: [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
    - probabilities: [0.1, 0.3, 0.5, 0.7, 0.9]
    
    -Returns:
      - A list with parameters and their agree_rate and steady_rate.
      
      agree_rate is the agreement rate between the number of last iterations and num_extra_verification_iteration.
      steady_rate is the rate at which both short and long simulations reach steady states.
      """
    is_steady_list = []
    is_agreement_list = []

    grid_sizes = [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

    for grid_size in grid_sizes:
        for f in probabilities:
            for p in probabilities:
                states = run_simulation(grid_size, f, p, r=0, num_grid_states=num_states + num_extra_verification_iterations)

                #check steady-state for simulation with extra interations (long)
                longer_percentages_arr = compute_state_percentages(states)
                #check steady-state for last iterations of a simulation (short)
                shorter_percentages_array = longer_percentages_arr[:, :-num_extra_verification_iterations]


                shorter_reached_steady_state = is_steady(shorter_percentages_array, num_last_iterations, pct_steady_range, f, above_fire_rate_margin)
                longer_reached_stady_state = is_steady(longer_percentages_arr, (num_last_iterations + num_extra_verification_iterations) , pct_steady_range, f, above_fire_rate_margin)

                #check agreement between short and long
                agree_on_steady_or_not = (shorter_reached_steady_state == longer_reached_stady_state)

                #check both agreement and steady-state for short
                both_cases_steady_state_reached = agree_on_steady_or_not and shorter_reached_steady_state

                is_agreement_list.append(agree_on_steady_or_not)
                is_steady_list.append(both_cases_steady_state_reached)

    agree_rate = np.array(is_agreement_list).mean()
    steady_rate = np.array(is_steady_list).mean()
    return [num_states, num_last_iterations, pct_steady_range, above_fire_rate_margin, agree_rate, steady_rate]




# ## FULL SIMULATIONS EXAMPLE
# grid_sizes = [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
# probabilities = [0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.9]
# rs = [0., 0.25, 0.5, 0.75, 0.9, 1.]
# simulations_results_df = run_simulations(grid_sizes, fs=probabilities, ps=probabilities, rs=rs, num_states=300, num_simulations_per_params=5, num_last_iterations=50, pct_steady_range=20, above_fire_rate_margin=5)
# simulations_results_df.to_csv("full_simulations_results.csv", index=False)


### TO RUN A SIMULATION FOR VISUALIZATION EXAMPLE
#run_simulation((3,3), 0.3, 0.3, r=0.0, num_grid_states=50, visualise = True)
