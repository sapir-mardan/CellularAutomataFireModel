import numpy as np
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd
from tqdm import tqdm

EMPTY = 0
TREE = 1
FIRE = 2
FIRE_SPREADING = 3


def create_grid(grid_size):
    return rng.integers(low=0, high=2, size=grid_size)

def random_choice(p, vals, size):
    return np.random.choice(vals, p=[p, 1-p], size=size)

def compute_next_grid(grid, ps, fs, rs, t):
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
  down = np.pad(is_any_type_fire_grid, ((1,0), (0,0)))[:-1]
  up = np.pad(is_any_type_fire_grid, ((0,1), (0,0)))[1:]
  right = np.pad(is_any_type_fire_grid, ((0,0), (1,0)))[:, :-1]
  left = np.pad(is_any_type_fire_grid, ((0,0), (0,1)))[:, 1:]
  return down + up + right + left

def run_simulation(grid_size, f, p, r=0.0, num_grid_states=200):
    states = np.empty((num_grid_states, grid_size[0], grid_size[1]))
    ps, fs, rs = random_choice(p, [TREE, EMPTY], states.shape), random_choice(f, [FIRE, TREE], states.shape), random_choice(r, [TREE, EMPTY], states.shape)

    grid = create_grid(grid_size)
    states[0] = grid

    for t in range(1, num_grid_states):
        grid = compute_next_grid(grid, ps, fs, rs, t)
        states[t] = grid
    return states


def count_cell_state_in_grid_states(states, cell_state):
    return np.sum(states == cell_state, axis=(1, 2))

def compute_state_percentages(states):
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

def analyze_parameters(num_states, num_last_iterations, pct_steady_range, above_fire_rate_margin, num_extra_verification_iterations):
  is_steady_list = []
  is_agreement_list = []

  grid_sizes = [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
  probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

  for grid_size in grid_sizes:
    for f in probabilities:
      for p in probabilities:
          states = run_simulation(grid_size, f, p, r=0, num_grid_states=num_states + num_extra_verification_iterations)

          longer_percentages_arr = compute_state_percentages(states)
          shorter_percentages_array = longer_percentages_arr[:, :-num_extra_verification_iterations]


          shorter_reached_steady_state = is_steady(shorter_percentages_array, num_last_iterations, pct_steady_range, f, above_fire_rate_margin)
          longer_reached_stady_state = is_steady(longer_percentages_arr, (num_last_iterations + num_extra_verification_iterations) , pct_steady_range, f, above_fire_rate_margin)

          agree_on_steady_or_not = (shorter_reached_steady_state == longer_reached_stady_state)

          both_cases_steady_state_reached = agree_on_steady_or_not and shorter_reached_steady_state

          is_agreement_list.append(agree_on_steady_or_not)
          is_steady_list.append(both_cases_steady_state_reached)

  agree_rate = np.array(is_agreement_list).mean()
  steady_rate = np.array(is_steady_list).mean()
  return [num_states, num_last_iterations, pct_steady_range, above_fire_rate_margin, agree_rate, steady_rate]


from tqdm import tqdm

def is_steady(percentages_arr, num_last_iterations, pct_steady_range, f, above_fire_rate_margin):
    last_percentages_arr = percentages_arr[:, -num_last_iterations:]
    percentage_minimums = last_percentages_arr.min(axis=1)
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


def run_simulations(grid_sizes, fs, ps, rs, num_states, num_simulations_per_params, num_last_iterations, pct_steady_range, above_fire_rate_margin):
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


# ##FULL SIMULATION EXAMPLE
# grid_sizes = [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
# probabilities = [0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.9]
# rs = [0., 0.25, 0.5, 0.75, 0.9, 1.]

# simulations_results_df = run_simulations(grid_sizes, fs=probabilities, ps=probabilities, rs=rs, num_states=300, num_simulations_per_params=5, num_last_iterations=50, pct_steady_range=20, above_fire_rate_margin=5)
# simulations_results_df.to_csv("full_simulations_results.csv", index=False)

# analysis_parameters_evaluation_results = []

# for num_states in ([200, 300, 400]):
#     for num_last_iterations in [50, 100]:
#         for pct_steady_range in [5, 10, 12, 15, 20]:
#             for above_fire_margin in [3, 5]:
#                 analysis_results = analyze_parameters(num_states, num_last_iterations, pct_steady_range, above_fire_margin, num_last_iterations)
#                 analysis_parameters_evaluation_results.append(analysis_results)

# simulations_results_df = pd.DataFrame(data=analysis_parameters_evaluation_results, columns=['num_states', 'num_last_iterations', 'pct_steady_range', 'margin', 'agree_rate', 'steady_rate'])
# simulations_results_df.to_csv("test_analize2.csv", index=False)
