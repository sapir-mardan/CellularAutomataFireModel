from  simulation import *

if __name__ == "__main__":
    grid_sizes = [(i, i) for i in range(5, 30)] + [(i, i) for i in range(30, 100, 5)]
    probabilities = [0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.9]
    rs = [0., 0.25, 0.5, 0.75, 0.9, 1.]
    simulations_results_df = run_simulations(grid_sizes, fs=probabilities, ps=probabilities, rs=rs, num_states=300, num_simulations_per_params=5, num_last_iterations=50, pct_steady_range=20, above_fire_rate_margin=5)
    simulations_results_df.to_csv("results/final_simulation.csv", ascending = False)
    

