# Simulation Of Model Of The Propagation Of A Forest Fire



https://github.com/sapir-mardan/CellularAutomataFireModel/assets/92859243/b5b380e4-2dcf-4a92-9174-282a1a9eb340


### Table of Contents 
- [Project Description](#project-description)
- [How It Works](#how-it-works)
- [Required Technologies](#required-technologies)
- [Usage](#usage)
- [License](#license)

### Project Description

A Python based project that employs cellular automata to simulate forest fire propagation, aiming to explore dynamic system behaviour under varied parameters. 

## How It Works

## Base Model

This simulation represents a forest fire propagation model on a grid, aiming to explore how model parameters influence the system's dynamics towards achieving a steady state. The grid evolves over time based on the following rules applied at each time step:

1. **Burn-out**: Any cell that is on fire will become empty in the next state.
2. **Fire Spread**: Any tree will catch fire if it has at least one neighboring cell on fire.
3. **Lightning Strike**: A tree will spontaneously catch fire with a probability 'f', independent of its neighbors.
4. **Tree Growth**: An empty cell will become a tree with a probability 'p'.
And for the extended model An additional rule introduces rain:
5. **Burn-out or Rain**: A cell on fire can either become empty or revert to a tree with a probability 'r', simulating rain extinguishing the fire.

The simulation uses four parameters:
- Tree growth rate ('p')
- Lightning strike rate ('f')
- Grid size
- Number of states (time steps)

## Steady-State Investigation

The goal is to determine conditions under which the system reaches a steady state, defined by:
- The number of trees and fires remains relatively constant over the last 50 of 300 iterations.
- Fires caused by spreading (not just by lightning strikes) are consistent, ensuring dynamic stability.


### Results and Analysis

The simulation outputs help analyze how varying the parameters 'p', 'f', and the grid size, as well as introducing rain, affect the likelihood and stability of reaching a steady state. This model provides valuable insights into forest fire dynamics and the effectiveness of various fire suppression techniques modeled by the simulation rules.



### Required Technologies

* Python 3.11.4 or higher.
* NumPy library.
* Matplotlib library.
* Pandas library.

Disclaimer: The system requires a computer with the ability to run computationally intensive tasks for extended durations, as the simulation may require extended processing times.

### Usage

*FireModel* includes all functions for running simulations and statistical analysis of steady state.

*Run Simulations*

* To run one simulation, change your current directory to the location of simulation.py. Run the following code in terminal to execute the function “run_simulation” with preferable parameters. The parameters to conclude a steady-state are the ones which were determined most valid.

  ```bash
	python -c "from simulation import run_simulation; run_simulation(grid size, f, p, r, num_grid_states, visualise)" --simulation.py
  ```

The description of arguments required can be found with help(simulation.run_simulation).
(In terminal; launch python -> import simulation -> type help(simulation.run_simulation)).


* Run multiple simulations run_simulations.py.  An example of the results were added in folder results as full_simulations_results.csv.

* Run visualisation for a simulation with visualise_simulation.ipynb (such as in the top of this ReadMe and additional graph).

Steady-State Parameters Investigation

* Run investigation to assess the validity of simulation parameters with run_investigate_stats.py. The results were also added in folder results as investigate_parameters_results.csv.

* analysis.ipynb shows the analysis made of results from run_simulations.py and run_investigate_stats.py for scientific report.

### License

This project is licensed under the MIT License.

Acknowledgments:
- This project was assigned by the University of Bristol as part of the academic curriculum for the Master’s program in Bioinformatics.


