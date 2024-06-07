# Simulation Of Model Of The Propagation Of A Forest Fire

### Table of Contents 
- [Project Description](#project-description)
- [Required Technologies](#required-technologies)
- [Usage](#usage)
- [License](#license)

### Project Description

A Python based project that employs cellular automata to simulate forest fire propagation, aiming to explore dynamic system behaviour under varied parameters. 

### Required Technologies

* Python 3.11.4 or higher.
* NumPy library.
* Matplotlib library.
* Pandas library.

Disclaimer: The system requires a computer with the ability to run computationally intensive tasks for extended durations, as the simulation may require extended processing times.

### Usage

*Run Simulations*

* To run one simulation, change your current directory to the location of simulation.py. Run the following code in terminal to execute the function “run_simulation” with preferable parameters. The parameters to conclude a steady-state are the ones which were determined most valid.

  ```bash
	python -c "from simulation import run_simulation; run_simulation(grid size, f, p, r, num_grid_states, visualise)" --simulation.py
  ```

The description of arguments required can be found with help(simulation.run_simulation).
(In terminal; launch python -> import simulation -> type help(simulation.run_simulation)).


* Run multiple simulations run_simulations.py. Because of a long execution time, the results were added in folder results as full_simulations_results.csv.

* Run visualisation for a simulation with visualise_simulation.ipynb.


Steady-State Parameters Investigation

* Run investigation to assess the validity of simulation parameters with run_investigate_stats.py. The results were also added in folder results as investigate_parameters_results.csv.

* analysis.ipynb shows the analysis made of results from run_simulations.py and run_investigate_stats.py for scientific report.

### License



