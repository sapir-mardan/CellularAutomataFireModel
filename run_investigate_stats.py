from  FireModel import *

if __name__ == "__main__":
    analysis_parameters_evaluation_results = []

    for num_states in ([200, 300, 400]):
        for num_last_iterations in [50, 100]:
            for pct_steady_range in [5, 10, 12, 15, 20]:
                for above_fire_margin in [3, 5]:
                    analysis_results = analyze_parameters(num_states, num_last_iterations, pct_steady_range, above_fire_margin, num_last_iterations)
                    analysis_parameters_evaluation_results.append(analysis_results)

    simulations_results_df = pd.DataFrame(data=analysis_parameters_evaluation_results, columns=['num_states', 'num_last_iterations', 'pct_steady_range', 'margin', 'agree_rate', 'steady_rate'])
    simulations_results_df.to_csv("results/full_simulations_results.csv", index=False)
    
