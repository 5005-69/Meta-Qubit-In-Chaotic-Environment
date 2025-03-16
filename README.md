# MetaQubit Dynamic Network Simulation

A quantum-inspired approach to dynamic network self-organization using MetaQubits. This project demonstrates how a small quantum-inspired system (MetaQubit) can drive the evolution of a large classical network, leading to self-organized, non-linear behavior in network edge weights.

## Overview

In this project, we use a MetaQubit system (implemented using [PennyLane](https://pennylane.ai/)) to update the edge weights of a complete graph dynamically. The simulation tracks how these weights evolve over multiple simulation steps and repetitions, providing insights into:
- **Network Convergence:** Analysis of how the standard deviation (variance) of edge weights changes over time.
- **Cost Evolution:** Monitoring a simple cost function derived from the MetaQubit output.
- **Edge Weight Distribution:** Histograms at selected simulation steps that show how most edge weights concentrate around zero, with a few outliers driving the overall variance.

The code is modularized into functions that create and update the network, run multiple simulation repetitions, analyze the results, and generate professional-quality visualizations.

## Features

- **Dynamic Network Creation:** Builds a complete graph with random initial edge weights.
- **MetaQubit Integration:** Uses the output of a MetaQubit (a quantum-inspired circuit) to update network edge weights.
- **Multiple Simulations:** Runs repeated simulations to capture statistical behavior across various network configurations.
- **Data Analysis:** Computes mean and standard deviation of edge weights and cost function over simulation steps.
- **Professional Visualizations:** Generates and saves high-resolution plots for:
  - Convergence of the network's edge weight variance.
  - Convergence of the cost function.
  - Histograms showing edge weight distribution at key simulation steps.

## Code Structure
create_dynamic_network(num_nodes):
Creates a complete graph with num_nodes nodes, each edge initialized with a random weight in the range [−1,1].

update_network(network, meta_qubit):
Updates the network's edge weights using the output of the MetaQubit. A simple cost function (defined as 
−
mean(output)
−mean(output)) is also computed.

run_multiple_simulations(meta_qubit, steps, nodes, repetitions):
Runs multiple simulations of network updates over a given number of steps and repetitions, storing edge weights and cost values.

analyze_results(all_weights) & analyze_costs(all_costs):
Compute the mean and standard deviation of the edge weights and cost values across simulation runs.

plot_convergence_of_weights(std_weights, steps):
Plots and saves the evolution of the average standard deviation of the edge weights.

plot_cost_convergence(mean_costs, std_costs, steps):
Plots and saves the evolution of the cost function over simulation steps.

plot_weight_histograms(all_weights, step_indices):
Generates histograms for the edge weight distribution at selected simulation steps.

## Results
The simulation results illustrate that despite random initial conditions and stochastic updates from the MetaQubit:

Network Behavior: The majority of edge weights converge toward a narrow range near zero, while a few outlier edges significantly affect overall variance.
Cost Dynamics: The cost function exhibits a noisy but balanced behavior, reflecting the inherent quantum-inspired stochasticity.
Scalability: While simulations with 100 nodes run efficiently, scaling to larger networks (e.g., 100 nodes) may require further resource optimization.
A detailed report of the simulation results and in-depth analysis is provided in a separate document
