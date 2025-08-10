"""
Monitoring and visualization system for Genetic Algorithm metrics.
This module provides real-time and post-analysis plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import time

@dataclass
class GAMetrics:
    """Data class to store all genetic algorithm metrics for a single generation."""
    generation: int
    timestamp: float
    random_state: bool
    mutation_intensity: int
    # Fitness metrics
    best_fitness: float
    mean_fitness: np.floating
    std_fitness: float
    min_fitness: float
    max_fitness: float
    median_fitness: float
    # Population metrics
    population_size: int
    diversity: float
    elite_count: int
    # Algorithm parameters
    mutation_rate: float
    elitism_rate: float
    selection_pressure: float
    # Performance metrics
    stagnation_counter: int
    improvement_rate: float
    convergence_speed: float
    # Additional metrics
    heap_sizes: List[int] = field(default_factory=list)
    validation_pass_rate: float = 1.0


class GAMonitor:
    """
    Comprehensive monitoring system for Genetic Algorithm performance.
    Tracks metrics and provides visualization capabilities.
    """

    def __init__(self, enable_live_plots: bool = False, save_data: bool = True):
        self.enable_live_plots = enable_live_plots
        self.save_data = save_data
        self.metrics_history: List[GAMetrics] = []
        self.start_time = time.time()

        if self.enable_live_plots:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
            self.fig.suptitle('Genetic Algorithm Live Monitoring')
            self._setup_initial_live_plots()

    def _setup_plot_axis(self, ax, title: str, xlabel: str, ylabel: str, clear_first: bool = True):
        """A helper function to set up common properties for a plot axis."""
        if clear_first:
            ax.clear()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)

    def _setup_initial_live_plots(self):
        """Setup the initial titles and labels for the live plotting interface."""
        self._setup_plot_axis(self.axes[0, 0], 'Fitness Evolution', 'Generation', 'Fitness', clear_first=False)
        self._setup_plot_axis(self.axes[0, 1], 'Fitness Statistics', 'Generation', 'Fitness', clear_first=False)
        self._setup_plot_axis(self.axes[0, 2], 'Population Diversity', 'Generation', 'Diversity', clear_first=False)
        self._setup_plot_axis(self.axes[1, 0], 'Algorithm Parameters', 'Generation', 'Rate', clear_first=False)
        self._setup_plot_axis(self.axes[1, 1], 'Convergence Metrics', 'Generation', 'Value', clear_first=False)
        self._setup_plot_axis(self.axes[1, 2], 'Heap Size Distribution', 'Heap Size', 'Frequency', clear_first=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    def record_generation(self,
                         generation: int,
                         population: List[Any],
                         fitness_scores: List[float],
                         ga_instance: Any) -> GAMetrics:
        """
        Record metrics for a single generation.
        """
        timestamp = time.time() - self.start_time
        fitness_array = np.array(fitness_scores)

        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.metrics_history) > 0:
            prev_best = self.metrics_history[-1].best_fitness
            improvement_rate = (prev_best - fitness_array.min()) / max(abs(prev_best), 1e-10)

        # Calculate convergence speed
        convergence_speed = 0.0
        if len(self.metrics_history) >= 10:
            past_fitness = self.metrics_history[-10].best_fitness
            convergence_speed = (past_fitness - fitness_array.min()) / 10

        metrics = GAMetrics(
            generation=generation,
            timestamp=timestamp,
            best_fitness=fitness_array.min(),
            mean_fitness=np.mean(fitness_array),
            std_fitness=np.std(fitness_array),
            min_fitness=fitness_array.min(),
            max_fitness=fitness_array.max(),
            median_fitness=np.median(fitness_array),
            population_size=len(population),
            diversity=ga_instance.calculate_population_diversity(population, fitness_scores),
            elite_count=max(1, int(ga_instance.elitism_rate * ga_instance.population_size)),
            mutation_rate=ga_instance.mutation_rate,
            elitism_rate=ga_instance.elitism_rate,
            selection_pressure=ga_instance.selection_pressure,
            stagnation_counter=ga_instance.stagnation_counter,
            improvement_rate=improvement_rate,
            convergence_speed=convergence_speed,
            heap_sizes=[len(design.heap) for design in population],
            random_state=ga_instance.random_pull,
            mutation_intensity=ga_instance.mutation_intensity,
        )

        self.metrics_history.append(metrics)

        if self.enable_live_plots and generation % 5 == 0:
            self.update_live_plots()

        return metrics

    def update_live_plots(self):
        """Update the live plotting interface with the latest data."""
        if not self.metrics_history:
            return

        generations = [m.generation for m in self.metrics_history]

        # Plot 1: Fitness Evolution
        self._setup_plot_axis(self.axes[0, 0], 'Fitness Evolution', 'Generation', 'Fitness')
        self.axes[0, 0].plot(generations, [m.best_fitness for m in self.metrics_history], 'b-', label='Best', linewidth=2)
        self.axes[0, 0].plot(generations, [m.mean_fitness for m in self.metrics_history], 'r--', label='Mean', alpha=0.7)
        self.axes[0, 0].legend()

        # Plot 2: Fitness Statistics
        self._setup_plot_axis(self.axes[0, 1], 'Fitness Statistics', 'Generation', 'Fitness')
        self.axes[0, 1].fill_between(generations, [m.min_fitness for m in self.metrics_history], [m.max_fitness for m in self.metrics_history], color='blue', alpha=0.1, label='Min-Max Range')
        self.axes[0, 1].errorbar(generations, [m.mean_fitness for m in self.metrics_history], yerr=[m.std_fitness for m in self.metrics_history], errorevery=5, capsize=3, label='Mean ± Std', fmt='r--')
        self.axes[0, 1].legend()

        # Plot 3: Population Diversity
        self._setup_plot_axis(self.axes[0, 2], 'Population Diversity', 'Generation', 'Diversity')
        self.axes[0, 2].plot(generations, [m.diversity for m in self.metrics_history], 'purple', linewidth=2)

        # Plot 4: Algorithm Parameters
        self._setup_plot_axis(self.axes[1, 0], 'Algorithm Parameters', 'Generation', 'Rate')
        self.axes[1, 0].plot(generations, [m.mutation_rate for m in self.metrics_history], 'orange', label='Mutation Rate')
        self.axes[1, 0].plot(generations, [m.elitism_rate for m in self.metrics_history], 'cyan', label='Elitism Rate')
        self.axes[1, 0].legend()

        # Plot 5: Convergence Metrics
        self._setup_plot_axis(self.axes[1, 1], 'Convergence Metrics', 'Generation', 'Stagnation Count')
        self.axes[1, 1].plot(generations, [m.stagnation_counter for m in self.metrics_history], 'red', label='Stagnation')
        self.axes[1, 1].tick_params(axis='y', labelcolor='red')
        ax5_twin = self.axes[1, 1].twinx()
        ax5_twin.set_ylabel('Improvement Rate', color='green')
        ax5_twin.plot(generations, [m.improvement_rate for m in self.metrics_history], 'green', linestyle='--', label='Improvement')
        ax5_twin.tick_params(axis='y', labelcolor='green')

        # Plot 6: Heap Size Distribution
        current_heap_sizes = self.metrics_history[-1].heap_sizes
        self._setup_plot_axis(self.axes[1, 2], f'Heap Size Distribution (Gen {generations[-1]})', 'Heap Size', 'Frequency')
        self.axes[1, 2].hist(current_heap_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

    def generate_final_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive final report with plots and statistics."""
        if not self.metrics_history:
            print("No metrics history to generate a report.")
            return {}

        if self.enable_live_plots:
            plt.ioff() # Turn off interactive mode for final plot
            plt.close(self.fig) # Close the live plot window

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Genetic Algorithm Final Analysis Report', fontsize=16, fontweight='bold')

        generations = [m.generation for m in self.metrics_history]

        # Plotting logic remains similar, but now uses the helper
        # Plot 1: Complete Fitness Evolution
        self._setup_plot_axis(axes[0, 0], 'Fitness Evolution Over Generations', 'Generation', 'Fitness Value', clear_first=False)
        axes[0, 0].plot(generations, [m.best_fitness for m in self.metrics_history], 'b-', label='Best Fitness', linewidth=2)
        axes[0, 0].plot(generations, [m.mean_fitness for m in self.metrics_history], 'r--', label='Mean Fitness', alpha=0.8)
        axes[0, 0].fill_between(generations, np.array([m.mean_fitness for m in self.metrics_history]) - np.array([m.std_fitness for m in self.metrics_history]), np.array([m.mean_fitness for m in self.metrics_history]) + np.array([m.std_fitness for m in self.metrics_history]), alpha=0.2, color='red', label='±1 Std')
        axes[0, 0].legend()

        # ... other plots would follow a similar pattern ...

        # Plot 6: Final Statistics Summary (text-based)
        axes[2, 1].axis('off')
        final_metrics = self.metrics_history[-1]
        stats_text = f"""
        FINAL ALGORITHM STATISTICS
        Number of population: {final_metrics.population_size}
        State of Randomness: {final_metrics.random_state}
        Mutation intensity: {final_metrics.mutation_intensity}
        Best Fitness Achieved: {final_metrics.best_fitness:.6f}
        Total Generations: {final_metrics.generation}
        Total Runtime: {final_metrics.timestamp:.2f} seconds
        
        Final Population Stats:
        • Mean Fitness: {final_metrics.mean_fitness:.6f}
        • Std Fitness: {final_metrics.std_fitness:.6f}
        • Population Diversity: {final_metrics.diversity:.3f}
        """
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Final report saved to: {save_path}")

        plt.show()

        return {
            'total_generations': final_metrics.generation,
            'total_runtime': final_metrics.timestamp,
            'best_fitness': final_metrics.best_fitness,
        }

    def save_metrics_to_csv(self, filepath: str):
        """Save all recorded metrics to a CSV file for further analysis."""
        if not self.metrics_history:
            print("No metrics to save.")
            return

        try:
            import pandas as pd

            # Use asdict for cleaner conversion from dataclass to dict
            data = [asdict(m) for m in self.metrics_history]

            # Post-process the data to calculate aggregate stats
            for row in data:
                heap_sizes = row.pop('heap_sizes', []) # Remove list and get value
                row['avg_heap_size'] = np.mean(heap_sizes) if heap_sizes else 0
                row['std_heap_size'] = np.std(heap_sizes) if heap_sizes else 0

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"Metrics saved to: {filepath}")

        except ImportError:
            print("⚠️ pandas not available, using basic CSV implementation.")
            import csv

            # Get headers from the dataclass fields + new aggregate fields
            headers = [f.name for f in field(GAMetrics)] + ['avg_heap_size', 'std_heap_size']
            headers.remove('heap_sizes')

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for m in self.metrics_history:
                    row_dict = asdict(m)
                    heap_sizes = row_dict.pop('heap_sizes', [])
                    row_dict['avg_heap_size'] = np.mean(heap_sizes) if heap_sizes else 0
                    row_dict['std_heap_size'] = np.std(heap_sizes) if heap_sizes else 0
                    writer.writerow([row_dict.get(h, '') for h in headers])
            print(f"Metrics saved to: {filepath}")
