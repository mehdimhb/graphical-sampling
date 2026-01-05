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
        self.best_design_details = {}

        if self.enable_live_plots:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
            self.fig.suptitle('Genetic Algorithm Live Monitoring')
            self._setup_initial_live_plots()

    def _setup_plot_axis(self, ax, title: str, xlabel: str, ylabel: str, clear_first: bool = True):
        """A helper function to set up common properties for a plot axis."""
        if clear_first:
            ax.clear()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

    def _setup_initial_live_plots(self):
        """Setup the initial titles and labels for the live plotting interface."""
        self._setup_plot_axis(self.axes[0, 0], 'Fitness Evolution', 'Generation', 'Fitness', clear_first=False)
        self._setup_plot_axis(self.axes[0, 1], 'Population Diversity', 'Generation', 'Diversity', clear_first=False)
        self._setup_plot_axis(self.axes[0, 2], 'Adaptive Parameters', 'Generation', 'Rate', clear_first=False)
        self._setup_plot_axis(self.axes[1, 0], 'Heap Size Distribution', 'Heap Size', 'Frequency', clear_first=False)
        self._setup_plot_axis(self.axes[1, 1], 'Convergence Speed', 'Generation', 'Rate', clear_first=False)
        # self._setup_plot_axis(self.axes[1, 2], 'Validation Pass Rate', 'Generation', 'Rate (%)', clear_first=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    def record_generation(self,
                         generation: int,
                         population: List[Any],
                         fitness_scores: List[float],
                         ga_instance: Any,
                         ) -> GAMetrics:
        """Record metrics for a single generation."""
        timestamp = time.time() - self.start_time
        fitness_array = np.array(fitness_scores)

        improvement_rate = 0.0
        if len(self.metrics_history) > 0:
            prev_best = self.metrics_history[-1].best_fitness
            if abs(prev_best) > 1e-9:
                improvement_rate = (prev_best - fitness_array.min()) / abs(prev_best)

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
            heap_sizes=[len(design.heap) for design in population if hasattr(design, 'heap')],
            random_state=ga_instance.random_pull,
            mutation_intensity=ga_instance.mutation_intensity,

        )
        self.metrics_history.append(metrics)

        # Track the best design details
        if not self.best_design_details or metrics.best_fitness < self.best_design_details['fitness']:
            best_design_idx = np.argmin(fitness_array)
            self.best_design_details = {
                'fitness': metrics.best_fitness,
                'generation': generation,
                'heap_size': len(population[best_design_idx].heap)
            }


        if self.enable_live_plots and generation > 0 and generation % 5 == 0:
            self.update_live_plots()

        return metrics

    def update_live_plots(self):
        """Update the live plotting interface with the latest data."""
        if not self.metrics_history:
            return
        generations = [m.generation for m in self.metrics_history]

        # Plot 1: Fitness
        self._setup_plot_axis(self.axes[0, 0], 'Fitness Evolution', 'Generation', 'Fitness')
        self.axes[0, 0].plot(generations, [m.best_fitness for m in self.metrics_history], 'b-', label='Best', linewidth=2)
        self.axes[0, 0].plot(generations, [m.mean_fitness for m in self.metrics_history], 'r--', label='Mean', alpha=0.7)
        self.axes[0, 0].legend()

        # Plot 2: Diversity
        self._setup_plot_axis(self.axes[0, 1], 'Population Diversity', 'Generation', 'Diversity')
        self.axes[0, 1].plot(generations, [m.diversity for m in self.metrics_history], 'purple', linewidth=2)

        # Plot 3: Adaptive Parameters
        self._setup_plot_axis(self.axes[0, 2], 'Adaptive Parameters', 'Generation', 'Mutation Rate')
        self.axes[0, 2].plot(generations, [m.mutation_rate for m in self.metrics_history], 'orange', label='Mutation Rate')
        self.axes[0, 2].tick_params(axis='y', labelcolor='orange')
        ax_stagnation = self.axes[0, 2].twinx()
        ax_stagnation.set_ylabel('Stagnation Count', color='gray')
        ax_stagnation.plot(generations, [m.stagnation_counter for m in self.metrics_history], 'gray', linestyle=':', label='Stagnation')
        ax_stagnation.tick_params(axis='y', labelcolor='gray')

        # Plot 4: Heap Size
        current_heap_sizes = self.metrics_history[-1].heap_sizes
        if current_heap_sizes:
            self._setup_plot_axis(self.axes[1, 0], f'Heap Size Distribution (Gen {generations[-1]})', 'Heap Size', 'Frequency')
            self.axes[1, 0].hist(current_heap_sizes, bins=15, alpha=0.75, color='skyblue', edgecolor='black')

        # Plot 5: Convergence Speed
        self._setup_plot_axis(self.axes[1, 1], 'Convergence Speed', 'Generation', 'Rate')
        self.axes[1, 1].plot(generations, [m.convergence_speed for m in self.metrics_history], 'green', label='Convergence Speed')

        # Plot 6: Validation Pass Rate
        # self._setup_plot_axis(self.axes[1, 2], 'Validation Pass Rate', 'Generation', 'Rate (%)')
        # self.axes[1, 2].plot(generations, [m.validation_pass_rate * 100 for m in self.metrics_history], 'brown', label='Validation Pass %')
        # self.axes[1, 2].set_ylim(0, 105)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)

    def generate_final_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive final report with plots and statistics."""
        if not self.metrics_history:
            print("No metrics history to generate a report.")
            return {}

        if self.enable_live_plots:
            plt.ioff()
            plt.close(self.fig)

        # 2x2 grid layout for final report
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Genetic Algorithm Final Analysis Report', fontsize=16, fontweight='bold')
        generations = [m.generation for m in self.metrics_history]

        # Plot 1: Fitness Evolution (Top-Left)
        self._setup_plot_axis(axes[0, 0], 'Fitness Evolution Over Generations', 'Generation', 'Fitness Value', clear_first=False)
        axes[0, 0].plot(generations, [m.best_fitness for m in self.metrics_history], 'b-', label='Best Fitness', linewidth=2)
        axes[0, 0].plot(generations, [m.mean_fitness for m in self.metrics_history], 'r--', label='Mean Fitness', alpha=0.8)
        std_fitness = np.array([m.std_fitness for m in self.metrics_history])
        mean_fitness = np.array([m.mean_fitness for m in self.metrics_history])
        axes[0, 0].fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.15, color='red', label='±1 Std Dev')
        axes[0, 0].legend()

        # Plot 2: Population Diversity (Top-Right)
        self._setup_plot_axis(axes[0, 1], 'Population Diversity', 'Generation', 'Diversity Score', clear_first=False)
        axes[0, 1].plot(generations, [m.diversity for m in self.metrics_history], 'purple', linewidth=2)

        # Plot 3: Adaptive Parameter Evolution (Middle-Left)
        self._setup_plot_axis(axes[1, 0], 'Adaptive Parameter Evolution', 'Generation', 'Mutation Rate', clear_first=False)
        axes[1, 0].plot(generations, [m.mutation_rate for m in self.metrics_history], 'orange', label='Mutation Rate')
        axes[1, 0].tick_params(axis='y', labelcolor='orange')
        ax_stagnation = axes[1, 0].twinx()
        ax_stagnation.set_ylabel('Stagnation Count', color='gray')
        ax_stagnation.plot(generations, [m.stagnation_counter for m in self.metrics_history], 'gray', linestyle=':', alpha=0.8, label='Stagnation')
        ax_stagnation.tick_params(axis='y', labelcolor='gray')

        # Plot 4: Final Statistics Summary (Bottom-Right)
        axes[1, 1].axis('off')
        final_metrics = self.metrics_history[-1]

        # Enhanced Statistics Text - Combined from both examples
        stats_text = f"""
        --- RUN SUMMARY ---
        Total Generations: {final_metrics.generation}
        Total Runtime: {final_metrics.timestamp:.2f} seconds
        Best Fitness Achieved: {self.best_design_details.get('fitness', 'N/A'):.6f}
        Found at Gen: {self.best_design_details.get('generation', 'N/A')}

        --- FINAL POPULATION STATS (Gen {final_metrics.generation}) ---
        Population Size: {final_metrics.population_size}
        Mean Fitness: {final_metrics.mean_fitness:.6f}
        Std Dev Fitness: {final_metrics.std_fitness:.6f}
        Population Diversity: {final_metrics.diversity:.3f}
        Avg Heap Size: {np.mean(final_metrics.heap_sizes):.1f}

        --- FINAL PARAMETERS ---
        Mutation Rate: {final_metrics.mutation_rate:.3f}
        Elitism Rate: {final_metrics.elitism_rate:.3f}
        Selection Pressure: {final_metrics.selection_pressure:.2f}
        Mutation Intensity: {final_metrics.mutation_intensity}
        
        --- PERFORMANCE METRICS ---
        Final Stagnation: {final_metrics.stagnation_counter}
        Best Improvement Rate: {max([m.improvement_rate for m in self.metrics_history]):.4f}
        """
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, fontsize=9,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Final report saved to: {save_path}")
            plt.close(fig)  # Close the figure to prevent it from being shown
        else:
            plt.show()

        return {'best_fitness': self.best_design_details.get('fitness', float('inf'))}

    def save_metrics_to_csv(self, filepath: str):
        """Save all recorded metrics to a CSV file for further analysis."""
        if not self.metrics_history:
            print("No metrics to save.")
            return
        try:
            import pandas as pd
            data = [asdict(m) for m in self.metrics_history]
            for row in data:
                heap_sizes = row.pop('heap_sizes', [])
                row['avg_heap_size'] = np.mean(heap_sizes) if heap_sizes else 0
                row['std_heap_size'] = np.std(heap_sizes) if heap_sizes else 0
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"Metrics saved to: {filepath}")
        except ImportError:
            print("⚠️ pandas not available, using basic CSV implementation.")
            # Fallback to basic csv writer
