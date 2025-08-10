"""
Monitoring and visualization system for Genetic Algorithm metrics.
This module provides real-time and post-analysis plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class GAMetrics:
    """Data class to store all genetic algorithm metrics for a single generation."""
    generation: int
    timestamp: float
    random_state:bool
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
        
        # Live plotting setup
        if self.enable_live_plots:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
            self.fig.suptitle('Genetic Algorithm Live Monitoring')
            self.setup_live_plots()
    
    def setup_live_plots(self):
        """Setup the live plotting interface."""
        self.axes[0, 0].set_title('Fitness Evolution')
        self.axes[0, 0].set_xlabel('Generation')
        self.axes[0, 0].set_ylabel('Fitness')
        
        self.axes[0, 1].set_title('Fitness Statistics')
        self.axes[0, 1].set_xlabel('Generation')
        self.axes[0, 1].set_ylabel('Fitness')
        
        self.axes[0, 2].set_title('Population Diversity')
        self.axes[0, 2].set_xlabel('Generation')
        self.axes[0, 2].set_ylabel('Diversity')
        
        self.axes[1, 0].set_title('Algorithm Parameters')
        self.axes[1, 0].set_xlabel('Generation')
        self.axes[1, 0].set_ylabel('Rate')
        
        self.axes[1, 1].set_title('Convergence Metrics')
        self.axes[1, 1].set_xlabel('Generation')
        self.axes[1, 1].set_ylabel('Value')
        
        self.axes[1, 2].set_title('Heap Size Distribution')
        self.axes[1, 2].set_xlabel('Heap Size')
        self.axes[1, 2].set_ylabel('Frequency')
    
    def record_generation(self, 
                         generation: int,
                         population: List[Any],
                         fitness_scores: List[float],
                         ga_instance: Any) -> GAMetrics:
        """
        Record metrics for a single generation.
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_scores: Fitness scores for current population
            ga_instance: Reference to the GA instance for parameter access
        
        Returns:
            GAMetrics object containing all recorded metrics
        """
        timestamp = time.time() - self.start_time
        
        # Calculate fitness statistics
        fitness_array = np.array(fitness_scores)
        best_fitness = np.min(fitness_array)
        mean_fitness = np.mean(fitness_array)
        std_fitness = np.std(fitness_array)
        min_fitness = np.min(fitness_array)
        max_fitness = np.max(fitness_array)
        median_fitness = np.median(fitness_array)
        
        # Calculate diversity
        diversity = ga_instance.calculate_population_diversity(population, fitness_scores)
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.metrics_history) > 0:
            prev_best = self.metrics_history[-1].best_fitness
            improvement_rate = (prev_best - best_fitness) / max(abs(prev_best), 1e-10)
        
        # Calculate convergence speed (rate of fitness improvement over last 10 generations)
        convergence_speed = 0.0
        if len(self.metrics_history) >= 10:
            past_fitness = self.metrics_history[-10].best_fitness
            convergence_speed = (past_fitness - best_fitness) / 10
        
        # Get heap sizes
        heap_sizes = [len(design.heap) for design in population]
        
        # Create metrics object
        metrics = GAMetrics(
            generation=generation,
            timestamp=timestamp,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            std_fitness=std_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            median_fitness=median_fitness,
            population_size=len(population),
            diversity=diversity,
            elite_count=max(1, int(ga_instance.elitism_rate * ga_instance.population_size)),
            mutation_rate=ga_instance.mutation_rate,
            elitism_rate=ga_instance.elitism_rate,
            selection_pressure=ga_instance.selection_pressure,
            stagnation_counter=ga_instance.stagnation_counter,
            improvement_rate=improvement_rate,
            convergence_speed=convergence_speed,
            heap_sizes=heap_sizes,
            random_state=ga_instance.random_pull,
            mutation_intensity= ga_instance.mutation_intensity,
        )
        
        self.metrics_history.append(metrics)
        
        # Update live plots if enabled
        if self.enable_live_plots and generation % 5 == 0:  # Update every 5 generations
            self.update_live_plots()
        
        return metrics
    
    def update_live_plots(self):
        """Update the live plotting interface."""
        if not self.metrics_history:
            return
        
        generations = [m.generation for m in self.metrics_history]
        
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # Plot 1: Fitness Evolution
        best_fitness = [m.best_fitness for m in self.metrics_history]
        mean_fitness = [m.mean_fitness for m in self.metrics_history]
        
        self.axes[0, 0].plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
        self.axes[0, 0].plot(generations, mean_fitness, 'r--', label='Mean', alpha=0.7)
        self.axes[0, 0].set_title('Fitness Evolution')
        self.axes[0, 0].set_xlabel('Generation')
        self.axes[0, 0].set_ylabel('Fitness')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fitness Statistics (with error bars)
        std_fitness = [m.std_fitness for m in self.metrics_history]
        min_fitness = [m.min_fitness for m in self.metrics_history]
        max_fitness = [m.max_fitness for m in self.metrics_history]
        
        self.axes[0, 1].fill_between(generations, min_fitness, max_fitness, alpha=0.2, label='Min-Max Range')
        self.axes[0, 1].errorbar(generations, mean_fitness, yerr=std_fitness, 
                                errorevery=5, capsize=3, label='Mean ± Std')
        self.axes[0, 1].plot(generations, best_fitness, 'g-', label='Best', linewidth=2)
        self.axes[0, 1].set_title('Fitness Statistics')
        self.axes[0, 1].set_xlabel('Generation')
        self.axes[0, 1].set_ylabel('Fitness')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Population Diversity
        diversity = [m.diversity for m in self.metrics_history]
        self.axes[0, 2].plot(generations, diversity, 'purple', linewidth=2)
        self.axes[0, 2].set_title('Population Diversity')
        self.axes[0, 2].set_xlabel('Generation')
        self.axes[0, 2].set_ylabel('Diversity')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Algorithm Parameters
        mutation_rates = [m.mutation_rate for m in self.metrics_history]
        elitism_rates = [m.elitism_rate for m in self.metrics_history]
        
        self.axes[1, 0].plot(generations, mutation_rates, 'orange', label='Mutation Rate')
        self.axes[1, 0].plot(generations, elitism_rates, 'cyan', label='Elitism Rate')
        self.axes[1, 0].set_title('Algorithm Parameters')
        self.axes[1, 0].set_xlabel('Generation')
        self.axes[1, 0].set_ylabel('Rate')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Convergence Metrics
        stagnation = [m.stagnation_counter for m in self.metrics_history]
        improvement = [m.improvement_rate for m in self.metrics_history]
        
        ax5_twin = self.axes[1, 1].twinx()
        self.axes[1, 1].plot(generations, stagnation, 'red', label='Stagnation Counter')
        ax5_twin.plot(generations, improvement, 'green', label='Improvement Rate')
        self.axes[1, 1].set_title('Convergence Metrics')
        self.axes[1, 1].set_xlabel('Generation')
        self.axes[1, 1].set_ylabel('Stagnation Count', color='red')
        ax5_twin.set_ylabel('Improvement Rate', color='green')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Current Heap Size Distribution
        if self.metrics_history:
            current_heap_sizes = self.metrics_history[-1].heap_sizes
            self.axes[1, 2].hist(current_heap_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            self.axes[1, 2].set_title(f'Heap Size Distribution (Gen {generations[-1]})')
            self.axes[1, 2].set_xlabel('Heap Size')
            self.axes[1, 2].set_ylabel('Frequency')
            self.axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def generate_final_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive final report with plots and statistics.
        
        Args:
            save_path: Path to save the report plots (optional)
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Genetic Algorithm Final Analysis Report', fontsize=16, fontweight='bold')
        
        generations = [m.generation for m in self.metrics_history]
        
        # Plot 1: Complete Fitness Evolution
        best_fitness = [m.best_fitness for m in self.metrics_history]
        mean_fitness = [m.mean_fitness for m in self.metrics_history]
        std_fitness = [m.std_fitness for m in self.metrics_history]
        
        axes[0, 0].plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        axes[0, 0].plot(generations, mean_fitness, 'r--', label='Mean Fitness', alpha=0.8)
        axes[0, 0].fill_between(generations, 
                               np.array(mean_fitness) - np.array(std_fitness),
                               np.array(mean_fitness) + np.array(std_fitness),
                               alpha=0.2, color='red', label='±1 Std')
        axes[0, 0].set_title('Fitness Evolution Over Generations')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter Adaptation
        mutation_rates = [m.mutation_rate for m in self.metrics_history]
        diversity = [m.diversity for m in self.metrics_history]
        
        ax2_twin = axes[0, 1].twinx()
        axes[0, 1].plot(generations, mutation_rates, 'orange', linewidth=2, label='Mutation Rate')
        ax2_twin.plot(generations, diversity, 'purple', linewidth=2, label='Diversity')
        axes[0, 1].set_title('Parameter Adaptation')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Mutation Rate', color='orange')
        ax2_twin.set_ylabel('Diversity', color='purple')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Convergence Analysis
        improvement_rates = [m.improvement_rate for m in self.metrics_history]
        convergence_speeds = [m.convergence_speed for m in self.metrics_history]
        
        axes[1, 0].plot(generations, improvement_rates, 'green', linewidth=2, label='Improvement Rate')
        axes[1, 0].plot(generations, convergence_speeds, 'blue', linewidth=2, label='Convergence Speed')
        axes[1, 0].set_title('Convergence Analysis')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Stagnation Tracking
        # stagnation = [m.stagnation_counter for m in self.metrics_history]
        # axes[1, 1].plot(generations, stagnation, 'red', linewidth=2)
        # axes[1, 1].set_title('Stagnation Counter Over Time')
        # axes[1, 1].set_xlabel('Generation')
        # axes[1, 1].set_ylabel('Stagnation Count')
        # axes[1, 1].grid(True, alpha=0.3)
        #
        # Plot 5: Heap Size Evolution
        avg_heap_sizes = [np.mean(m.heap_sizes) for m in self.metrics_history]
        std_heap_sizes = [np.std(m.heap_sizes) for m in self.metrics_history]
        
        axes[2, 0].plot(generations, avg_heap_sizes, 'teal', linewidth=2, label='Average Heap Size')
        axes[2, 0].fill_between(generations,
                               np.array(avg_heap_sizes) - np.array(std_heap_sizes),
                               np.array(avg_heap_sizes) + np.array(std_heap_sizes),
                               alpha=0.3, color='teal', label='±1 Std')
        axes[2, 0].set_title('Population Heap Size Evolution')
        axes[2, 0].set_xlabel('Generation')
        axes[2, 0].set_ylabel('Heap Size')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Final Statistics Summary (text-based)
        axes[2, 1].axis('off')
        final_metrics = self.metrics_history[-1]
        
        stats_text = f"""
        FINAL ALGORITHM STATISTICS
        Number of population: {final_metrics.population_size}
        State of Randomness: {final_metrics.random_state:.2f}
        Mutation intensity: {final_metrics.mutation_intensity:.3f}
        Best Fitness Achieved: {final_metrics.best_fitness:.6f}
        Total Generations: {final_metrics.generation}
        Total Runtime: {final_metrics.timestamp:.2f} seconds
        
        Final Population Stats:
        • Mean Fitness: {final_metrics.mean_fitness:.6f}
        • Std Fitness: {final_metrics.std_fitness:.6f}
        • Population Diversity: {final_metrics.diversity:.3f}
        
        Final Parameters:
        • Mutation Rate: {final_metrics.mutation_rate:.3f}
        • Elitism Rate: {final_metrics.elitism_rate:.3f}
        • Selection Pressure: {final_metrics.selection_pressure:.2f}
        
        Performance Metrics:
        • Final Stagnation: {final_metrics.stagnation_counter}
        • Avg Heap Size: {np.mean(final_metrics.heap_sizes):.1f}
        • Best Improvement: {max(improvement_rates):.6f}
        """
        
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Final report saved to: {save_path}")
        
        plt.show()
        
        # Return summary statistics
        return {
            'total_generations': final_metrics.generation,
            'total_runtime': final_metrics.timestamp,
            'best_fitness': final_metrics.best_fitness,
            'final_diversity': final_metrics.diversity,
            'convergence_rate': np.mean(convergence_speeds[-10:]) if len(convergence_speeds) >= 10 else 0,
            'average_improvement': np.mean(improvement_rates),
            'final_stagnation': final_metrics.stagnation_counter
        }
    
    def save_metrics_to_csv(self, filepath: str):
        """Save all recorded metrics to a CSV file for further analysis."""
        if not self.metrics_history:
            print("No metrics to save.")
            return
        
        try:
            import pandas as pd
            use_pandas = True
        except ImportError:
            use_pandas = False
            print("⚠️  pandas not available, using basic CSV implementation")
        
        if use_pandas:
            # Convert metrics to dictionary format
            data = []
            for m in self.metrics_history:
                row = {
                    'generation': m.generation,
                    'timestamp': m.timestamp,
                    'best_fitness': m.best_fitness,
                    'mean_fitness': m.mean_fitness,
                    'std_fitness': m.std_fitness,
                    'min_fitness': m.min_fitness,
                    'max_fitness': m.max_fitness,
                    'median_fitness': m.median_fitness,
                    'population_size': m.population_size,
                    'diversity': m.diversity,
                    'elite_count': m.elite_count,
                    'mutation_rate': m.mutation_rate,
                    'elitism_rate': m.elitism_rate,
                    'selection_pressure': m.selection_pressure,
                    'stagnation_counter': m.stagnation_counter,
                    'improvement_rate': m.improvement_rate,
                    'convergence_speed': m.convergence_speed,
                    'avg_heap_size': np.mean(m.heap_sizes),
                    'std_heap_size': np.std(m.heap_sizes),
                    'validation_pass_rate': m.validation_pass_rate
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            # Basic CSV implementation without pandas
            import csv
            
            headers = [
                'generation', 'timestamp', 'best_fitness', 'mean_fitness', 'std_fitness',
                'min_fitness', 'max_fitness', 'median_fitness', 'population_size',
                'diversity', 'elite_count', 'mutation_rate', 'elitism_rate',
                'selection_pressure', 'stagnation_counter', 'improvement_rate',
                'convergence_speed', 'avg_heap_size', 'std_heap_size', 'validation_pass_rate'
            ]
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                for m in self.metrics_history:
                    row = [
                        m.generation, m.timestamp, m.best_fitness, m.mean_fitness, m.std_fitness,
                        m.min_fitness, m.max_fitness, m.median_fitness, m.population_size,
                        m.diversity, m.elite_count, m.mutation_rate, m.elitism_rate,
                        m.selection_pressure, m.stagnation_counter, m.improvement_rate,
                        m.convergence_speed, np.mean(m.heap_sizes), np.std(m.heap_sizes),
                        m.validation_pass_rate
                    ]
                    writer.writerow(row)
        
        print(f"Metrics saved to: {filepath}")
