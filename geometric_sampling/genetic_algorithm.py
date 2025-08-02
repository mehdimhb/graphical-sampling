"""
Geometric Sampling Genetic Algorithm Implementation
Based on the paper's geometric sampling framework for optimal variance reduction.
"""
import numpy as np
from typing import List, Tuple

from geometric_sampling.design import DesignGenetic
from geometric_sampling.GeneticOptimizer import GeneticOptimizer
from geometric_sampling.criteria import VarNHT


class GeometricSamplingGA:
    # Algorithm constants
    DEFAULT_MAX_GENERATIONS = 1000
    VALIDATION_TOLERANCE = 1e-6
    EARLY_SEARCH_PHASE = 0.3
    LATE_SEARCH_PHASE = 0.7
    
    def __init__(
        self, 
        inclusions: np.ndarray, 
        auxiliary_var: np.ndarray,
        population_size: int = 40,
        elitism_rate: float = 0.15,
        mutation_intensity: int = 3,
        use_partitions: bool = True,
        random_pull: bool = False,
        adaptive_parameters: bool = True     # New: enable adaptive control
    ):
        """
        Initialize Geometric Sampling Genetic Algorithm
        
        Args:
            inclusions: First-order inclusion probabilities for each unit
            auxiliary_var: Auxiliary variable for variance calculation (for VarNHT criterion)
            population_size: Number of designs in population
            elitism_rate: Initial fraction of best designs to preserve each generation
            mutation_intensity: Number of segment interchanges per mutation
            use_partitions: Whether to use partition constraints during mutation
            adaptive_parameters: Whether to use adaptive parameter control
        """

        # Core algorithm parameters
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.initial_population_size = population_size
        self.population_size = population_size
        self.initial_elitism_rate = elitism_rate
        self.elitism_rate = elitism_rate
        self.initial_mutation_intensity = mutation_intensity
        self.mutation_intensity = mutation_intensity
        self.use_partitions = use_partitions
        self.adaptive_parameters = adaptive_parameters
        
        # Adaptive parameter control settings
        self.mutation_rate = 0.8 if adaptive_parameters else 1.0  # Start high for exploration
        self.min_mutation_rate = 0.1
        self.max_mutation_rate = 0.9
        self.improvement_threshold = 1e-6  # Minimum improvement to be considered progress
        self.stagnation_counter = 0
        
        # Adaptive diversity threshold settings
        self.initial_diversity_threshold = 0.03
        self.diversity_threshold = 0.03  # Dynamic: will adapt based on search phase
        self.min_diversity_threshold = 0.015  # More aggressive threshold for late search
        self.max_diversity_threshold = 0.1   # Conservative threshold for early search
        
        # Adaptive mutation intensity settings
        self.min_mutation_intensity = 1
        self.max_mutation_intensity = min(8, mutation_intensity * 2)  # Domain-safe upper bound
        
        # Performance tracking
        self.generation = 0
        self.last_best_fitness = float('inf')
        self.diversity_history = []
        self.parameter_history = []
        
        # Enhanced statistics tracking
        self.fitness_statistics = {
            'best_per_generation': [],
            'worst_per_generation': [],
            'mean_per_generation': [],
            'std_per_generation': []
        }
        self.population_statistics = {
            'sample_count_mean': [],
            'sample_count_std': [],
            'sample_count_min': [],
            'sample_count_max': []
        }
        self.convergence_statistics = {
            'fitness_improvements': [],
            'stagnation_periods': [],
            'diversity_drops': [],
            'parameter_changes': []
        }
        
        # Algorithm components initialization
        self.optimizer = GeneticOptimizer()
        self.criterion = VarNHT(auxiliary_var, inclusions)
        self.rng = np.random.default_rng()
        
        # Setup partitions and border units if requested
        if self.use_partitions:
            self.partitions, self.border_units = self.optimizer.partition_design(
                inclusions.tolist(), 2
            )
        else:
            self.partitions = None
            self.border_units = None
        
        # Algorithm state tracking
        self.fitness_history = []
        self.best_design = None
        self.best_fitness = float('inf')
        self.random_pull = random_pull

    def create_initial_population(self) -> List[DesignGenetic]:
        """
        Create initial population starting with fixed-size design and adding entropy variants
        """
        population = []
        
        # First design: Perfect fixed-size arrangement
        base_design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(42))
        population.append(base_design)
        
        # Create variants by applying different amounts of segment interchanges
        for i in range(self.population_size - 1):
            # Use different random seeds for diversity
            variant = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(i))
            
            # Apply random number of segment interchanges to increase entropy
            num_interchanges = self.rng.integers(1, 50)  # Varying amounts of mixing
            for _ in range(num_interchanges):
                if len(variant.heap) >= 2:
                    variant.iterate(
                        random_pull=self.random_pull,
                        switch_coefficient=0.5,
                        partitions=self.partitions,
                        border_units=self.border_units
                    )
            
            # Merge identical samples to clean up the design
            variant.merge_identical()
            population.append(variant)
        
        return population

    def evaluate_fitness(self, design: DesignGenetic) -> float:
        """
        Calculate fitness using VarNHT criterion (C1 from paper)
        Lower values indicate better designs (lower variance)
        """
        try:
            # Calculate VarNHT criterion - lower is better
            variance = self.criterion(design)  # Use __call__ method
            return variance
        except Exception as e:
            # If calculation fails, return very high penalty
            print(f"Fitness calculation failed: {e}")
            return 1e10

    def tournament_selection(self, population: List[DesignGenetic], fitness_scores: List[float]) -> DesignGenetic:
        """
        Tournament selection: pick 2 random designs, return the one with lower variance
        """
        idx1, idx2 = self.rng.choice(len(population), size=2, replace=False)
        
        # Lower variance is better
        if fitness_scores[idx1] <= fitness_scores[idx2]:
            return population[idx1].copy()
        else:
            return population[idx2].copy()

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        """
        Apply segment interchange operations with adaptive mutation rate
        Each iterate() call performs one segment swap between two samples
        """
        # Adaptive mutation: apply mutation based on current mutation rate
        if self.rng.random() > self.mutation_rate:
            return design.copy()
        
        mutated = design.copy()
        
        # Apply several segment interchanges to increase entropy
        for _ in range(self.mutation_intensity):
            if len(mutated.heap) >= 2:
                mutated.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.rng.uniform(0.3, 0.7),  # Vary interchange size
                    partitions=self.partitions,
                    border_units=self.border_units
                )
        
        # Clean up the design by merging identical samples
        mutated.merge_identical()
        return mutated

    def crossover(self, parent1: DesignGenetic, parent2: DesignGenetic) -> Tuple[DesignGenetic, DesignGenetic]:
        """
        Create offspring by combining geometric arrangements from two parents
        """
        try:
            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2], random_pull=self.random_pull)
            
            # Clean up children
            child1.merge_identical()
            child2.merge_identical()
            
            return child1, child2
        except Exception as e:
            print(f"Crossover failed: {e}")
            # Return mutated copies of parents as fallback
            return self.mutate_design(parent1), self.mutate_design(parent2)

    def select_elites(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        """
        Select the best designs to preserve in next generation
        """
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        return [population[i].copy() for i in elite_indices]

    def validate_design(self, design: DesignGenetic) -> bool:
        """
        Validate that design maintains inclusion probability constraints
        """
        # Calculate actual inclusion probabilities
        id_probs = {}
        for sample in design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability
        
        # Check if all units are present and probabilities match
        for i, expected_prob in enumerate(self.inclusions):
            actual_prob = id_probs.get(i, 0)
            if abs(actual_prob - expected_prob) > self.VALIDATION_TOLERANCE:
                return False
        
        return True

    def calculate_population_diversity(self, population: List[DesignGenetic], fitness_scores: List[float]) -> float:
        """
        Calculate population diversity based on fitness variance and design structure differences
        Returns value between 0 (no diversity) and 1 (high diversity)
        """
        if len(population) < 2:
            return 0.0
        
        # Fitness diversity component
        fitness_std = np.std(fitness_scores)
        fitness_mean = np.mean(fitness_scores)
        fitness_diversity = fitness_std / (fitness_mean + 1e-10) if fitness_mean > 0 else 0
        
        # Structural diversity component (number of samples in each design)
        sample_counts = [len(design.heap) for design in population]
        structural_diversity = np.std(sample_counts) / (np.mean(sample_counts) + 1e-10)
        
        # Combined diversity score (normalized to 0-1)
        combined_diversity = (fitness_diversity + structural_diversity) / 2
        return min(1.0, combined_diversity)

    def adapt_parameters(self, current_fitness: float, population_diversity: float, generation: int, max_generations: int = None):
        """
        Adapt algorithm parameters based on current performance and population state
        Research-based adaptive control following established GA principles
        Enhanced with adaptive diversity_threshold and mutation_intensity
        """
        if not self.adaptive_parameters:
            return
        
        if max_generations is None:
            max_generations = self.DEFAULT_MAX_GENERATIONS
        
        # Check for improvement
        improvement = self.last_best_fitness - current_fitness
        has_improvement = improvement > self.improvement_threshold
        
        if has_improvement:
            self.stagnation_counter = 0
            self.last_best_fitness = current_fitness
        else:
            self.stagnation_counter += 1
        
        # Calculate search phase indicators
        search_progress = generation / max_generations if max_generations > 0 else 0.5
        
        # Adaptive diversity threshold (Research principle: tighter control in later phases)
        if search_progress < self.EARLY_SEARCH_PHASE:
            # Early phase: conservative threshold to allow natural exploration
            target_threshold = self.max_diversity_threshold * (1.0 - search_progress * 0.5)
        elif search_progress < self.LATE_SEARCH_PHASE:
            # Middle phase: standard threshold with slight tightening
            target_threshold = self.initial_diversity_threshold * (1.0 - search_progress * 0.3)
        else:
            # Late phase: aggressive threshold for exploitation
            target_threshold = self.min_diversity_threshold + (self.initial_diversity_threshold - self.min_diversity_threshold) * (1.0 - search_progress)
        
        # Adjust based on current performance
        if self.stagnation_counter > 15:
            # Long stagnation: relax threshold to encourage more drastic action
            target_threshold *= 1.5
        elif has_improvement and population_diversity > self.diversity_threshold:
            # Good progress with diversity: maintain current sensitivity
            target_threshold = self.diversity_threshold
        
        # Smooth adaptation
        self.diversity_threshold = 0.7 * self.diversity_threshold + 0.3 * target_threshold
        self.diversity_threshold = np.clip(self.diversity_threshold, self.min_diversity_threshold, self.max_diversity_threshold)
        
        # Adaptive mutation intensity (Research principle: adaptive disruption strength)
        if population_diversity < self.diversity_threshold and self.stagnation_counter > 10:
            # Low diversity + stagnation: increase disruption
            self.mutation_intensity = min(self.max_mutation_intensity, self.mutation_intensity + 1)
        elif has_improvement and population_diversity > self.diversity_threshold * 1.5:
            # Good progress with high diversity: reduce disruption for fine-tuning
            self.mutation_intensity = max(self.min_mutation_intensity, self.mutation_intensity - 1)
        elif search_progress > 0.8:
            # Late search: conservative mutations for exploitation
            target_intensity = max(self.min_mutation_intensity, int(self.initial_mutation_intensity * 0.7))
            self.mutation_intensity = max(target_intensity, self.mutation_intensity - 1) if self.mutation_intensity > target_intensity else self.mutation_intensity
        
        # Ensure mutation intensity stays within safe bounds
        self.mutation_intensity = np.clip(self.mutation_intensity, self.min_mutation_intensity, self.max_mutation_intensity)
        
        # Adaptive mutation rate (key research principle: exploration → exploitation)
        if population_diversity < self.diversity_threshold:
            # Low diversity: increase mutation for exploration
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.1)
        elif has_improvement:
            # Making progress: gradually reduce mutation for exploitation
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.98)
        else:
            # Stagnant: increase mutation slightly
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.05)
        
        # Adaptive elitism rate
        if has_improvement and population_diversity > self.diversity_threshold:
            # Good progress with diversity: increase elitism to preserve gains
            self.elitism_rate = min(0.3, self.initial_elitism_rate * 1.2)
        elif population_diversity < self.diversity_threshold:
            # Low diversity: reduce elitism to allow more exploration
            self.elitism_rate = max(0.05, self.initial_elitism_rate * 0.8)
        else:
            # Default: gradually return to initial rate
            target_rate = self.initial_elitism_rate
            self.elitism_rate = 0.9 * self.elitism_rate + 0.1 * target_rate
        
        # Adaptive population size (expand if needed for diversity)
        if self.stagnation_counter > 20 and population_diversity < self.diversity_threshold:
            self.population_size = min(80, int(self.population_size * 1.1))
        elif self.stagnation_counter == 0 and population_diversity > 0.5:
            # Good progress: can reduce population for efficiency
            self.population_size = max(self.initial_population_size, int(self.population_size * 0.95))
        
        # Record parameter history for analysis
        self.parameter_history.append({
            'generation': generation,
            'mutation_rate': self.mutation_rate,
            'elitism_rate': self.elitism_rate,
            'population_size': self.population_size,
            'diversity': population_diversity,
            'diversity_threshold': self.diversity_threshold,
            'mutation_intensity': self.mutation_intensity,
            'stagnation': self.stagnation_counter,
            'improvement': improvement,
            'search_progress': search_progress
        })

    def run(self, max_generations: int = 100, verbose: bool = True) -> DesignGenetic:
        """
        Run the genetic algorithm to find optimal geometric sampling design
        """
        if verbose:
            print("Initializing Geometric Sampling Genetic Algorithm...")
            print(f"Population size: {self.population_size}")
            print(f"Inclusion probabilities: {self.inclusions}")
            print(f"Auxiliary variable: {self.auxiliary_var}")
            print(f"Using partitions: {self.use_partitions}")
            print("="*50)
        
        # Initialize population
        population = self.create_initial_population()
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate fitness for all designs
            fitness_scores = [self.evaluate_fitness(design) for design in population]
            
            # Track best design
            best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_design = population[best_idx].copy()
            
            self.fitness_history.append(current_best_fitness)
            
            # Calculate population diversity and adapt parameters
            population_diversity = self.calculate_population_diversity(population, fitness_scores)
            self.diversity_history.append(population_diversity)
            
            # Collect detailed statistics
            self._collect_generation_statistics(population, fitness_scores, population_diversity)
            
            self.adapt_parameters(current_best_fitness, population_diversity, generation, max_generations)
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.8f}, "
                      f"Diversity = {population_diversity:.3f}, "
                      f"MutRate = {self.mutation_rate:.3f}, "
                      f"Elite = {self.elitism_rate:.3f}, "
                      f"DivThresh = {self.diversity_threshold:.4f}, "
                      f"MutInt = {self.mutation_intensity}")
            
            # Adjust population size if needed
            population, fitness_scores = self._adjust_population_size(population, fitness_scores)
            
            # Create next generation
            population = self._create_next_generation(population, fitness_scores)
        
        if verbose:
            print("="*50)
            print(f"Algorithm completed after {max_generations} generations")
            print(f"Best fitness achieved: {self.best_fitness:.8f}")
            
            # Validate final best design
            if self.validate_design(self.best_design):
                print("✅ Best design passes validation!")
            else:
                print("❌ Best design failed validation!")
        
        return self.best_design

    def _collect_generation_statistics(self, population: List[DesignGenetic], fitness_scores: List[float], diversity: float):
        """
        Collect comprehensive statistics about the current generation
        """
        # Fitness statistics
        self.fitness_statistics['best_per_generation'].append(np.min(fitness_scores))
        self.fitness_statistics['worst_per_generation'].append(np.max(fitness_scores))
        self.fitness_statistics['mean_per_generation'].append(np.mean(fitness_scores))
        self.fitness_statistics['std_per_generation'].append(np.std(fitness_scores))
        
        # Population structure statistics
        sample_counts = [len(design.heap) for design in population]
        self.population_statistics['sample_count_mean'].append(np.mean(sample_counts))
        self.population_statistics['sample_count_std'].append(np.std(sample_counts))
        self.population_statistics['sample_count_min'].append(np.min(sample_counts))
        self.population_statistics['sample_count_max'].append(np.max(sample_counts))
        
        # Convergence tracking
        if len(self.fitness_statistics['best_per_generation']) > 1:
            current_best = self.fitness_statistics['best_per_generation'][-1]
            previous_best = self.fitness_statistics['best_per_generation'][-2]
            improvement = previous_best - current_best
            self.convergence_statistics['fitness_improvements'].append(improvement)
        else:
            self.convergence_statistics['fitness_improvements'].append(0.0)
        
        # Track stagnation periods
        self.convergence_statistics['stagnation_periods'].append(self.stagnation_counter)
        
        # Track diversity drops
        if len(self.diversity_history) > 1:
            diversity_change = self.diversity_history[-1] - self.diversity_history[-2]
            self.convergence_statistics['diversity_drops'].append(diversity_change)
        else:
            self.convergence_statistics['diversity_drops'].append(0.0)
        
        # Track parameter changes if adaptive
        if self.adaptive_parameters and len(self.parameter_history) > 1:
            current_params = self.parameter_history[-1]
            previous_params = self.parameter_history[-2]
            param_changes = 0
            param_changes += abs(current_params['mutation_rate'] - previous_params['mutation_rate']) > 0.001
            param_changes += abs(current_params['elitism_rate'] - previous_params['elitism_rate']) > 0.001
            param_changes += current_params['population_size'] != previous_params['population_size']
            param_changes += abs(current_params['diversity_threshold'] - previous_params['diversity_threshold']) > 0.0001
            param_changes += current_params['mutation_intensity'] != previous_params['mutation_intensity']
            self.convergence_statistics['parameter_changes'].append(param_changes)
        else:
            self.convergence_statistics['parameter_changes'].append(0)

    def _adjust_population_size(self, population: List[DesignGenetic], fitness_scores: List[float]) -> Tuple[List[DesignGenetic], List[float]]:
        """
        Adjust population size if needed and return updated population and fitness scores
        """
        if len(population) == self.population_size:
            return population, fitness_scores
            
        if len(population) < self.population_size:
            # Need to add individuals
            while len(population) < self.population_size:
                new_design = self.create_initial_population()[0]  # Create one new individual
                population.append(new_design)
        else:
            # Need to remove individuals (keep the best ones)
            combined = list(zip(population, fitness_scores))
            combined.sort(key=lambda x: x[1])  # Sort by fitness
            population = [design for design, _ in combined[:self.population_size]]
        
        # Recalculate fitness scores for the adjusted population
        fitness_scores = [self.evaluate_fitness(design) for design in population]
        return population, fitness_scores

    def _create_next_generation(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        """
        Create the next generation using elitism, crossover, and mutation
        """
        # 1. Elitism: preserve best designs
        new_population = self.select_elites(population, fitness_scores)
        
        # 2. Fill rest with offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate_design(child1)
            child2 = self.mutate_design(child2)
            
            # Validate and add children
            if self.validate_design(child1):
                new_population.append(child1)
            if len(new_population) < self.population_size and self.validate_design(child2):
                new_population.append(child2)
            
            # Safety check to prevent infinite loop
            if len(new_population) >= self.population_size:
                break
        
        # Trim to exact population size
        return new_population[:self.population_size]

    def get_statistics(self) -> dict:
        """
        Get comprehensive algorithm performance statistics
        """
        return {
            # Basic statistics
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'generations_run': len(self.fitness_history),
            'convergence_generation': np.argmin(self.fitness_history) if self.fitness_history else 0,
            'improvement_ratio': (self.fitness_history[0] - self.best_fitness) / self.fitness_history[0] if self.fitness_history and self.fitness_history[0] > 0 else 0,
            
            # Enhanced fitness statistics
            'fitness_statistics': self.fitness_statistics,
            
            # Population structure statistics  
            'population_statistics': self.population_statistics,
            
            # Convergence behavior statistics
            'convergence_statistics': self.convergence_statistics,
            
            # Parameter evolution (if adaptive)
            'parameter_history': self.parameter_history if self.adaptive_parameters else [],
            
            # Summary metrics
            'total_improvements': sum(1 for imp in self.convergence_statistics.get('fitness_improvements', []) if imp > 1e-6),
            'max_stagnation_period': max(self.convergence_statistics.get('stagnation_periods', [0])),
            'average_diversity': np.mean(self.diversity_history) if self.diversity_history else 0,
            'final_sample_count': len(self.best_design.heap) if self.best_design else 0
        }

    def plot_evolution_statistics(self, save_path: str = None):
        """
        Create comprehensive plots showing algorithm evolution over time
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available. Please install it to generate plots: pip install matplotlib")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        generations = range(len(self.fitness_history))
        
        # 1. Fitness Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(generations, self.fitness_statistics['best_per_generation'], 'b-', label='Best', linewidth=2)
        ax1.plot(generations, self.fitness_statistics['mean_per_generation'], 'g--', label='Mean', alpha=0.7)
        ax1.fill_between(generations, 
                        np.array(self.fitness_statistics['mean_per_generation']) - np.array(self.fitness_statistics['std_per_generation']),
                        np.array(self.fitness_statistics['mean_per_generation']) + np.array(self.fitness_statistics['std_per_generation']),
                        alpha=0.3, color='gray', label='±1 STD')
        ax1.set_title('Fitness Evolution Over Time')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (Variance)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Population Diversity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(generations, self.diversity_history, 'r-', linewidth=2)
        ax2.set_title('Population Diversity Over Time')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity Index')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample Count Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(generations, self.population_statistics['sample_count_mean'], 'purple', label='Mean', linewidth=2)
        ax3.fill_between(generations,
                        self.population_statistics['sample_count_min'],
                        self.population_statistics['sample_count_max'],
                        alpha=0.3, color='purple', label='Min-Max Range')
        ax3.set_title('Sample Count per Design')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Number of Samples')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Adaptive Parameters (if available)
        if self.adaptive_parameters and self.parameter_history:
            ax4 = fig.add_subplot(gs[1, 0])
            param_gens = [p['generation'] for p in self.parameter_history]
            ax4.plot(param_gens, [p['mutation_rate'] for p in self.parameter_history], 'b-', label='Mutation Rate')
            ax4.plot(param_gens, [p['elitism_rate'] for p in self.parameter_history], 'g-', label='Elitism Rate')
            ax4.plot(param_gens, [p['diversity_threshold'] for p in self.parameter_history], 'r-', label='Diversity Threshold')
            ax4.set_title('Adaptive Parameters Evolution')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Parameter Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Convergence Behavior
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(generations, self.convergence_statistics['fitness_improvements'], 'orange', alpha=0.7)
        ax5.set_title('Fitness Improvements per Generation')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Improvement')
        ax5.grid(True, alpha=0.3)
        
        # 6. Stagnation Tracking
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(generations, self.convergence_statistics['stagnation_periods'], 'red', alpha=0.7)
        ax6.set_title('Stagnation Periods')
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('Stagnation Counter')
        ax6.grid(True, alpha=0.3)
        
        # 7. Population Size Evolution (if adaptive)
        if self.adaptive_parameters and self.parameter_history:
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(param_gens, [p['population_size'] for p in self.parameter_history], 'navy', linewidth=2)
            ax7.set_title('Population Size Evolution')
            ax7.set_xlabel('Generation')
            ax7.set_ylabel('Population Size')
            ax7.grid(True, alpha=0.3)
        
        # 8. Mutation Intensity Evolution (if adaptive)
        if self.adaptive_parameters and self.parameter_history:
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(param_gens, [p['mutation_intensity'] for p in self.parameter_history], 'darkorange', linewidth=2)
            ax8.set_title('Mutation Intensity Evolution')
            ax8.set_xlabel('Generation')
            ax8.set_ylabel('Mutation Intensity')
            ax8.grid(True, alpha=0.3)
        
        # 9. Parameter Changes Activity
        if self.convergence_statistics['parameter_changes']:
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.bar(generations, self.convergence_statistics['parameter_changes'], alpha=0.7, color='teal')
            ax9.set_title('Parameter Adaptation Activity')
            ax9.set_xlabel('Generation')
            ax9.set_ylabel('Number of Parameter Changes')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Genetic Algorithm Evolution Statistics', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

    def print_best_design_info(self):
        """
        Print detailed information about the best design found
        """
        if self.best_design is None:
            print("No best design available. Run the algorithm first.")
            return
        
        print("\n" + "="*50)
        print("BEST DESIGN INFORMATION")
        print("="*50)
        
        # Calculate inclusion probabilities
        id_probs = {}
        for sample in self.best_design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability
        
        print("Inclusion Probabilities:")
        print("Unit\tExpected\tActual\t\tDifference")
        print("-" * 50)
        for i, expected_prob in enumerate(self.inclusions):
            actual_prob = id_probs.get(i, 0)
            diff = abs(actual_prob - expected_prob)
            print(f"{i}\t{expected_prob:.6f}\t{actual_prob:.6f}\t{diff:.8f}")
        
        print(f"\nNumber of samples in design: {len(self.best_design.heap)}")
        print(f"Best fitness (variance): {self.best_fitness:.8f}")
        
        print("\nSample composition:")
        for i, sample in enumerate(self.best_design.heap):
            print(f"Sample {i}: Prob={sample.probability:.6f}, Units={sorted(sample.ids)}")
