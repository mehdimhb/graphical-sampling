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
        adaptive_parameters: bool = False,     # Simplified: default to False
        selection_pressure: float = 1.5       # Rank-based selection pressure (1.1-2.0)
    ):
        # Core algorithm parameters
        self.random_pull = random_pull
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_intensity = mutation_intensity
        self.selection_pressure = selection_pressure
        self.use_partitions = use_partitions
        self.adaptive_parameters = adaptive_parameters

        #diversity threshold
        self.diversity_threshold = 0.1  # Threshold for diversity to trigger adaptive
        self.reseed_fraction = 0.3   # Reseed RNG every 10 generations for diversity

        # Simple adaptive parameters (only if enabled)
        if adaptive_parameters:
            self.initial_mutation_rate = 0.2  # Start with moderate mutation
            self.mutation_rate = self.initial_mutation_rate
            self.stagnation_counter = 0
            self.last_best_fitness = float('inf')
        else:
            self.mutation_rate = 2.0  # Always mutate when not adaptive
            self.stagnation_counter = 0  # Initialize for statistics even when not adaptive
            self.last_best_fitness = float('inf')

        # Performance tracking
        self.generation = 0
        self.diversity_history = []
        self.fitness_history = []
        self.parameter_history = []

        # Basic statistics tracking
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

    def chaotic_initial_population(self, M=100, alpha=0.5):
        """
        Generate high-entropy designs by applying multiple segment interchanges (Chaotic GFS).
        """
        population = []
        for _ in range(self.population_size):
            design = DesignGenetic(self.inclusions, rng=self.rng)
            for _ in range(M):
                design.iterate(
                    random_pull=True,
                    switch_coefficient=self.rng.uniform(alpha * 0.5, alpha * 1.5),
                    partitions=self.partitions,
                    border_units=self.border_units
                )
            design.merge_identical()
            population.append(design)
        return population

    def evaluate_fitness(self, design: DesignGenetic) -> float:
        try:
            # Calculate VarNHT criterion - lower is better
            variance = self.criterion(design)  # Use __call__ method
            return variance
        except Exception as e:
            # If calculation fails, return very high penalty
            print(f"Fitness calculation failed: {e}")
            return 1e10

    def rank_based_selection(self, population: List[DesignGenetic], fitness_scores: List[float],
                            parent_counts: dict, max_children: int = 6, selection_pressure: float = 1.5) -> int:

        available_indices = [i for i in range(len(population)) if parent_counts.get(i, 0) < max_children]

        if not available_indices:
            parent_counts.clear()
            available_indices = list(range(len(population)))

        # Get fitness scores and rank them (lower fitness = better rank)
        available_fitness = [fitness_scores[i] for i in available_indices]
        sorted_indices = sorted(range(len(available_fitness)), key=lambda i: available_fitness[i])
        
        # Assign ranks (0 = best, n-1 = worst)
        ranks = [0] * len(available_fitness)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # Calculate selection probabilities using linear ranking
        n = len(available_fitness)
        weights = []
        for rank in ranks:
            # Linear ranking formula: P(i) = (2 - SP + 2*(SP-1)*(N-i-1)/(N-1)) / N
            # where SP = selection_pressure, N = population_size, i = rank
            prob = (2 - selection_pressure + 2 * (selection_pressure - 1) * (n - rank - 1) / (n - 1)) / n
            weights.append(max(0.01, prob))  # Ensure minimum probability

        # Create cumulative distribution
        cumulative_weights = np.cumsum(weights)
        total_weight = cumulative_weights[-1]

        # Select
        random_value = self.rng.random() * total_weight
        selected_position = np.searchsorted(cumulative_weights, random_value)
        selected_position = min(selected_position, len(available_indices) - 1)

        return available_indices[selected_position]

    def select_parent(self, population: List[DesignGenetic], fitness_scores: List[float],
                     parent_counts: dict, max_children: int = 6) -> int:
        return self.rank_based_selection(population, fitness_scores, parent_counts, max_children, self.selection_pressure)

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        # Adaptive mutation: apply mutation based on current mutation rate
        if self.rng.random() > self.mutation_rate:
            return design.copy()

        mutated = design.copy()
        if self.rng.random() < 0.5: # 20% chance: apply chaotic mutation
            for _ in range(self.mutation_intensity *7):
                design.iterate(
                    random_pull=True,
                    switch_coefficient=self.rng.uniform(0.2, 0.8),
                    partitions=None,  # intentionally ignore partitions
                    border_units=self.border_units
                )
        mutated.merge_identical()
        # else:
        # Apply several segment interchanges to increase entropy
        for _ in range(self.mutation_intensity*7):
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
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        return [population[i].copy() for i in elite_indices]

    def validate_design(self, design: DesignGenetic) -> bool:
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

    def adapt_parameters(self, current_fitness: float, generation: int):
        if not self.adaptive_parameters:
            return

        # Check for improvement
        improvement = self.last_best_fitness - current_fitness
        has_improvement = improvement > 1e-6

        if has_improvement:
            # Good progress: slightly reduce mutation rate for fine-tuning
            self.mutation_rate *= 0.95
            self.stagnation_counter = 0
            self.last_best_fitness = current_fitness
        else:
            # No improvement: increase stagnation counter
            self.stagnation_counter += 1

            # If stagnated for several generations, increase mutation
            if self.stagnation_counter > 5:
                self.mutation_rate *= 1.05

            # Reset if stagnated too long
            if self.stagnation_counter > 20:
                self.mutation_rate = self.initial_mutation_rate
                self.stagnation_counter = 0

        # Keep mutation rate in reasonable bounds
        self.mutation_rate = max(0.05, min(0.8, self.mutation_rate))

        # Record parameter history for analysis (simplified)
        if generation % 5 == 0:  # Record every 5 generations to reduce overhead
            self.parameter_history.append({
                'generation': generation,
                'mutation_rate': self.mutation_rate,
                'elitism_rate': self.elitism_rate,
                'population_size': self.population_size,
                'mutation_intensity': self.mutation_intensity,
                'stagnation': self.stagnation_counter,
                'improvement': improvement
            })

    def run(self, max_generations: int = 100, verbose: bool = True) -> DesignGenetic:
        if verbose:
            print("Initializing Geometric Sampling Genetic Algorithm...")
            print(f"Population size: {self.population_size}")
            print(f"Inclusion probabilities: {self.inclusions}")
            print(f"Auxiliary variable: {self.auxiliary_var}")
            print(f"Using partitions: {self.use_partitions}")
            print("="*50)

        # Initialize population
        population = self.create_initial_population()[:self.population_size // 2]+\
                     self.chaotic_initial_population(M=100, alpha=0.7)[:self.population_size // 2]

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

            # if population_diversity < self.diversity_threshold:
            #     print(f"⚠️ Diversity dropped to {population_diversity:.3f}. Re-seeding population...")
            #     num_reseed = int(self.reseed_fraction * self.population_size)
            #     # Generate new designs using chaotic initialization
            #     reseeded_population = self.chaotic_initial_population(M=100, alpha=0.7)[:num_reseed]
            #     # Replace worst-performing designs with new diverse ones
            #     worst_indices = np.argsort(fitness_scores)[-num_reseed:]
            #     for idx, new_design in zip(worst_indices, reseeded_population):
            #         population[idx] = new_design

            # Collect detailed statistics
            self._collect_generation_statistics(population, fitness_scores, population_diversity)

            self.adapt_parameters(current_best_fitness, generation)

            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.8f}, "
                      f"Diversity = {population_diversity:.3f}, "
                      f"MutRate = {self.mutation_rate:.3f}, "
                      f"Elite = {self.elitism_rate:.3f}")

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
            # Only track the parameters we actually adapt in the simplified system
            self.convergence_statistics['parameter_changes'].append(param_changes)
        else:
            self.convergence_statistics['parameter_changes'].append(0)

    def _create_next_generation(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        """
        Create the next generation using elitism, crossover, and mutation
        """
        # 1. Elitism: preserve best designs
        new_population = self.select_elites(population, fitness_scores)

        # 2. Initialize selection tracking
        parent_counts = {}  # Dictionary tracking how many children each parent has produced
        max_children_per_parent = 3  # Each individual can be parent at most 3 times

        # 3. Fill rest with offspring
        while len(new_population) < self.population_size:
            # Smart selection using configured method (optimized for VarNHT minimization)
            parent1_idx = self.select_parent(population, fitness_scores, parent_counts, max_children_per_parent)
            parent2_idx = self.select_parent(population, fitness_scores, parent_counts, max_children_per_parent)

            # Make sure parents are different
            attempts = 0
            while parent1_idx == parent2_idx and attempts < 10:
                parent2_idx = self.select_parent(population, fitness_scores, parent_counts, max_children_per_parent)
                attempts += 1

            # Get parent designs
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate_design(child1)
            child2 = self.mutate_design(child2)

            # Validate and add children, then update parent counts
            children_added = 0
            if self.validate_design(child1):
                new_population.append(child1)
                children_added += 1
            if len(new_population) < self.population_size and self.validate_design(child2):
                new_population.append(child2)
                children_added += 1
            
            # Update parent counts only if children were actually added
            if children_added > 0:
                parent_counts[parent1_idx] = parent_counts.get(parent1_idx, 0) + children_added
                parent_counts[parent2_idx] = parent_counts.get(parent2_idx, 0) + children_added
            
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
            ax4.plot(param_gens, [p['mutation_rate'] for p in self.parameter_history], 'b-', label='Mutation Rate', linewidth=2)
            ax4.set_title('Simple Adaptive Mutation Rate')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Mutation Rate')
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
        
        # 7. Stagnation Counter Evolution (if adaptive)
        if self.adaptive_parameters and self.parameter_history:
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(param_gens, [p['stagnation'] for p in self.parameter_history], 'red', linewidth=2)
            ax7.set_title('Stagnation Counter Over Time')
            ax7.set_xlabel('Generation')
            ax7.set_ylabel('Stagnation Counter')
            ax7.grid(True, alpha=0.3)
        
        # 8. Improvement Tracking (if adaptive)
        if self.adaptive_parameters and self.parameter_history:
            ax8 = fig.add_subplot(gs[2, 1])
            improvements = [p['improvement'] for p in self.parameter_history]
            ax8.plot(param_gens, improvements, 'green', alpha=0.7)
            ax8.set_title('Fitness Improvements Over Time')
            ax8.set_xlabel('Generation')
            ax8.set_ylabel('Improvement Value')
            ax8.grid(True, alpha=0.3)
        
        # 9. Parameter Changes Activity
        if self.convergence_statistics['parameter_changes']:
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.bar(generations, self.convergence_statistics['parameter_changes'], alpha=0.7, color='teal')
            ax9.set_title('Mutation Rate Changes')
            ax9.set_xlabel('Generation')
            ax9.set_ylabel('Parameter Changes')
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
