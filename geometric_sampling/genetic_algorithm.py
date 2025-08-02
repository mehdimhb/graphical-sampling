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
    def __init__(
        self, 
        inclusions: np.ndarray, 
        auxiliary_var: np.ndarray,
        population_size: int = 30,
        elitism_rate: float = 0.1,
        mutation_intensity: int = 5,
        use_partitions: bool = True
    ):
        """
        Initialize Geometric Sampling Genetic Algorithm
        
        Args:
            inclusions: First-order inclusion probabilities for each unit
            auxiliary_var: Auxiliary variable for variance calculation (for VarNHT criterion)
            population_size: Number of designs in population
            elitism_rate: Fraction of best designs to preserve each generation
            mutation_intensity: Number of segment interchanges per mutation
            use_partitions: Whether to use partition constraints during mutation
        """
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_intensity = mutation_intensity
        self.use_partitions = use_partitions
        
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
        
        # Track algorithm progress
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
                        random_pull=True,
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
        Apply segment interchange operations (Algorithm 4 from paper)
        Each iterate() call performs one segment swap between two samples
        """
        mutated = design.copy()
        
        # Apply several segment interchanges to increase entropy
        for _ in range(self.mutation_intensity):
            if len(mutated.heap) >= 2:
                mutated.iterate(
                    random_pull=True,
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
            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2], random_pull=True)
            
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
            if abs(actual_prob - expected_prob) > 1e-6:
                return False
        
        return True

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
            # Evaluate fitness for all designs
            fitness_scores = [self.evaluate_fitness(design) for design in population]
            
            # Track best design
            best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_design = population[best_idx].copy()
            
            self.fitness_history.append(current_best_fitness)
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.8f}")
            
            # Create next generation
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
            population = new_population[:self.population_size]
            if generation == 29:
                for design in population:
                    design.merge_identical()

                for design in population:
                    print("*"*50)
                    for sample in design.heap:
                        if len(sample.ids) !=4:
                            print(f"Sample {sample.index} has {len(sample.ids)} IDs, expected 4")
                for design in population:
                    id_probs_tmp = {}

                    print("&"*100)
                    for sample in design.heap:
                        for unit in sample.ids:
                            id_probs_tmp[unit] = id_probs_tmp.get(unit, 0) + sample.probability
                    for unit, prob in sorted(id_probs_tmp.items()):
                        print(f"ID {unit}: {prob}")
                # for design in population:
                #     design.show()
        
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

    def get_statistics(self) -> dict:
        """
        Get algorithm performance statistics
        """
        return {
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'generations_run': len(self.fitness_history),
            'convergence_generation': np.argmin(self.fitness_history),
            'improvement_ratio': (self.fitness_history[0] - self.best_fitness) / self.fitness_history[0] if self.fitness_history[0] > 0 else 0
        }

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
