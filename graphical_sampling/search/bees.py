from dataclasses import dataclass
from typing import Any
import numpy as np
from joblib import Parallel, delayed
from ..criteria.criteria import Criteria
from ..new_design import NewDesign


@dataclass
class FoodSource:
    design: NewDesign
    criteria_value: float
    trial_counter: int = 0

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, FoodSource):
            return NotImplemented
        return self.criteria_value < other.criteria_value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FoodSource):
            return NotImplemented
        return self.criteria_value == other.criteria_value


class Bees:
    def __init__(
        self,
        initial_designs: list[NewDesign],
        criteria: Criteria,
        *,
        colony_size: int = 20,
        limit: int = 50,
        threshold: float = -1.0,
    ) -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria
        self.colony_size = colony_size
        self.limit = limit
        self.threshold = threshold
        self.rng = np.random.default_rng()

        print('Evaluating initial designs...')
        self.initial_criteria_value = np.array([
            self.criteria(design) for design in self.initial_designs
        ])

        best_idx = np.argmin(self.initial_criteria_value)
        self.best_design = self.initial_designs[best_idx]
        self.best_criteria_value = self.initial_criteria_value[best_idx]

        print(f'ABC initialized - Best initial criteria value: {self.best_criteria_value}')

    def iterate_design(
        self,
        design: NewDesign,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1
    ) -> NewDesign:
        new_design = design.copy()
        new_design.iterate(
            n_clusters_to_change_order_zone,
            n_clusters_to_change_order_units,
            n_zones_to_change_order_units,
            n_changes_in_order_of_units,
            n_changes_in_order_of_zones,
        )
        return new_design

    def calculate_fitness(self, criteria_value: float) -> float:
        if criteria_value >= 0:
            return 1.0 / (1.0 + criteria_value)
        else:
            return 1.0 + abs(criteria_value)

    def calculate_selection_probabilities(self, food_sources: list[FoodSource]) -> np.ndarray:
        fitness_values = np.array([
            self.calculate_fitness(fs.criteria_value) for fs in food_sources
        ])
        total_fitness = np.sum(fitness_values)

        if total_fitness == 0:
            return np.ones(len(food_sources)) / len(food_sources)

        return fitness_values / total_fitness

    def _generate_new_food_source(
        self,
        old_fs: FoodSource,
        new_design: NewDesign
    ) -> FoodSource:
        new_criteria_value = self.criteria(new_design)
        if new_criteria_value < old_fs.criteria_value:
            return FoodSource(
                design=new_design,
                criteria_value=new_criteria_value,
                trial_counter=0,
            )
        else:
            return FoodSource(
                design=old_fs.design,
                criteria_value=old_fs.criteria_value,
                trial_counter=old_fs.trial_counter + 1,
            )

    def employed_bee_phase(
        self,
        food_sources: list[FoodSource],
        n_clusters_to_change_order_zone: int | str,
        n_clusters_to_change_order_units: int | str,
        n_zones_to_change_order_units: int | str,
        n_changes_in_order_of_units: int,
        n_changes_in_order_of_zones: int,
        n_jobs: int,
    ) -> list[FoodSource]:
        new_designs = Parallel(n_jobs=n_jobs)(
            delayed(self.iterate_design)(
                fs.design,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            ) for fs in food_sources
        )

        new_food_sources = []
        for i, (food_source, new_design) in enumerate(zip(food_sources, new_designs)):
            new_food_sources.append(self._generate_new_food_source(food_source, new_design))
        return new_food_sources

    def onlooker_bee_phase(
        self,
        food_sources: list[FoodSource],
        n_clusters_to_change_order_zone: int | str,
        n_clusters_to_change_order_units: int | str,
        n_zones_to_change_order_units: int | str,
        n_changes_in_order_of_units: int,
        n_changes_in_order_of_zones: int,
        n_jobs: int,
    ) -> list[FoodSource]:
        probabilities = self.calculate_selection_probabilities(food_sources)

        selected_indices = self.rng.choice(
            len(food_sources), size=self.colony_size, replace=True, p=probabilities
        ).tolist()

        new_designs = Parallel(n_jobs=n_jobs)(
            delayed(self.iterate_design)(
                food_sources[i].design,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            ) for i in selected_indices
        )

        new_food_sources = food_sources.copy()
        for idx, new_design in zip(selected_indices, new_designs):
            new_food_sources[idx] = self._generate_new_food_source(new_food_sources[idx], new_design)
        return new_food_sources

    def scout_bee_phase(
        self,
        food_sources: list[FoodSource],
        n_clusters_to_change_order_zone: int | str,
        n_clusters_to_change_order_units: int | str,
        n_zones_to_change_order_units: int | str,
        n_changes_in_order_of_units: int,
        n_changes_in_order_of_zones: int,
    ) -> list[FoodSource]:
        new_food_sources = []

        for food_source in food_sources:
            if food_source.trial_counter >= self.limit:
                # Scout: generate a new design
                if hasattr(NewDesign, 'random'):
                    new_design = NewDesign.random()
                else:
                    random_initial = self.rng.choice(self.initial_designs)
                    new_design = self.iterate_design(
                        random_initial,
                        n_clusters_to_change_order_zone,
                        n_clusters_to_change_order_units,
                        n_zones_to_change_order_units,
                        n_changes_in_order_of_units * 10,
                        n_changes_in_order_of_zones * 10,
                    )
                new_criteria_value = self.criteria(new_design)
                new_food_sources.append(FoodSource(
                    design=new_design,
                    criteria_value=new_criteria_value,
                    trial_counter=0,
                ))
                print(f'  Scout: Abandoned and found new source with criteria {new_criteria_value:.6f}')
            else:
                new_food_sources.append(food_source)

        return new_food_sources

    def run(
        self,
        max_iterations: int,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1,
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> int:
        food_sources = []
        for i in range(min(self.colony_size, len(self.initial_designs))):
            food_sources.append(FoodSource(
                design=self.initial_designs[i],
                criteria_value=self.initial_criteria_value[i],
                trial_counter=0
            ))

        while len(food_sources) < self.colony_size:
            random_design = self.rng.choice(self.initial_designs)
            new_design = self.iterate_design(
                random_design,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            )
            criteria_value = self.criteria(new_design)
            food_sources.append(FoodSource(
                design=new_design,
                criteria_value=criteria_value,
                trial_counter=0
            ))

        if verbose:
            print(f'\nStarting ABC with {len(food_sources)} food sources')
            print(f'Colony size: {self.colony_size}, Limit: {self.limit}')

        iteration_best_found = 0

        for it in range(max_iterations):
            if verbose:
                print(f'\nIteration {it + 1}/{max_iterations}')

            # Phase 1: Employed Bees
            food_sources = self.employed_bee_phase(
                food_sources,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
                n_jobs,
            )

            # Phase 2: Onlooker Bees
            food_sources = self.onlooker_bee_phase(
                food_sources,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
                n_jobs,
            )

            # Phase 3: Scout Bees
            food_sources = self.scout_bee_phase(
                food_sources,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            )

            current_best = min(food_sources, key=lambda fs: fs.criteria_value)
            if current_best.criteria_value < self.best_criteria_value:
                self.best_design = current_best.design
                self.best_criteria_value = current_best.criteria_value
                iteration_best_found = it

                if verbose:
                    print('='*60)
                    print(f'NEW BEST at iteration {it + 1}')
                    print(f'Criteria value: {self.best_criteria_value:.8f}')
                    print('='*60)

                # Check stopping criterion
                if self.best_criteria_value < self.threshold:
                    if verbose:
                        print(f'\nThreshold reached! Stopping at iteration {it + 1}')
                    return iteration_best_found

            if verbose:
                avg_criteria = np.mean([fs.criteria_value for fs in food_sources])
                print(f'  Best: {self.best_criteria_value:.6f}, Avg: {avg_criteria:.6f}')

        if verbose:
            print(f'\n{"="*60}')
            print(f'ABC completed!')
            print(f'Best criteria: {self.best_criteria_value:.8f}')
            print(f'Found at iteration: {iteration_best_found + 1}')
            print(f'{"="*60}')

        return iteration_best_found
