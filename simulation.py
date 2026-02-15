import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from models import SocialNetwork, MarkovOpinionDynamics, User


class SimulationManager:
    def __init__(self):
        self.results = []

    def run_simulation(self,
                      n_users: int = 100,
                      network_type: str = 'small_world',
                      infection_prob: float = 0.3,
                      skeptic_prob: float = 0.2,
                      recovery_prob: float = 0.1,
                      initial_spreaders: int = 1,
                      max_steps: int = 100,
                      seed: Optional[int] = None) -> Dict:
        if seed is not None:
            np.random.seed(seed)

        users, edges = SocialNetwork.generate_network(
            n_users=n_users,
            network_type=network_type
        )
        network_stats = SocialNetwork.get_network_statistics(users)
        model = MarkovOpinionDynamics(
            users=users,
            infection_prob=infection_prob,
            skeptic_prob=skeptic_prob,
            recovery_prob=recovery_prob,
            initial_spreaders=initial_spreaders
        )
        steps = model.simulate(max_steps=max_steps)
        stats = model.get_statistics()
        result = {
            'parameters': {
                'n_users': n_users,
                'network_type': network_type,
                'infection_prob': infection_prob,
                'skeptic_prob': skeptic_prob,
                'recovery_prob': recovery_prob,
                'initial_spreaders': initial_spreaders
            },
            'network_stats': network_stats,
            'model_stats': stats,
            'history': model.history.copy(),
            'history_users': model.history_users.copy(),
            'users': users,
            'edges': edges,
            'model': model,
        }

        self.results.append(result)
        return result

    def run_parameter_sweep(self,
                           param_name: str,
                           param_values: List,
                           n_simulations: int = 5,
                           n_users: int = 100,
                           network_type: str = 'small_world',
                           **kwargs) -> List[Dict]:
        sweep_results = []

        for param_value in tqdm(param_values, desc="Параметр"):
            param_results = []

            for sim_idx in range(n_simulations):
                params = {
                    'n_users': n_users,
                    'network_type': network_type,
                    **kwargs
                }
                params[param_name] = param_value

                result = self.run_simulation(**params)
                param_results.append(result)

            avg_infection_rate = np.mean([r['model_stats']['infection_rate']
                                         for r in param_results])
            avg_max_spreaders = np.mean([r['model_stats']['max_spreaders']
                                        for r in param_results])
            avg_steps = np.mean([r['model_stats']['steps']
                                for r in param_results])

            sweep_result = {
                'param_name': param_name,
                'param_value': param_value,
                'results': param_results,
                'avg_infection_rate': avg_infection_rate,
                'avg_max_spreaders': avg_max_spreaders,
                'avg_steps': avg_steps,
                'history': param_results[0]['history']
            }

            sweep_results.append(sweep_result)

        return sweep_results

    def compare_network_types(self,
                             network_types: List[str],
                             n_simulations: int = 5,
                             n_users: int = 100,
                             **kwargs) -> Dict[str, List]:
        comparison_results = {}

        for network_type in tqdm(network_types, desc="Тип сети"):
            results = []

            for sim_idx in range(n_simulations):
                result = self.run_simulation(
                    n_users=n_users,
                    network_type=network_type,
                    **kwargs
                )
                results.append(result)

            infection_rates = [r['model_stats']['infection_rate'] for r in results]
            max_spreaders_list = [r['model_stats']['max_spreaders'] for r in results]
            steps_list = [r['model_stats']['steps'] for r in results]

            comparison_results[network_type] = {
                'results': results,
                'avg_infection_rate': np.mean(infection_rates),
                'std_infection_rate': np.std(infection_rates),
                'avg_max_spreaders': np.mean(max_spreaders_list),
                'std_max_spreaders': np.std(max_spreaders_list),
                'avg_steps': np.mean(steps_list),
                'std_steps': np.std(steps_list)
            }

        return comparison_results

    def save_results(self, filename: str = 'simulation_results.npy'):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)

    def load_results(self, filename: str = 'simulation_results.npy'):
        import pickle
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)