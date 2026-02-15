import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from enum import Enum


class UserState(Enum):
    UNAWARE = 0
    SPREADER = 1
    SKEPTIC = 2


class User:
    def __init__(self, user_id: int, state: UserState = UserState.UNAWARE):
        self.id = user_id
        self.state = state
        self.neighbors = []

    def add_neighbor(self, neighbor: 'User'):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def get_neighbor_states(self) -> Dict[UserState, int]:
        states_count = {state: 0 for state in UserState}
        for neighbor in self.neighbors:
            states_count[neighbor.state] += 1
        return states_count

    def __repr__(self):
        return f"User({self.id}, {self.state.name})"


class SocialNetwork:
    @staticmethod
    def generate_network(n_users: int,
                         network_type: str = 'small_world',
                         directed: bool = False,
                         **kwargs) -> Tuple[List[User], List[Tuple[int, int]]]:
        users = [User(i) for i in range(n_users)]
        edges = []

        if network_type == 'complete':
            for i in range(n_users):
                for j in range(i + 1, n_users):
                    users[i].add_neighbor(users[j])
                    if not directed:
                        users[j].add_neighbor(users[i])
                    edges.append((i, j))

        elif network_type == 'random':
            p = kwargs.get('p', 0.1)
            for i in range(n_users):
                for j in range(i + 1, n_users):
                    if np.random.random() < p:
                        users[i].add_neighbor(users[j])
                        if not directed:
                            users[j].add_neighbor(users[i])
                        edges.append((i, j))

        elif network_type == 'small_world':
            # Граф малого мира (Уоттса-Строгаца)
            k = kwargs.get('k', 4)
            beta = kwargs.get('beta', 0.1)

            for i in range(n_users):
                for j in range(1, k // 2 + 1):
                    neighbor = (i + j) % n_users
                    users[i].add_neighbor(users[neighbor])
                    if not directed:
                        users[neighbor].add_neighbor(users[i])
                    edges.append((i, neighbor))

            for i in range(n_users):
                for j in range(1, k // 2 + 1):
                    if np.random.random() < beta:
                        neighbor = (i + j) % n_users
                        if users[neighbor] in users[i].neighbors:
                            users[i].neighbors.remove(users[neighbor])
                            if not directed:
                                users[neighbor].neighbors.remove(users[i])
                            edges.remove((i, neighbor))

                        new_neighbor = np.random.choice([u for u in users if u.id != i and u not in users[i].neighbors])
                        users[i].add_neighbor(new_neighbor)
                        if not directed:
                            new_neighbor.add_neighbor(users[i])
                        edges.append((i, new_neighbor.id))
        elif network_type == 'scale_free':
            # Безмасштабная сеть (Барабаши-Альберт)
            m = kwargs.get('m', 2)

            for i in range(min(m, n_users)):
                for j in range(i + 1, min(m, n_users)):
                    users[i].add_neighbor(users[j])
                    if not directed:
                        users[j].add_neighbor(users[i])
                    edges.append((i, j))

            for i in range(m, n_users):
                degrees = [len(u.neighbors) for u in users[:i]]
                total_degree = sum(degrees)

                if total_degree > 0:
                    probs = [deg / total_degree for deg in degrees]
                    targets = np.random.choice(range(i), size=min(m, i), replace=False, p=probs)

                    for target in targets:
                        users[i].add_neighbor(users[target])
                        if not directed:
                            users[target].add_neighbor(users[i])
                        edges.append((i, target))

        return users, edges

    @staticmethod
    def get_network_statistics(users: List[User]) -> Dict:
        degrees = [len(user.neighbors) for user in users]

        return {
            'n_users': len(users),
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'density': sum(degrees) / (len(users) * (len(users) - 1)) if len(users) > 1 else 0
        }


class MarkovOpinionDynamics:
    def __init__(self,
                 users: List[User],
                 infection_prob: float = 0.3,
                 skeptic_prob: float = 0.2,
                 recovery_prob: float = 0.1,
                 initial_spreaders: int = 1):
        self.users = users
        self.infection_prob = infection_prob
        self.skeptic_prob = skeptic_prob
        self.recovery_prob = recovery_prob

        self._initialize_states(initial_spreaders)

        self.history_users = []
        self.history = {
            'unaware': [],
            'spreader': [],
            'skeptic': []
        }
        self._record_states()

    def _initialize_states(self, initial_spreaders: int):
        for user in self.users:
            user.state = UserState.UNAWARE

        spreader_ids = np.random.choice(len(self.users), initial_spreaders, replace=False)
        for idx in spreader_ids:
            self.users[idx].state = UserState.SPREADER

    def _record_states(self):
        counts = {state: 0 for state in UserState}
        for user in self.users:
            counts[user.state] += 1

        cur_users = [User(i) for i in range(len(self.users))]
        for i in range(len(cur_users)):
            cur_users[i].state = self.users[i].state
            cur_users[i].neighbors = self.users[i].neighbors.copy()
            cur_users[i].id = self.users[i].id
        self.history_users.append(cur_users.copy())

        self.history['unaware'].append(counts[UserState.UNAWARE])
        self.history['spreader'].append(counts[UserState.SPREADER])
        self.history['skeptic'].append(counts[UserState.SKEPTIC])

    def _calculate_transition_prob(self, user: User) -> Dict[UserState, float]:
        neighbor_states = user.get_neighbor_states()
        n_neighbors = len(user.neighbors)

        if n_neighbors == 0:
            if user.state == UserState.UNAWARE:
                return {UserState.UNAWARE: 1.0, UserState.SPREADER: 0.0, UserState.SKEPTIC: 0.0}
            elif user.state == UserState.SPREADER:
                return {
                    UserState.UNAWARE: 0.0,
                    UserState.SPREADER: 1 - self.recovery_prob,
                    UserState.SKEPTIC: self.recovery_prob
                }
            else:  # SKEPTIC
                return {
                    UserState.UNAWARE: 0.0,
                    UserState.SPREADER: 0.0,
                    UserState.SKEPTIC: 1.0
                }

        if user.state == UserState.UNAWARE:
            spreader_fraction = neighbor_states[UserState.SPREADER] / n_neighbors
            infection_prob = min(1.0, self.infection_prob * spreader_fraction)

            skeptic_fraction = neighbor_states[UserState.SKEPTIC] / n_neighbors
            become_skeptic_prob = min(1.0, self.skeptic_prob * skeptic_fraction)

            stay_unaware_prob = max(0.0, 1 - infection_prob - become_skeptic_prob)

            return {
                UserState.UNAWARE: stay_unaware_prob,
                UserState.SPREADER: infection_prob,
                UserState.SKEPTIC: become_skeptic_prob
            }

        elif user.state == UserState.SPREADER:
            return {
                UserState.UNAWARE: 0.0,
                UserState.SPREADER: 1 - self.recovery_prob,
                UserState.SKEPTIC: self.recovery_prob
            }

        else:  # SKEPTIC
            return {
                UserState.UNAWARE: 0.0,
                UserState.SPREADER: 0.0,
                UserState.SKEPTIC: 1.0
            }

    def step(self) -> bool:
        new_states = []

        for user in self.users:
            if user.state == UserState.SKEPTIC:
                new_states.append(UserState.SKEPTIC)
            else:
                probs = self._calculate_transition_prob(user)
                states = list(probs.keys())
                probabilities = list(probs.values())
                new_state = np.random.choice(states, p=probabilities)
                new_states.append(new_state)

        for user, new_state in zip(self.users, new_states):
            user.state = new_state

        self._record_states()

        spreader_count = self.history['spreader'][-1]
        return spreader_count > 0

    def simulate(self, max_steps: int = 100) -> int:
        for step in range(max_steps):
            active = self.step()
            if not active:
                return step + 1
        return max_steps

    def get_statistics(self) -> Dict:
        final_unaware = self.history['unaware'][-1]
        final_spreader = self.history['spreader'][-1]
        final_skeptic = self.history['skeptic'][-1]
        total_users = len(self.users)

        return {
            'total_users': total_users,
            'final_unaware': final_unaware,
            'final_spreader': final_spreader,
            'final_skeptic': final_skeptic,
            'final_informed': final_spreader + final_skeptic,
            'infection_rate': (final_spreader + final_skeptic) / total_users if total_users > 0 else 0,
            'max_spreaders': max(self.history['spreader']) if self.history['spreader'] else 0,
            'steps': len(self.history['unaware'])
        }