import argparse
from experiments import run_all_experiments
from simulation import SimulationManager
from visualization import plot_dynamics, plot_network


def main():
    parser = argparse.ArgumentParser(description='Моделирование динамики общественного мнения в социальных сетях')
    parser.add_argument('--mode', type=str, default='experiments',
                        choices=['single', 'experiments'],
                        help='Режим работы: single - одна симуляция, experiments - все эксперименты')
    parser.add_argument('--users', type=int, default=100, help='Количество пользователей')
    parser.add_argument('--network', type=str, default='small_world',
                        choices=['random', 'small_world', 'complete', 'scale_free'],
                        help='Тип сети')
    parser.add_argument('--infection', type=float, default=0.3, help='Вероятность заражения')
    parser.add_argument('--skeptic', type=float, default=0.2, help='Вероятность стать скептиком')
    parser.add_argument('--recovery', type=float, default=0.1, help='Вероятность восстановления')
    parser.add_argument('--initial', type=int, default=1, help='Начальные распространители')
    parser.add_argument('--steps', type=int, default=100, help='Максимальное количество шагов')
    parser.add_argument('--seed', type=int, default=None, help='Seed для воспроизводимости')
    parser.add_argument('--save', action='store_true', help='Сохранить результаты')

    args = parser.parse_args()

    if args.mode == 'single':
        print("Запуск одиночной симуляции...")

        manager = SimulationManager()
        result = manager.run_simulation(
            n_users=args.users,
            network_type=args.network,
            infection_prob=args.infection,
            skeptic_prob=args.skeptic,
            recovery_prob=args.recovery,
            initial_spreaders=args.initial,
            max_steps=args.steps,
            seed=args.seed
        )

        plot_dynamics(result['history'],
                      title=f"Симуляция: {args.network} сеть, {args.users} пользователей")
        plot_network(result['users'], result['edges'],
                     title=f"Структура сети: {args.network}")

        if args.save:
            manager.save_results('single_simulation_results.pkl')

    elif args.mode == 'experiments':
        run_all_experiments()


if __name__ == "__main__":
    main()