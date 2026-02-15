import numpy as np
from simulation import SimulationManager
from visualization import (
    plot_dynamics,
    plot_network,
    plot_multiple_simulations,
    create_animation
)
import matplotlib.pyplot as plt


def experiment_basic_simulation():
    print("=== Базовый эксперимент ===")
    manager = SimulationManager()
    result = manager.run_simulation(
        n_users=200,
        network_type='small_world',
        infection_prob=0.4,
        skeptic_prob=0.1,
        recovery_prob=0.05,
        initial_spreaders=3,
        max_steps=100,
        seed=42
    )

    print(f"\nСтатистика сети:")
    for key, value in result['network_stats'].items():
        print(f"  {key}: {value}")

    print(f"\nСтатистика модели:")
    for key, value in result['model_stats'].items():
        print(f"  {key}: {value}")

    plot_dynamics(
        result['history'],
        title="Базовая симуляция: Динамика общественного мнения"
    )

    plot_network(
        result['users'],
        result['edges'],
        title="Структура социальной сети"
    )

    create_animation(result['history_users'], result['edges'], 200, r'C:\Users\Andrey\PycharmProjects\social_network_opinion_dynamics\animation.gif')

    return result


def experiment_threshold_effect():
    print("\n=== Эксперимент: Пороговый эффект ===")

    manager = SimulationManager()
    infection_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    sweep_results = manager.run_parameter_sweep(
        param_name='infection_prob',
        param_values=infection_probs,
        n_simulations=3,
        n_users=150,
        network_type='scale_free',
        skeptic_prob=0.1,
        recovery_prob=0.05,
        initial_spreaders=2,
        max_steps=100
    )

    plot_multiple_simulations(
        sweep_results,
        param_name='infection_prob',
        title="Пороговый эффект: влияние вероятности заражения"
    )

    infection_rates = [r['avg_infection_rate'] for r in sweep_results]

    plt.figure(figsize=(10, 6))
    plt.plot(infection_probs, infection_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Вероятность заражения', fontsize=14)
    plt.ylabel('Доля информированных пользователей', fontsize=14)
    plt.title('Пороговый эффект распространения информации', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    threshold = 0.3
    plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, label=f'Порог (~{threshold})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('threshold_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nАнализ порогового эффекта:")
    print("При низких вероятностях заражения информация не распространяется.")
    print("При превышении порогового значения происходит массовое распространение.")

    return sweep_results


def experiment_network_structure():
    print("\n=== Эксперимент: Влияние структуры сети ===")

    manager = SimulationManager()
    network_types = ['random', 'small_world', 'scale_free', 'complete']
    comparison = manager.compare_network_types(
        network_types=network_types,
        n_simulations=5,
        n_users=100,
        infection_prob=0.3,
        skeptic_prob=0.1,
        recovery_prob=0.05,
        initial_spreaders=2,
        max_steps=100
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['avg_infection_rate', 'avg_max_spreaders', 'avg_steps']
    metric_names = ['Доля информированных', 'Макс. распространители', 'Длительность']
    metric_colors = ['blue', 'green', 'red']

    for idx, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, metric_colors)):
        if idx < 3:
            values = [comparison[nt][metric] for nt in network_types]
            stds = [comparison[nt][f'std_{metric[4:]}'] for nt in network_types]

            bars = axes[idx].bar(network_types, values, color=color, alpha=0.7, yerr=stds, capsize=5)
            axes[idx].set_ylabel(metric_name, fontsize=12)
            axes[idx].set_title(f'{metric_name} по типам сетей', fontsize=14)
            axes[idx].tick_params(axis='x', rotation=45)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    ax = axes[3]
    ax.axis('off')

    info_text = "Сравнение типов сетей:\n\n"
    info_text += "1. Random: Случайные связи\n"
    info_text += "2. Small World: Локальные связи + случайные\n"
    info_text += "3. Scale Free: Степени по степенному закону\n"
    info_text += "4. Complete: Все связаны со всеми\n\n"

    for nt in network_types:
        info_text += f"{nt}: инфекция {comparison[nt]['avg_infection_rate']:.2%}\n"

    ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.suptitle('Влияние структуры сети на распространение информации',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, nt in enumerate(network_types):
        result = comparison[nt]['results'][0]
        history = result['history']

        steps = range(len(history['unaware']))
        ax = axes[idx]

        ax.plot(steps, history['unaware'], 'b-', label='Неосведомленные', alpha=0.7)
        ax.plot(steps, history['spreader'], 'r-', label='Распространители', alpha=0.7)
        ax.plot(steps, history['skeptic'], 'g-', label='Скептики', alpha=0.7)

        ax.set_xlabel('Шаг времени')
        ax.set_ylabel('Количество')
        ax.set_title(f'Тип сети: {nt}', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Динамика распространения в различных типах сетей',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('network_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

    return comparison


def experiment_initial_conditions():
    print("\n=== Эксперимент: Влияние начальных условий ===")

    manager = SimulationManager()
    initial_spreaders_list = [1, 2, 5, 10, 20]
    sweep_results = manager.run_parameter_sweep(
        param_name='initial_spreaders',
        param_values=initial_spreaders_list,
        n_simulations=3,
        n_users=200,
        network_type='small_world',
        infection_prob=0.3,
        skeptic_prob=0.1,
        recovery_prob=0.05,
        max_steps=100
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    infection_rates = [r['avg_infection_rate'] for r in sweep_results]
    axes[0].plot(initial_spreaders_list, infection_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Начальные распространители')
    axes[0].set_ylabel('Доля информированных')
    axes[0].set_title('Влияние начальных распространителей')
    axes[0].grid(True, alpha=0.3)

    max_spreaders = [r['avg_max_spreaders'] for r in sweep_results]
    axes[1].plot(initial_spreaders_list, max_spreaders, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Начальные распространители')
    axes[1].set_ylabel('Макс. распространителей')
    axes[1].set_title('Пиковое распространение')
    axes[1].grid(True, alpha=0.3)

    steps = [r['avg_steps'] for r in sweep_results]
    axes[2].plot(initial_spreaders_list, steps, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Начальные распространители')
    axes[2].set_ylabel('Длительность')
    axes[2].set_title('Время распространения')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Влияние начальных условий на распространение информации',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initial_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()

    return sweep_results


def run_all_experiments():
    print("\nНачало экспериментов...")

    results = {'basic': experiment_basic_simulation(),
               'threshold': experiment_threshold_effect(),
               'network_structure': experiment_network_structure(),
               'initial_conditions': experiment_initial_conditions()}
    print("Все эксперименты завершены!")

    return results


if __name__ == "__main__":
    all_results = run_all_experiments()
