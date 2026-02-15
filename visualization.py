import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from matplotlib.animation import FuncAnimation
from models import UserState, User

plt.style.use('seaborn-v0_8-darkgrid')


def plot_dynamics(history: Dict[str, List[int]],
                  title: str = "Динамика общественного мнения в социальной сети",
                  save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = range(len(history['unaware']))

    ax.plot(steps, history['unaware'], 'b-', linewidth=2, label='Неосведомленные')
    ax.plot(steps, history['spreader'], 'r-', linewidth=2, label='Распространители')
    ax.plot(steps, history['skeptic'], 'g-', linewidth=2, label='Скептики')

    ax.set_xlabel('Шаг времени', fontsize=14)
    ax.set_ylabel('Количество пользователей', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    total = history['unaware'][0] + history['spreader'][0] + history['skeptic'][0]
    final_informed = history['spreader'][-1] + history['skeptic'][-1]
    infection_rate = final_informed / total if total > 0 else 0

    info_text = f"Всего пользователей: {total}\n"
    info_text += f"Финальная информированность: {infection_rate:.1%}\n"
    info_text += f"Макс распространителей: {max(history['spreader'])}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_network(users: List[User],
                 edges: List[Tuple[int, int]],
                 title: str = "Структура социальной сети",
                 save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 10))

    G = nx.Graph()
    G.add_nodes_from([user.id for user in users])
    G.add_edges_from(edges)

    node_colors = []
    for user in users:
        if user.state == UserState.UNAWARE:
            node_colors.append('lightblue')
        elif user.state == UserState.SPREADER:
            node_colors.append('red')
        else:  # SKEPTIC
            node_colors.append('lightgreen')

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Неосведомленные'),
        Patch(facecolor='red', label='Распространители'),
        Patch(facecolor='lightgreen', label='Скептики')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    degrees = [len(user.neighbors) for user in users]
    info_text = f"Узлов: {len(users)}\n"
    info_text += f"Связей: {len(edges)}\n"
    info_text += f"Средняя степень: {np.mean(degrees):.2f}\n"
    info_text += f"Макс. степень: {max(degrees)}"

    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_multiple_simulations(results_list: List[Dict],
                              param_name: str,
                              title: str = "Сравнение различных параметров",
                              save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

    for idx, (result, color) in enumerate(zip(results_list, colors)):
        history = result['history']
        param_value = result['param_value']

        ax1 = axes[0]
        steps = range(len(history['unaware']))
        ax1.plot(steps, history['spreader'], color=color,
                 alpha=0.7, label=f'{param_name}={param_value}')

        ax2 = axes[1]
        final_informed = history['spreader'][-1] + history['skeptic'][-1]
        total = history['unaware'][0] + history['spreader'][0] + history['skeptic'][0]
        infection_rate = final_informed / total if total > 0 else 0
        ax2.bar(idx, infection_rate, color=color, alpha=0.7)

        ax3 = axes[2]
        max_spreaders = max(history['spreader'])
        ax3.bar(idx, max_spreaders, color=color, alpha=0.7)

        ax4 = axes[3]
        peak_time = history['spreader'].index(max_spreaders)
        ax4.bar(idx, peak_time, color=color, alpha=0.7)

    axes[0].set_xlabel('Шаг времени')
    axes[0].set_ylabel('Количество распространителей')
    axes[0].set_title('Динамика распространителей')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Параметр')
    axes[1].set_ylabel('Доля информированных')
    axes[1].set_title('Финальная информированность')
    axes[1].set_xticks(range(len(results_list)))
    axes[1].set_xticklabels([str(r['param_value']) for r in results_list])
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Параметр')
    axes[2].set_ylabel('Макс. распространителей')
    axes[2].set_title('Пиковое число распространителей')
    axes[2].set_xticks(range(len(results_list)))
    axes[2].set_xticklabels([str(r['param_value']) for r in results_list])
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel('Параметр')
    axes[3].set_ylabel('Шаг до пика')
    axes[3].set_title('Время достижения пика')
    axes[3].set_xticks(range(len(results_list)))
    axes[3].set_xticklabels([str(r['param_value']) for r in results_list])
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_animation(users_history: List[List[User]],
                     edges: List[Tuple[int, int]],
                     interval: int = 200,
                     save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    G = nx.Graph()
    G.add_nodes_from(range(len(users_history[0])))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    time_steps = []
    unaware_counts = []
    spreader_counts = []
    skeptic_counts = []

    def update(frame):
        ax1.clear()
        ax2.clear()

        users = users_history[frame]

        node_colors = []
        for user in users:
            if user.state == UserState.UNAWARE:
                node_colors.append('lightblue')
            elif user.state == UserState.SPREADER:
                node_colors.append('red')
            else:  # SKEPTIC
                node_colors.append('lightgreen')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=300, alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax1)

        counts = {state: 0 for state in UserState}
        for user in users:
            counts[user.state] += 1

        time_steps.append(frame)
        unaware_counts.append(counts[UserState.UNAWARE])
        spreader_counts.append(counts[UserState.SPREADER])
        skeptic_counts.append(counts[UserState.SKEPTIC])

        ax2.plot(time_steps, unaware_counts, 'b-', label='Неосведомленные')
        ax2.plot(time_steps, spreader_counts, 'r-', label='Распространители')
        ax2.plot(time_steps, skeptic_counts, 'g-', label='Скептики')

        ax1.set_title(f'Шаг {frame}', fontsize=14)
        ax1.axis('off')

        ax2.set_xlabel('Шаг времени')
        ax2.set_ylabel('Количество пользователей')
        ax2.set_title('Динамика распространения')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(users_history))
        ax2.set_ylim(0, len(users))

        info_text = f"Шаг: {frame}\n"
        info_text += f"Неосведомленные: {counts[UserState.UNAWARE]}\n"
        info_text += f"Распространители: {counts[UserState.SPREADER]}\n"
        info_text += f"Скептики: {counts[UserState.SKEPTIC]}"

        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    anim = FuncAnimation(fig, update, frames=len(users_history),
                         interval=interval, repeat=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=5)

    plt.tight_layout()
    plt.show()

    return anim