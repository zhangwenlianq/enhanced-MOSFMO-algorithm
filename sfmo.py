"""羊群迁徙优化（Sheep Flock Migrate Optimization, SFMO）。

本模块是论文算法的简洁基准实现。算法只处理有界连续最小化问题；
如需求解最大化问题，可由调用者对目标函数取负。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
Objective = Callable[[Array], float]


@dataclass(frozen=True)
class SFMOResult:
    """SFMO 单次运行结果。"""

    best_position: Array
    """搜索到的历史最优位置。"""

    best_fitness: float
    """历史最优位置对应的目标函数值。"""

    convergence_curve: Array
    """每次外层迭代结束后的历史最优值。"""

    initial_best_fitness: float
    """初始化种群中的最优值，用于检查算法是否发生退化。"""

    evaluations: int
    """目标函数累计评价次数。"""


def _unit_vector(vector: Array) -> Array:
    """返回单位方向；零向量没有方向，因此仍返回零向量。"""
    length = float(np.linalg.norm(vector))
    if length <= np.finfo(float).eps:
        return np.zeros_like(vector)
    return vector / length


def _prepare_bound(
    bound: float | Sequence[float] | Array,
    dimension: int,
    name: str,
) -> Array:
    """把标量或逐维边界统一转换为长度为 ``dimension`` 的数组。"""
    try:
        result = np.broadcast_to(np.asarray(bound, dtype=float), (dimension,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} 必须是标量或长度为 dimension 的一维序列。") from exc

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须全部为有限数。")
    return result


def sfmo(
    objective: Objective,
    dimension: int,
    lower_bound: float | Sequence[float] | Array,
    upper_bound: float | Sequence[float] | Array,
    *,
    population_size: int = 50,
    max_iterations: int = 100,
    max_grazing_steps: int = 20,
    ndvi_threshold: float = 0.3,
    leader_search_step: float = 15.0,
    search_step: float = 10.0,
    migration_step: float = 10.0,
    direction_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    compensation_factor: float = 2.0,
    seed: int | None = None,
) -> SFMOResult:
    """使用 SFMO 最小化一个有界连续目标函数。

    Parameters
    ----------
    objective:
        接收形状为 ``(dimension,)`` 的位置向量并返回一个有限标量。
    dimension:
        决策变量维数。
    lower_bound, upper_bound:
        搜索空间上下界，可以是标量，也可以是逐维序列。
    population_size:
        羊群规模 ``Np``。
    max_iterations:
        外层最大迭代次数 ``T_max``。
    max_grazing_steps:
        每次迭代中放牧算子的最大搜索次数 ``T_GOmax``。
    ndvi_threshold:
        由放牧阶段切换到集体运动阶段的 NDVI 阈值。
    leader_search_step, search_step:
        头羊和普通羊的基准放牧步长。
    migration_step:
        羊群的基准迁徙步长。
    direction_weights:
        论文中的 ``(omega_alpha, omega_beta, omega_gamma)``，依次对应
        自身渴望方向、前一只羊的移动方向和朝前一只羊的跟随方向。
    compensation_factor:
        收缩后下一轮放牧范围的放大倍数，论文记为 ``e``。
    seed:
        NumPy 随机数种子；相同输入和种子可得到相同结果。
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("dimension 必须是正整数。")
    if (
        not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or population_size < 2
    ):
        raise ValueError("population_size 必须是不小于 2 的整数。")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations 必须是正整数。")
    if (
        not isinstance(max_grazing_steps, int)
        or isinstance(max_grazing_steps, bool)
        or max_grazing_steps <= 0
    ):
        raise ValueError("max_grazing_steps 必须是正整数。")

    lower = _prepare_bound(lower_bound, dimension, "lower_bound")
    upper = _prepare_bound(upper_bound, dimension, "upper_bound")
    if np.any(lower >= upper):
        raise ValueError("每一维都必须满足 lower_bound < upper_bound。")

    scalar_parameters = {
        "ndvi_threshold": ndvi_threshold,
        "leader_search_step": leader_search_step,
        "search_step": search_step,
        "migration_step": migration_step,
        "compensation_factor": compensation_factor,
    }
    if not all(np.isfinite(value) for value in scalar_parameters.values()):
        raise ValueError("所有算法参数都必须是有限数。")
    if not 0.0 <= ndvi_threshold <= 1.0:
        raise ValueError("ndvi_threshold 必须位于 [0, 1]。")
    if min(leader_search_step, search_step, migration_step, compensation_factor) <= 0:
        raise ValueError("搜索步长、迁徙步长和补偿因子必须大于 0。")

    weights = np.asarray(direction_weights, dtype=float)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)):
        raise ValueError("direction_weights 必须包含三个有限数。")
    if np.any(weights < 0.0) or not np.isclose(
        weights.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("direction_weights 必须非负且总和为 1。")
    omega_alpha, omega_beta, omega_gamma = weights

    rng = np.random.default_rng(seed)
    evaluations = 0

    def evaluate(position: Array) -> float:
        """统一记录评价次数，并拒绝非标量或非有限目标值。"""
        nonlocal evaluations
        evaluations += 1
        value = np.asarray(objective(position.copy()))
        if value.ndim != 0:
            raise ValueError("objective 必须返回一个标量。")
        fitness_value = float(value)
        if not np.isfinite(fitness_value):
            raise ValueError("objective 返回了 NaN 或 Inf。")
        return fitness_value

    def evaluate_population(positions: Array) -> Array:
        return np.asarray([evaluate(position) for position in positions], dtype=float)

    def clip(position: Array) -> Array:
        """论文未规定越界策略；本基准按已确认规则逐维截断。"""
        return np.clip(position, lower, upper)

    # 在搜索空间中均匀初始化，并从完整种群中选择历史最优个体。
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    fitness = evaluate_population(population)
    best_index = int(np.argmin(fitness))
    best_position = population[best_index].copy()
    best_fitness = float(fitness[best_index])
    initial_best_fitness = best_fitness
    convergence_curve = np.empty(max_iterations, dtype=float)

    # 只有执行过收缩策略，紧接着的一轮放牧才使用补偿因子。
    compensate_next_grazing = False

    def update_global_best(positions: Array, values: Array) -> None:
        nonlocal best_position, best_fitness
        index = int(np.argmin(values))
        if values[index] < best_fitness:  # 相等不视为改善
            best_fitness = float(values[index])
            best_position = positions[index].copy()

    for iteration in range(1, max_iterations + 1):
        # 每轮都按论文伪代码重置个体最优和 NDVI。
        iteration_start = population.copy()
        start_fitness = fitness.copy()
        personal_best = iteration_start.copy()
        personal_best_fitness = start_fitness.copy()
        ndvi = np.zeros(population_size, dtype=float)

        leader = int(np.argmin(start_fitness))
        group_best_before_grazing = float(start_fitness[leader])

        # 式（8）、（9）、（14）：步长随外层迭代线性衰减到 0.1。
        ratio = iteration / max_iterations
        leader_step = leader_search_step - (leader_search_step - 0.1) * ratio
        follower_step = search_step - (search_step - 0.1) * ratio
        collective_step = migration_step - (migration_step - 0.1) * ratio

        compensation_active = compensate_next_grazing
        compensate_next_grazing = False
        if compensation_active:
            leader_step *= compensation_factor
            follower_step *= compensation_factor

        # 放牧过程按 MATLAB 实现：普通轮围绕固定起点采样；补偿轮围绕
        # 已搜索到的个体最优位置继续采样。候选点不会直接替换种群位置。
        for _ in range(max_grazing_steps):
            for sheep_index in range(population_size):
                center = (
                    personal_best[sheep_index]
                    if compensation_active
                    else iteration_start[sheep_index]
                )
                step = leader_step if sheep_index == leader else follower_step
                disturbance = rng.uniform(-1.0, 1.0, size=dimension)
                candidate = clip(center + step * disturbance)
                candidate_fitness = evaluate(candidate)

                if candidate_fitness < personal_best_fitness[sheep_index]:
                    personal_best[sheep_index] = candidate
                    personal_best_fitness[sheep_index] = candidate_fitness

                    denominator = abs(start_fitness[sheep_index]) + abs(candidate_fitness)
                    ndvi[sheep_index] = (
                        abs(start_fitness[sheep_index] - candidate_fitness) / denominator
                        if denominator > 0.0
                        else 0.0
                    )

            # 满足论文条件 1 时提前结束；否则最多执行 T_GOmax 次搜索。
            if (
                float(np.min(personal_best_fitness)) < group_best_before_grazing
                and float(np.mean(ndvi)) > ndvi_threshold
            ):
                break

        update_global_best(personal_best, personal_best_fitness)

        # 羊群按照放牧阶段的个体最优适应度稳定排序。
        order = np.argsort(personal_best_fitness, kind="stable")
        direction_to_personal_best = np.asarray(
            [
                _unit_vector(personal_best[index] - iteration_start[index])
                for index in range(population_size)
            ]
        )

        if personal_best_fitness[order[0]] < group_best_before_grazing:
            # 移动策略 1：从放牧开始时的种群位置执行整体迁徙。
            while True:
                leader_index = int(order[0])
                previous_leader_fitness = float(fitness[leader_index])
                old_position = population[leader_index].copy()

                proposed = old_position + (
                    collective_step
                    * rng.random()
                    * direction_to_personal_best[leader_index]
                )
                population[leader_index] = clip(proposed)
                fitness[leader_index] = evaluate(population[leader_index])
                previous_move = population[leader_index] - old_position
                previous_index = leader_index

                # 其余羊只跟随排序中紧邻的前一只羊，并依次移动。
                for current in order[1:]:
                    current_index = int(current)
                    old_position = population[current_index].copy()

                    yearning_direction = direction_to_personal_best[current_index]
                    leader_direction = _unit_vector(previous_move)
                    following_direction = _unit_vector(
                        population[previous_index] - old_position
                    )
                    combined_direction = (
                        omega_alpha * yearning_direction
                        + omega_beta * leader_direction
                        + omega_gamma * following_direction
                    )
                    moving_direction = _unit_vector(combined_direction)

                    proposed = old_position + (
                        collective_step * rng.random() * moving_direction
                    )
                    population[current_index] = clip(proposed)
                    fitness[current_index] = evaluate(population[current_index])
                    previous_move = population[current_index] - old_position
                    previous_index = current_index

                update_global_best(population, fitness)

                # 论文要求头羊“不再优化”即停止：相等和变差都结束迁徙。
                if fitness[leader_index] >= previous_leader_fitness:
                    break
        else:
            # 移动策略 2：头羊不动，其余羊同时参考头羊和自身个体最优。
            leader_index = int(order[0])
            delta_1 = ratio
            delta_2 = 1.0 - ratio
            for current in order[1:]:
                current_index = int(current)
                old_position = population[current_index].copy()
                proposed = (
                    old_position
                    + delta_1
                    * rng.random()
                    * (population[leader_index] - old_position)
                    + delta_2
                    * rng.random()
                    * (personal_best[current_index] - old_position)
                )
                population[current_index] = clip(proposed)
                fitness[current_index] = evaluate(population[current_index])

            update_global_best(population, fitness)
            compensate_next_grazing = True

        convergence_curve[iteration - 1] = best_fitness

    return SFMOResult(
        best_position=best_position,
        best_fitness=best_fitness,
        convergence_curve=convergence_curve,
        initial_best_fitness=initial_best_fitness,
        evaluations=evaluations,
    )
