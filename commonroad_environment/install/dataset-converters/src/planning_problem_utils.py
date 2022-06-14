import random

from commonroad.scenario.trajectory import State
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle


class NoCarException(Exception):
    pass


def obstacle_to_planning_problem(obstacle: DynamicObstacle, planning_problem_id: int, final_time_step=None,
                                 orientation_half_range: float = 0.2,
                                 velocity_half_range: float = 10, time_step_half_range: int = 25):
    """
    Generates planning problem using initial and final states of a DynamicObstacle
    """
    dynamic_obstacle_shape = obstacle.obstacle_shape
    dynamic_obstacle_initial_state = obstacle.initial_state
    dynamic_obstacle_final_state = obstacle.prediction.trajectory.final_state

    # define orientation, velocity and time step intervals as goal region
    orientation_interval = AngleInterval(dynamic_obstacle_final_state.orientation - orientation_half_range,
                                         dynamic_obstacle_final_state.orientation + orientation_half_range)
    velocity_interval = Interval(dynamic_obstacle_final_state.velocity - velocity_half_range,
                                 dynamic_obstacle_final_state.velocity + velocity_half_range)
    if final_time_step is None:
        final_time_step = dynamic_obstacle_final_state.time_step + time_step_half_range

    time_step_interval = Interval(0, final_time_step)

    goal_position = Rectangle(dynamic_obstacle_shape.length, dynamic_obstacle_shape.width,
                              center=dynamic_obstacle_final_state.position,
                              orientation=dynamic_obstacle_final_state.orientation)
    goal_region = GoalRegion([State(position=goal_position, orientation=orientation_interval,
                                    velocity=velocity_interval, time_step=time_step_interval)])

    dynamic_obstacle_initial_state.yaw_rate = 0.0
    dynamic_obstacle_initial_state.slip_angle = 0.0

    return PlanningProblem(planning_problem_id, dynamic_obstacle_initial_state, goal_region)


def generate_planning_problem(scenario: Scenario, orientation_half_range: float = 0.2, velocity_half_range: float = 10,
                              time_step_half_range: int = 25, keep_ego: bool = False,
                              dynamic_obstacle_selected=None) -> PlanningProblem:
    """
    Generates planning problem for scenario by taking obstacle trajectory
    :param scenario: CommonRoad scenario
    :param orientation_half_range: parameter for goal state orientation
    :param velocity_half_range: parameter for goal state velocity
    :param time_step_half_range: parameter for goal state time step
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param dynamic_obstacle_selected: currently just for testing the interaction dataset
    :return: CommonRoad planning problem
    """
    # random choose obstacle as ego vehicle
    random.seed(0)

    # only choose car type as ego vehicle
    car_obstacles = [obstacle for obstacle in scenario.dynamic_obstacles if obstacle.obstacle_type == ObstacleType.CAR
                     and obstacle.initial_state.time_step == 0]
    if len(car_obstacles) > 0:
        dynamic_obstacle_selected = random.choice(car_obstacles)
    else:
        raise NoCarException("There is no car in dynamic obstacles which can be used as planning problem.")

    if not keep_ego:
        planning_problem_id = dynamic_obstacle_selected.obstacle_id
        scenario.remove_obstacle(dynamic_obstacle_selected)
    else:
        planning_problem_id = scenario.generate_object_id()

    if len(scenario.dynamic_obstacles) > 0:
        max_time_step = max([obs.prediction.trajectory.state_list[-1].time_step for obs in scenario.dynamic_obstacles])
        final_time_step = min(
            dynamic_obstacle_selected.prediction.trajectory.final_state.time_step + time_step_half_range, max_time_step)
    else:
        final_time_step = dynamic_obstacle_selected.prediction.trajectory.final_state.time_step + time_step_half_range

    planning_problem = obstacle_to_planning_problem(dynamic_obstacle_selected,
                                                    planning_problem_id,
                                                    final_time_step=final_time_step,
                                                    orientation_half_range=orientation_half_range,
                                                    velocity_half_range=velocity_half_range,
                                                    time_step_half_range=time_step_half_range)

    return planning_problem
