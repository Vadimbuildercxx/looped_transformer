import math


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_loops = args.loops.start

        self.__n_dims_schedule = args.dims
        self.__n_points_schedule = args.points
        self.__n_loops_schedule = args.loops
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.__n_dims_schedule)
        self.n_points = self.update_var(
            self.n_points, self.__n_points_schedule)
        self.n_loops = self.update_var(
            self.n_loops, self.__n_loops_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


class CurriculumSimple:
    def __init__(self,
                 dims_start,
                 points_start,
                 loops_start,
                 dims_schedule: list,
                 points_schedule: list,
                 loops_schedule: list):

        """schedule in format [update_every_steps, end_dims, increment step]"""
        self.n_dims_truncated = dims_start
        self.n_points = points_start
        self.n_loops = loops_start

        self.__n_dims_schedule = dims_schedule
        self.__n_points_schedule = points_schedule
        self.__n_loops_schedule = loops_schedule

        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.__n_dims_schedule)
        self.n_points = self.update_var(
            self.n_points, self.__n_points_schedule)
        self.n_loops = self.update_var(
            self.n_loops, self.__n_loops_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule[0] == 0:
            var += schedule[2]

        return min(var, schedule[1])


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor(total_steps / n_steps) * inc

    return min(final_var, lim)
