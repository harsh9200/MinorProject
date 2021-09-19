from .bee import Bee
from bisect import bisect
from random import randint, random


def apply_mutation(curr_params: list, all_params: list) -> list:
    ''' apply_mutation: alters the value of one parameter in supplied list of
    values
    Args:
        curr_params (list): current parameter values, ints or floats
        all_params (list): list of Parameter objects
    Returns:
        list: parameter values with one value mutation
    '''

    to_change = randint(0, len(curr_params) - 1)
    new_params = curr_params[:]
    new_params[to_change] = all_params[to_change].mutate(new_params[to_change])
    return new_params


def call_objective_fn(params: list, 
                      obj_fn: callable, 
                      obj_fn_args: dict
                      ) -> tuple:
    """Calls supplied objective function, evaluating using input parameters; 
        callable in single- and multi-processed configurations
    
    Args:
        params (list): list of ints or floats corresponding to current bee
            parameter values.
        obj_fn (callable): Function to accept list of paramters, returns a
            quantitative measurement of fitness.
        obj_fn_args (dict): Non-Tunable Kwargs to pass to objective function.
    
    Returns:
        tuple: (params, objective function return value)
    """

    return params, obj_fn(params, **obj_fn_args)


def choose_bee(bees: list) -> Bee:
    ''' choose_bee: choose a bee based on probabilities a given bee will be
    chosen; probabilities based on fitness score (higher fitness score ==
    higher probability of being chosen)
    Args:
        bees (list): list of Bee objects
    Returns:
        Bee: chosen Bee
    '''

    fitness_sum = sum(b._fitness_score for b in bees)
    probabilities = [b._fitness_score / fitness_sum for b in bees]
    cdf_vals = []
    cumsum = 0
    for p in probabilities:
        cumsum += p
        cdf_vals.append(cumsum)
    idx = bisect(cdf_vals, random())
    return bees[idx]


def determine_best_bee(bees: list) -> tuple:
    ''' determine_best_bee: return highest fitness score w/ corresponding
    objective function return value and parameters given a list of bees
    Args:
        bees (list): list of Bee objects
    Returns:
        tuple: (best fitness score, best return value, best parameters)
    '''

    best_fitness = 0
    best_ret_val = None
    best_params = None
    for bee in bees:
        if bee._fitness_score > best_fitness:
            best_fitness = bee._fitness_score
            best_ret_val = bee._obj_fn_val
            best_params = bee._params
    return (best_fitness, best_ret_val, best_params)