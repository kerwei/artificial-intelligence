import math
import random

from copy import deepcopy
from collections import namedtuple


def uct_search(state, statlist, nround):
    plyturn = state.ply_count % 2
    next_state, statlist = tree_policy(state, statlist, nround)
    if not next_state.terminal_test():
        delta = default_policy(next_state, plyturn)
        statlist = backup_negamax(statlist, delta, plyturn, nround)

    next_action = best_child(state, statlist, 0)

    return next_action, statlist


def tree_policy(state, statlist, nround):
    statecopy = deepcopy(state)

    while not statecopy.terminal_test():
        # All taken actions at this depth
        tried = [s.action for s in statlist if s.state == statecopy]
        # See if there's any untried actions left
        untried = [a for a in statecopy.actions() if a not in tried]

        topop = []
        toappend = []

        if len(untried) > 0:
            next_action = random.choice(untried)
            statecopy, statlist = expand(statecopy, statlist, next_action, nround)
            break
        else:
            next_action = best_child(statecopy, statlist, 1)

            for k, s in enumerate(statlist):
                if s.state == statecopy and s.action == next_action:
                    visit1 = statlist[k].visit + 1
                    news = statlist[k]._replace(visit=visit1)
                    news = news._replace(nround=nround)

                    topop.append(k)
                    toappend.append(news)
                    break

            statlist = update_scores(statlist, topop, toappend)
            statecopy = statecopy.result(next_action)

    return statecopy, statlist


def expand(state, statlist, action, nround):
    """
    Returns a state resulting from taking an action from the list of untried nodes
    """
    Stat = namedtuple('Stat', 'state action utility visit nround')
    plyturn = state.ply_count % 2

    next_state = state.result(action)
    delta = map_delta(next_state.utility(plyturn))

    statlist.append(Stat(state, action, delta, 1, nround))
    return next_state, statlist


def best_child(state, statlist, c):
    """
    Returns the state resulting from taking the best action
    c value between 0 (max score) and 1 (prioritize exploration)
    """
    # All taken actions at this depth
    tried = [s for s in statlist if s.state == state]

    maxscore = -999
    maxaction = []
    # Compute the score
    for t in tried:
        score = (t.utility/t.visit) + c * math.sqrt(2 * math.log(t.nround)/t.visit)
        if score > maxscore:
            maxscore = score
            del maxaction[:]
            maxaction.append(t.action)
        elif score == maxscore:
            maxaction.append(t.action)

    return random.choice(maxaction)


def default_policy(state, plyturn):
    """
    The simulation to run when visiting unexplored nodes. Defaults to uniform random moves
    """
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))

    return map_delta(state.utility(plyturn))


def backup_negamax(statlist, delta, plyturn, nround):
    """
    Propagates the terminal utility up the search tree
    """
    topop = []
    toappend = []
    for k, s in enumerate(statlist):
        if s.nround == nround:
            if s.state.ply_count % 2 == plyturn:
                utility1 = s.utility + delta
                news = statlist[k]._replace(utility=utility1)
            elif s.state.ply_count % 2 != plyturn:
                utility1 = s.utility - delta
                news = statlist[k]._replace(utility=utility1)

            topop.append(k)
            toappend.append(news)

    statlist = update_scores(statlist, topop, toappend)

    return statlist


def update_scores(statlist, topop, toappend):
    # Remove outdated tuples. Order needs to be in reverse or pop will fail!
    for p in sorted(topop, reverse=True):
        statlist.pop(p)
    # Add the updated ones
    for a in toappend:
        statlist.append(a)

    return statlist


def map_delta(delta):
    """
    Normalizes the state utility to the range between -1 and 1
    """
    if delta < 0:
        delta = -1
    elif delta > 0:
        delta = 1

    return delta


