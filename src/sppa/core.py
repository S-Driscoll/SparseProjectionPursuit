"""
Sparse Projection Pursuit Analysis (SPPA) — genetic algorithm core.

Port of the MATLAB ppga2 / SPPA functions from SPPA.m.

Public entry point: ``sppa(X, ...)``
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

from .utils import projpursuit


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _rows_match_any(pool: np.ndarray, row: np.ndarray) -> bool:
    """Return True if *row* (sorted) appears in *pool* (rows sorted) more than once."""
    sorted_row = np.sort(row)
    sorted_pool = np.sort(pool, axis=1)
    return int(np.all(sorted_pool == sorted_row, axis=1).sum()) > 1


def _mutation(
    nchild: int,
    mutrate: float,
    nvars: int,
    totvars: int,
    popelite: np.ndarray,
    popchild: np.ndarray,
) -> np.ndarray:
    """
    Apply random mutation to the child population.

    No duplicate variables within an individual, and no duplicate individuals
    in the combined elite + child population.

    Parameters
    ----------
    nchild : int
        Number of child individuals.
    mutrate : float
        Mutation rate (0 < mutrate < 1).
    nvars : int
        Number of variables per individual.
    totvars : int
        Total number of variables available (0-based indices: 0..totvars-1).
    popelite : ndarray, shape (numret, nvars)
        Elite individuals carried forward unchanged.
    popchild : ndarray, shape (nchild, nvars)
        Children from mating (modified in-place).

    Returns
    -------
    pop : ndarray, shape (numret + nchild, nvars)
        New population = elite ∥ mutated children.
    """
    combined = np.vstack([popelite, popchild])
    for i in range(nchild):
        while _rows_match_any(combined, popchild[i]):
            # Decide which loci to mutate
            mutloc = np.random.rand(nvars) / mutrate
            locs = np.where(mutloc < 1)[0]
            for j in locs:
                new_var = np.random.randint(0, totvars)
                while np.sum(popchild[i] == new_var) > 1:
                    new_var = np.random.randint(0, totvars)
                popchild[i, j] = new_var
            # Rebuild combined for next duplicate check
            combined = np.vstack([popelite, popchild])
    return np.vstack([popelite, popchild])


def _mating7(
    nvars: int,
    pop: np.ndarray,
    nchild: int,
    fitness: np.ndarray,
    newpop: np.ndarray,
    totvars: int,
    pctrecomb: float,
    ctoff: float,
    exponent: float,
) -> np.ndarray:
    """
    Fitness-proportional mating with uniform crossover.

    Parameters
    ----------
    nvars : int
        Variables per individual.
    pop : ndarray, shape (popsize, nvars)
        Current sorted population (best = row 0).
    nchild : int
        Number of children to produce (must be even).
    fitness : ndarray, shape (popsize,)
        Fitness values (kurtosis — lower is better, sorted ascending).
    newpop : ndarray, shape (numret, nvars)
        Elite individuals already committed to next generation.
    totvars : int
        Total variables available.
    pctrecomb : float
        Recombination probability per locus (fraction of loci exchanged).
    ctoff : float
        Fitness floor — individuals below this are treated as equivalent.
    exponent : float
        Exponent for fitness-to-probability conversion (y = 1/f^exponent).

    Returns
    -------
    popchild : ndarray, shape (nchild, nvars)
    """
    popchild = np.full((nchild, nvars), -1, dtype=int)  # -1 = uninitialised sentinel
    fitness_clipped = np.where(fitness < ctoff, ctoff, fitness)
    y = 1.0 / (fitness_clipped ** exponent)
    ranks = np.cumsum(y) / y.sum()

    combined = np.vstack([newpop, popchild])

    i = 0
    while i < nchild - 1:
        it = 0
        parent1 = np.zeros(nvars, dtype=int)
        parent2 = np.zeros(nvars, dtype=int)

        while True:
            # No-duplicate-parent guard
            while np.array_equal(parent1, parent2):
                r1 = np.random.rand()
                r2 = np.random.rand()
                parent1 = pop[np.searchsorted(ranks, r1)]
                parent2 = pop[np.searchsorted(ranks, r2)]

            # Crossover
            crossover = np.random.rand(nvars) / pctrecomb
            loc1 = crossover < 1
            loc2 = np.random.choice(np.where(loc1)[0], size=int(loc1.sum()), replace=False) if loc1.any() else np.array([], dtype=int)

            child1 = parent1.copy()
            child2 = parent2.copy()
            if len(loc2) > 0:
                child1[loc1] = parent2[loc2]
                child2[loc2] = parent1[loc1]

            popchild[i] = child1
            popchild[i + 1] = child2
            combined = np.vstack([newpop, popchild])

            # Validate: no duplicate variables within each child
            ok1 = len(np.unique(child1)) == nvars
            ok2 = len(np.unique(child2)) == nvars
            # No duplicate individuals in population
            ok3 = not _rows_match_any(combined, child1)
            ok4 = not _rows_match_any(combined, child2)

            if ok1 and ok2 and ok3 and ok4:
                break

            it += 1
            if it > 30:
                # Give up and use random individuals
                popchild[i] = np.random.choice(totvars, nvars, replace=False)
                popchild[i + 1] = np.random.choice(totvars, nvars, replace=False)
                break

        i += 2

    return popchild


# ---------------------------------------------------------------------------
# Main GA loop: ppga2 (dimension-by-dimension)
# ---------------------------------------------------------------------------

def _ppga2(
    X: np.ndarray,
    dim: int = 2,
    nvars: int = 5,
    maxgen: int = 1000,
    maxtime: float = 300.0,
    meth: str = "uni",
    opt: str = "ord",
    mutrate: float = 0.1,
    popsize: int = 100,
    pctrecomb: float = 0.3,
    stat: int = 50,
    exponent: float = 4.0,
    ctoff: float = 1.5,
    classes: np.ndarray | None = None,
    class_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dimension-by-dimension sparse PP genetic algorithm (port of ppga2).

    Returns
    -------
    scores : ndarray (n_samples, dim)
    vectors : ndarray (n_vars_total, dim)
    variables : ndarray (dimvar, nvars)  — 0-based variable indices
    kurt : ndarray (dimvar,)
    pops : ndarray (popsize, nvars, dim)
    """
    meth = meth.lower()
    opt = opt.lower()

    if meth == "mul":
        dimvar = 1
        pursuitdim = dim
    else:
        dimvar = dim
        pursuitdim = 1

    nsamp, totvars = X.shape
    scores = np.zeros((nsamp, dim))
    variables = np.zeros((dimvar, nvars), dtype=int)
    vectors = np.zeros((totvars, dim))
    kurt = np.zeros(dimvar)

    numret = 1 if popsize % 2 else 2
    nchild = popsize - numret

    # Mean-centre
    Morig = X.mean(axis=0)
    X = X - Morig
    X0 = X.copy()

    pops = np.zeros((popsize, nvars, dim), dtype=int)

    fitness_histories: list[np.ndarray] = []

    start_time = time.time()

    T1 = np.zeros((nsamp, dim + 1))  # extra column for the cross-product deflation step
    P_defl = np.zeros((totvars, dim + 1))

    for d in range(dim):
        pop = np.zeros((popsize, nvars), dtype=int)
        populations = np.full((maxgen, popsize, nvars), -1, dtype=int)
        fitness = np.full((popsize, maxgen), np.inf)

        # ------------------------------------------------------------------
        # Initial population with balanced variable coverage
        # ------------------------------------------------------------------
        n1 = int(np.ceil(totvars / nvars))    # individuals per complete group
        n2 = totvars % nvars                   # leftover in last group of group-building
        n3 = popsize // n1                     # number of complete groups
        n4 = popsize % n1                      # leftover individuals

        for igrp in range(n3):
            valid = False
            while not valid:
                perm1 = np.random.permutation(totvars)
                # Pad last chunk so it has nvars unique elements
                if n2 > 0:
                    extra = np.random.choice(
                        [v for v in range(totvars) if v not in perm1[-n2:]],
                        size=nvars - n2,
                        replace=False,
                    )
                    perm1 = np.concatenate([perm1, extra])
                # perm1 has n1*nvars elements arranged in n1 rows of nvars
                indx1 = perm1[: n1 * nvars].reshape(n1, nvars)
                # Check last row has no duplicates
                valid = len(np.unique(indx1[-1])) == nvars

            base = igrp * n1
            for jj in range(n1):
                idx = base + jj
                pop[idx] = indx1[jj]
                _, _, ppout = projpursuit(X[:, pop[idx]], pursuitdim, 1, meth, opt)
                fitness[idx, 0] = ppout["K"]

        # Residual
        base = n3 * n1
        if n4 > 0:
            flat = np.random.choice(totvars, n4 * nvars, replace=False)
            indx1 = flat.reshape(n4, nvars)
            for jj in range(n4):
                idx = base + jj
                pop[idx] = indx1[jj]
                _, _, ppout = projpursuit(X[:, pop[idx]], pursuitdim, 1, meth, opt)
                fitness[idx, 0] = ppout["K"]

        isort = np.argsort(fitness[:, 0])
        fitness[:, 0] = fitness[isort, 0]
        pop = pop[isort]
        populations[0] = pop.copy()

        # ------------------------------------------------------------------
        # GA generational loop
        # ------------------------------------------------------------------
        converged_gen = maxgen  # will be updated on early stop
        for k in range(1, maxgen):
            popelite = pop[:numret].copy()

            popchild = _mating7(
                nvars, pop, nchild, fitness[:, k - 1],
                popelite, totvars, pctrecomb, ctoff, exponent,
            )
            pop = _mutation(nchild, mutrate, nvars, totvars, popelite, popchild)

            fitness2 = fitness.copy()
            for i in range(popsize):
                _, _, ppout = projpursuit(X[:, pop[i]], pursuitdim, 1, meth, opt)
                fitness[i, k] = ppout["K"]

                # Check last 50 generations for previously seen individual
                for r in range(1, min(51, k)):
                    matches = np.all(populations[k - r] == pop[i], axis=1)
                    if matches.any():
                        loc = int(np.argmax(matches))
                        fitness[i, k] = min(fitness[i, k], fitness2[loc, k - r])
                        break

            isort = np.argsort(fitness[:, k])
            fitness[:, k] = fitness[isort, k]

            oldpop = pop.copy()
            for i in range(popsize):
                pop[i] = oldpop[isort[i]]

            populations[k] = pop.copy()

            med_k = float(np.median(fitness[:, k]))
            min_k = float(fitness[0, k])
            dim_tag = f"dim {d + 1}  " if meth != "mul" else ""
            print(f"  gen {k:5d}  {dim_tag}median kurtosis: {med_k:8.4f}  min: {min_k:8.4f}")

            # Convergence checks
            if k == maxgen - 1:
                print(f"Failed to converge after {k + 1} generations")
                converged_gen = k
                break
            if k > stat and np.array_equal(populations[k][0], populations[k - stat][0]):
                print(f"Stopping due to static population ({k + 1} generations)")
                converged_gen = k
                break
            if time.time() - start_time > maxtime * (d + 1):
                print(f"Maximum time reached ({maxtime} seconds)")
                converged_gen = k
                break
        else:
            converged_gen = maxgen - 1

        fitness_histories.append(fitness[:, :converged_gen + 1])

        # ------------------------------------------------------------------
        # Extract best solution for this dimension
        # ------------------------------------------------------------------
        variables[d] = np.sort(pop[0])

        if meth == "mul":
            T_1, V_1, ppout = projpursuit(X[:, variables[d]], pursuitdim, 100, meth, opt)
            V2 = np.zeros((totvars, dim))
            V2[variables[d]] = V_1
            vectors = V2
            kurt[0] = ppout["K"]
            pops[:, :, d] = pop
            # Multivariate: all dims extracted at once
            scores = T_1
            break

        T_1, V_1, ppout = projpursuit(X[:, variables[d]], 1, 100, meth, opt)
        kurt[d] = ppout["K"]

        V2 = np.zeros(totvars)
        V2[variables[d]] = V_1.ravel()
        scores[:, d] = T_1.ravel() + (Morig @ V2) - (X0.mean(axis=0) @ V2)
        vectors[:, d] = V2
        pops[:, :, d] = pop

        t = X @ V2
        T1[:, d] = t
        P_defl[:, d] = X.T @ t / (t @ t)

        if d < dim - 1:  # Deflation
            X = X0 - T1[:, :dim] @ P_defl[:, :dim].T
            if d == 1:   # Special cross-product deflation for 3rd dimension
                t_cross = scores[:, 0] * scores[:, 1]
                T1[:, 2] = t_cross
                P_defl[:, 2] = X.T @ t_cross / (t_cross @ t_cross)
                X = X0 - T1[:, :3] @ P_defl[:, :3].T

    if meth != "mul":
        # Correct vectors for deflation: V_corrected = V * inv(P'V)
        V_raw = vectors
        P_raw = P_defl[:, :dim]
        try:
            V_corr = V_raw @ np.linalg.inv(P_raw.T @ V_raw)
        except np.linalg.LinAlgError:
            V_corr = V_raw  # fallback
        vectors = V_corr
        scores = X0 @ V_corr

    if _MATPLOTLIB:
        _plot_fitness(fitness_histories)

    # ------------------------------------------------------------------
    # Final scatter plot (if class labels provided)
    # ------------------------------------------------------------------
    if _MATPLOTLIB and classes is not None:
        _plot_scores(scores, dim, classes, class_labels)

    return scores, vectors, variables, kurt, pops


def _plot_fitness(fitness_histories: list[np.ndarray]) -> None:
    """Static plot of kurtosis convergence for each optimised dimension."""
    n = len(fitness_histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for d, (ax, fh) in enumerate(zip(axes[0], fitness_histories)):
        gens = np.arange(fh.shape[1])
        med_fit = np.median(fh, axis=0)
        min_fit = np.min(fh, axis=0)
        ax.plot(gens, med_fit, color="tab:red", linewidth=2.5, label="Median")
        ax.plot(gens, min_fit, color="tab:blue", linewidth=2.5, label="Minimum")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Kurtosis")
        ax.set_title("Kurtosis convergence" if n == 1 else f"Dimension {d + 1}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("GA fitness progression", fontsize=13)
    fig.tight_layout()


def _plot_scores(
    scores: np.ndarray,
    dim: int,
    classes: np.ndarray,
    class_labels: Sequence[str] | None,
) -> None:
    """Scatter plot of PP scores coloured by class membership."""
    colors = np.array([
        [255, 57, 33], [215, 25, 232], [53, 33, 255], [29, 156, 207],
        [12, 178, 85], [122, 43, 12], [0, 0, 0], [29, 84, 74],
    ], dtype=float) / 255.0

    n_classes = int(classes.max())
    if class_labels is None:
        class_labels = [f"Group {i + 1}" for i in range(n_classes)]

    fig, ax = plt.subplots()
    if dim == 1:
        for i in range(n_classes):
            mask = classes == (i + 1)
            ax.plot(scores[mask, 0], scores[mask, 0], ".", color=colors[i % len(colors)], markersize=5)
    elif dim == 2:
        for i in range(n_classes):
            mask = classes == (i + 1)
            ax.plot(scores[mask, 0], scores[mask, 1], ".", color=colors[i % len(colors)], markersize=10)
        ax.set_xlabel("Score 1")
        ax.set_ylabel("Score 2")
    elif dim == 3:
        ax3 = fig.add_subplot(111, projection="3d")
        for i in range(n_classes):
            mask = classes == (i + 1)
            ax3.plot(
                scores[mask, 0], scores[mask, 1], scores[mask, 2],
                ".", color=colors[i % len(colors)], markersize=10,
            )
    ax.legend(class_labels, loc="best")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sppa(
    X: np.ndarray,
    dim: int = 2,
    nvars: int = 5,
    mutrate: float = 0.1,
    popsize: int = 100,
    opt: str = "ord",
    meth: str = "uni",
    maxtime: float = 300.0,
    pctrecomb: float = 0.3,
    exponent: float = 4.0,
    ctoff: float | None = None,
    stat: int = 50,
    maxgen: int = 1000,
    classes: np.ndarray | None = None,
    class_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse Projection Pursuit Analysis (SPPA).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_variables)
        Data matrix (samples × variables).
    dim : int
        Number of separation dimensions (1–3). Default 2.
    nvars : int
        Number of variables to select per dimension. Default 5.
        Recommendation: nvars ≈ n_samples / 25, with nvars ≥ 3.
    mutrate : float
        Mutation rate, in [0, 1). Default 0.1.
    popsize : int
        GA population size. Default 100.
    opt : {'ord', 'rec'}
        'ord' — ordinary kurtosis (default).
        'rec' — recentered kurtosis (better for unbalanced classes).
    meth : {'uni', 'mul'}
        'uni' — stepwise univariate kurtosis (default).
        'mul' — simultaneous multivariate kurtosis.
    maxtime : float
        Maximum wall-clock time in seconds before early stopping. Default 300.
    pctrecomb : float
        Fraction of loci exchanged during crossover. Default 0.3.
    exponent : float
        Fitness exponent for parent selection. Default 4.
    ctoff : float or None
        Fitness floor for mating selection. If None, defaults to 1.5 (uni)
        or 4.5 (mul).
    stat : int
        Number of static generations before stopping. Default 50.
    maxgen : int
        Maximum number of generations. Default 1000.
    classes : ndarray of int, shape (n_samples,), optional
        Integer class labels (1-based) for scatter plot colouring.
    class_labels : list of str, optional
        Human-readable class names for the legend.

    Returns
    -------
    T : ndarray, shape (n_samples, dim)
        Projection pursuit scores.
    V : ndarray, shape (n_variables, dim)
        Projection vectors.
    Var : ndarray, shape (dimvar, nvars)
        Selected variable indices (0-based) for each dimension.
        dimvar = 1 for multivariate kurtosis, else = dim.
    kurt : ndarray, shape (dimvar,)
        Kurtosis values for the solution in each dimension.

    Examples
    --------
    >>> T, V, Var, kurt = sppa(X, dim=2, nvars=5, meth='uni')
    >>> T, V, Var, kurt = sppa(X, dim=2, nvars=10, opt='rec', mutrate=0.2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (samples × variables).")

    meth = meth.lower()
    opt = opt.lower()

    if meth not in ("uni", "mul"):
        raise ValueError("meth must be 'uni' or 'mul'.")
    if opt not in ("ord", "rec"):
        raise ValueError("opt must be 'ord' or 'rec'.")
    if not (0 < dim < 4):
        raise ValueError("dim must be 1, 2, or 3.")
    if nvars < 2:
        raise ValueError("nvars must be >= 2.")
    if not (0 <= mutrate < 1):
        raise ValueError("mutrate must be in [0, 1).")
    if popsize < 2:
        raise ValueError("popsize must be >= 2.")

    # Default cutoff
    if ctoff is None:
        ctoff = 4.5 if meth == "mul" else 1.5

    scores, vectors, variables, kurt, _pops = _ppga2(
        X,
        dim=dim,
        nvars=nvars,
        maxgen=maxgen,
        maxtime=maxtime,
        meth=meth,
        opt=opt,
        mutrate=mutrate,
        popsize=popsize,
        pctrecomb=pctrecomb,
        stat=stat,
        exponent=exponent,
        ctoff=ctoff,
        classes=classes,
        class_labels=class_labels,
    )

    return scores, vectors, variables, kurt
