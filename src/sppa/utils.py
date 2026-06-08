"""
Projection pursuit algorithms ported from MATLAB (SPPA.m).

Four algorithm variants are provided:
    - Ordinary univariate kurtosis PP   (_okurtpp)
    - Ordinary multivariate kurtosis PP  (_mulkurtpp)
    - Recentered univariate kurtosis PP  (_rckurtpp)
    - Recentered multivariate kurtosis PP(_rcmulkurtpp)

All are dispatched through the public ``projpursuit`` function.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def orthbasis(A: np.ndarray) -> np.ndarray:
    """Return an orthonormal column basis for A via the Gram-Schmidt process."""
    c = A.shape[1]
    V = np.empty_like(A, dtype=float)
    v0 = A[:, 0]
    V[:, 0] = v0 / np.linalg.norm(v0)
    for i in range(1, c):
        tem = A[:, i] - V[:, :i] @ (V[:, :i].T @ A[:, i])
        V[:, i] = tem / np.linalg.norm(tem)
    return V


def _kurtosis_cols(X: np.ndarray) -> np.ndarray:
    """Biased Pearson kurtosis per column (matches MATLAB kurtosis(X,1,1))."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    diff = X - mu
    m2 = np.mean(diff ** 2, axis=0)
    m4 = np.mean(diff ** 4, axis=0)
    return m4 / (m2 ** 2)


def _sqrtm(A: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root via eigendecomposition (A must be PSD)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)  # guard against tiny negative values
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Ordinary univariate kurtosis PP  (port of okurtpp)
# ---------------------------------------------------------------------------

def _okurtpp(
    X: np.ndarray,
    p: int = 2,
    guess: int = 100,
    maxmin: str = "Min",
    stsh: str = "Sh",
    vsorth: str = "SO",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Quasi-power method to optimise univariate kurtosis.

    Returns
    -------
    T, V, W, P, kurtObj, convFlag
    """
    Morig = X.mean(axis=0)
    X = X - Morig
    rk = np.linalg.matrix_rank(X)

    Uorig, Sorig, Vhorig = np.linalg.svd(X, full_matrices=False)
    Worig = Vhorig.T  # MATLAB V in [U,S,V]=svd(X,'econ')
    X = Uorig @ np.diag(Sorig)
    X = X[:, :rk]
    Worig = Worig[:, :rk]
    X0 = X.copy()

    r, c = X.shape
    maxcount = 10000
    convFlag: list[list[str]] = [[""] * p for _ in range(guess)]
    kurtObj = np.zeros((guess, p))
    T = np.zeros((r, p))
    W_mat = np.zeros((c, p))
    P_mat = np.zeros((c, p))

    stsh0 = stsh

    for j in range(p):
        cc = c - j  # dimensionality decreases each step
        convlimit = 1e-10 * cc

        wall = np.zeros((cc, guess))

        _, _, Vjh = np.linalg.svd(X, full_matrices=False)
        Vj = Vjh.T[:, :cc]  # (c_prev) × cc
        X = X @ Vj           # r × cc

        if maxmin == "Max":
            invMat2 = 1.0 / np.diag(X.T @ X)
        else:  # "Min"
            Mat2 = np.diag(X.T @ X)
            VM = np.zeros((cc * cc, r))
            for i in range(r):
                tem = X[i:i+1, :].T @ X[i:i+1, :]
                VM[:, i] = tem.ravel()

        for k in range(guess):
            w = np.random.randn(cc)
            w /= np.linalg.norm(w)
            oldw1 = w.copy()
            oldw2 = oldw1.copy()
            stsh = stsh0
            count = 0

            while True:
                count += 1
                x = X @ w

                if maxmin == "Max":
                    w = invMat2 * (X.T @ (x * x * x))
                else:  # "Min"
                    Mat1 = (VM @ (x * x)).reshape(cc, cc)
                    w = np.linalg.solve(Mat1, Mat2 * w)

                w /= np.linalg.norm(w)
                L1 = (w @ oldw1) ** 2

                if (1 - L1) < convlimit:
                    convFlag[k][j] = "Converged"
                    break
                if count > maxcount:
                    convFlag[k][j] = "Not converged"
                    break

                if stsh == "Sh":
                    w = w + 0.5 * oldw1
                    w /= np.linalg.norm(w)
                elif maxmin == "Min":
                    L2 = (w @ oldw2) ** 2
                    if L2 > L1 and L2 > 0.99:
                        stsh = "Sh"
                    oldw2 = oldw1.copy()

                oldw1 = w.copy()

            wall[:, k] = w

        kurtObj[:, j] = _kurtosis_cols(X @ wall)

        if maxmin == "Max":
            ind = int(np.argmax(kurtObj[:, j]))
        else:
            ind = int(np.argmin(kurtObj[:, j]))

        Wj = wall[:, ind]

        if vsorth == "VO":
            t = X @ Wj
            T[:, j] = t
            W_mat[:, j] = Vj @ Wj
            X = X0 - X0 @ W_mat @ W_mat.T
        else:  # "SO"
            t = X @ Wj
            T[:, j] = t
            W_mat[:, j] = Vj @ Wj
            Pj = X.T @ t / (t @ t)
            P_mat[:, j] = Vj @ Pj
            X = X0 - T @ P_mat.T

    # Transform back to original space
    W_mat = Worig @ W_mat
    if vsorth == "VO":
        V = W_mat
        W_out = None
        P_out = None
        T = T + (Morig @ V)
    else:
        P_mat = Worig @ P_mat
        V = W_mat @ np.linalg.inv(P_mat.T @ W_mat)
        T = T + (Morig @ V)
        norms = np.sqrt(np.sum(V ** 2, axis=0))
        V = V / norms
        T = T / norms
        P_mat = P_mat * norms
        W_out = W_mat
        P_out = P_mat

    return T, V, W_out, P_out, kurtObj, convFlag


# ---------------------------------------------------------------------------
# Ordinary multivariate kurtosis PP  (port of mulkurtpp)
# ---------------------------------------------------------------------------

def _mulkurtpp(
    X: np.ndarray,
    p: int = 2,
    guess: int = 100,
    maxmin: str = "Min",
    stsh: str = "Sh",
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, list]:
    """
    Quasi-power method to optimise multivariate kurtosis.

    Returns
    -------
    T, V, Vall, kurtObj, convFlag
    """
    Morig = X.mean(axis=0)
    X = X - Morig
    rk = np.linalg.matrix_rank(X)

    Uorig, Sorig, Vhorig = np.linalg.svd(X, full_matrices=False)
    Vorig = Vhorig.T  # MATLAB Vorig
    X = Uorig @ np.diag(Sorig)
    X = X[:, :rk]
    Vorig = Vorig[:, :rk]

    r, c = X.shape
    maxcount = 10000
    convlimit = 1e-10

    Vall: list = [None] * guess
    kurtObj = np.zeros(guess)
    convFlag: list = [""] * guess

    for k in range(guess):
        V = np.random.randn(c, p)
        V, _, _ = np.linalg.svd(V, full_matrices=False)  # orth(V)
        oldV = V.copy()
        count = 0

        while True:
            count += 1
            A = V.T @ X.T @ X @ V
            Ainv = np.linalg.inv(A)

            scal_mat = _sqrtm(Ainv) @ V.T @ X.T          # p × r
            scal = np.sqrt(np.sum(scal_mat ** 2, axis=0))  # length r

            Mat = ((np.ones((c, 1)) * scal) * X.T)   # c × r  (broadcast)
            Mat = Mat @ Mat.T                         # c × c

            XTX = X.T @ X
            if maxmin == "Max":
                M = np.linalg.solve(XTX, Mat)
            else:  # "Min"
                M = np.linalg.solve(Mat, XTX)

            if stsh == "St":
                V = M @ V
            elif stsh == "Sh":
                V = (M + np.eye(c) * np.trace(M) / c) @ V

            V, _, _ = np.linalg.svd(V, full_matrices=False)  # orthonormal

            diff = np.sum((oldV - V) ** 2) / (c * p)
            if diff < convlimit:
                convFlag[k] = "Converged"
                break
            if count > maxcount:
                convFlag[k] = "Not converged"
                break
            oldV = V.copy()

        # Multivariate kurtosis value
        scal_k = _sqrtm(Ainv) @ V.T @ X.T
        scal_k = np.sqrt(np.sum(scal_k ** 2, axis=0))
        kurtObj[k] = r * np.sum(scal_k ** 4)

        U_t, _, Vt_t = np.linalg.svd(X @ V @ V.T, full_matrices=False)
        Vt_out = Vorig @ Vt_t.T[:, :p]
        Vall[k] = Vt_out

    if maxmin == "Max":
        ind = int(np.argmax(kurtObj))
    else:
        ind = int(np.argmin(kurtObj))

    V = Vall[ind]
    T = X @ Vorig.T @ V + Morig @ V
    return T, V, Vall, kurtObj, convFlag


# ---------------------------------------------------------------------------
# Recentered univariate kurtosis PP  (port of rckurtpp)
# ---------------------------------------------------------------------------

def _rckurtpp(
    X: np.ndarray,
    p: int = 2,
    guess: int = 100,
    vsorth: str = "SO",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Quasi-power method to minimise recentered univariate kurtosis.

    Returns
    -------
    T, V, R, W, P, kurtObj, convFlag
    """
    Morig = X.mean(axis=0)
    X = X - Morig
    rk = np.linalg.matrix_rank(X)

    Uorig, Sorig, Vhorig = np.linalg.svd(X, full_matrices=False)
    Worig = Vhorig.T
    X = Uorig @ np.diag(Sorig)
    X = X[:, :rk]
    Worig = Worig[:, :rk]
    X0 = X.copy()

    r, c = X.shape
    maxcount = 10000
    convFlag: list[list[str]] = [[""] * p for _ in range(guess)]
    kurtObj = np.zeros((guess, p))
    T = np.zeros((r, p))
    W_mat = np.zeros((c, p))
    P_mat = np.zeros((c, p))
    ALPH = np.zeros(p)

    for j in range(p):
        cc = c - j
        convlimit = 1e-10 * cc

        wall = np.zeros((cc, guess))
        alphall = np.zeros(guess)

        _, _, Vjh = np.linalg.svd(X, full_matrices=False)
        Vj = Vjh.T[:, :cc]
        X = X @ Vj

        for k in range(guess):
            w = np.random.randn(cc)
            w /= np.linalg.norm(w)
            alph = float((X @ w).mean())
            oldw1 = w.copy()
            oldw2 = oldw1.copy()
            count = 0

            while True:
                count += 1
                x = X @ w
                xalph = x - alph
                alph = alph + np.sum(xalph ** 3) / (3.0 * np.sum(xalph ** 2))
                mu = alph * w

                tem = (x - alph) ** 2
                dalph_dv = (X.T @ tem) / np.sum(tem)
                tem1 = X.T - np.outer(dalph_dv, np.ones(r))  # cc × r
                tem2 = X - np.outer(np.ones(r), mu)           # r × cc

                Mat1 = ((np.ones((cc, 1)) * (tem2 @ w)) * tem1) @ tem2
                Mat2 = tem1 @ tem2
                w = np.linalg.solve(Mat1, Mat2 @ w)

                w /= np.linalg.norm(w)
                L1 = (w @ oldw1) ** 2

                if (1 - L1) < convlimit:
                    convFlag[k][j] = "Converged"
                    break
                if count > maxcount:
                    convFlag[k][j] = "Not converged"
                    break

                L2 = (w @ oldw2) ** 2
                if L2 > L1 and L2 > 0.95:
                    w = w + (np.random.rand() / 5 + 0.8) * oldw1
                    w /= np.linalg.norm(w)
                oldw2 = oldw1.copy()
                oldw1 = w.copy()

            wall[:, k] = w
            alphall[k] = alph

        # Recentered kurtosis
        proj = X @ wall - np.outer(np.ones(r), alphall)
        kurtObj[:, j] = (
            r * np.sum(proj ** 4, axis=0) /
            (np.sum(proj ** 2, axis=0) ** 2)
        )

        ind = int(np.argmin(kurtObj[:, j]))
        Wj = wall[:, ind]

        # Sign convention: match first nonzero element to positive
        for i in range(cc):
            if Wj[i] != 0:
                signum = np.sign(Wj[i])
                break
        else:
            signum = 1.0

        Wj *= signum
        ALPH[j] = alphall[ind] * signum

        if vsorth == "VO":
            t = X @ Wj
            T[:, j] = t
            W_mat[:, j] = Vj @ Wj
            X = X0 - X0 @ W_mat @ W_mat.T
        else:  # "SO"
            t = X @ Wj
            T[:, j] = t
            W_mat[:, j] = Vj @ Wj
            Pj = X.T @ t / (t @ t)
            P_mat[:, j] = Vj @ Pj
            X = X0 - T @ P_mat.T

    # Transform back to original space
    W_mat = Worig @ W_mat
    if vsorth == "VO":
        V = W_mat
        W_out = None
        P_out = None
        T = T + (Morig @ V)
        R = ALPH @ V.T + Morig
    else:
        P_mat = Worig @ P_mat
        V = W_mat @ np.linalg.inv(P_mat.T @ W_mat)
        T = T + (Morig @ V)
        R = ALPH @ (P_mat.T @ W_mat) @ W_mat.T + Morig
        norms = np.sqrt(np.sum(V ** 2, axis=0))
        V = V / norms
        T = T / norms
        P_mat = P_mat * norms
        W_out = W_mat
        P_out = P_mat

    return T, V, R, W_out, P_out, kurtObj, convFlag


# ---------------------------------------------------------------------------
# Recentered multivariate kurtosis PP  (port of rcmulkurtpp)
# ---------------------------------------------------------------------------

def _rcmulkurtpp(
    X: np.ndarray,
    p: int = 2,
    guess: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list, np.ndarray, list]:
    """
    Quasi-power method to minimise recentered multivariate kurtosis.

    Returns
    -------
    T, V, R, K, Vall, kurtObj, convFlag
    """
    n, m = X.shape
    Morig = X.mean(axis=0)
    X = X - Morig

    maxcount = 10000
    convlimit = 1e-10

    Vall: list = [None] * guess
    rall: list = [None] * guess
    kurtObj = np.zeros(guess)
    convFlag: list = [""] * guess

    for i in range(guess):
        count = 0
        V = np.random.randn(m, p)
        V = orthbasis(V)
        oldV1 = V.copy()
        R = X.mean(axis=0).copy()  # start as column mean (row vector)

        while True:
            count += 1

            # Update R
            Y = (X - np.outer(np.ones(n), R / p)) @ V   # n × p
            invPsi = np.linalg.inv(Y.T @ Y)
            gj = np.diag(Y @ invPsi @ Y.T)              # length n
            Yj = Y @ invPsi @ Y.sum(axis=0)             # length n
            J = (2 * Y.T @ ((np.outer(Yj, np.ones(p)) * Y) @ invPsi)
                 - np.eye(p) * (gj.sum() + 2)) / p
            f = np.sum(Y.T * (np.ones((p, 1)) * gj), axis=1)
            R = R - V @ np.linalg.solve(J, f)

            # Update V
            XX = X - np.outer(np.ones(n), R)            # note: no /p here
            Z = XX @ V                                   # n × p
            S = Z.T @ Z
            invS = np.linalg.inv(S)
            ai = np.diag(Z @ invS @ Z.T)                # length n
            Z_ai = (np.outer(ai, np.ones(p)) * Z)       # n × p
            Si_ai = Z.T @ Z_ai                          # p × p

            b1 = -np.linalg.solve(J.T, invS @ Si_ai @ invS @ Z.sum(axis=0))
            b2 = -np.linalg.solve(J.T, invS @ Z_ai.sum(axis=0))

            Yj_b1_Yj = (np.outer(Y @ b1, np.ones(p))) * Y  # n × p
            Yj_b2_Yj = (np.outer(Y @ b2, np.ones(p))) * Y
            Xj_gj = (gj[:, None] * X).sum(axis=0)           # length m

            M1 = X.T @ Z @ invS @ Si_ai
            M2 = -np.outer(Xj_gj, b1) @ S
            M3 = 2 * X.T @ Y @ (invPsi @ Y.T @ Yj_b1_Yj @ invPsi @ S)
            M4 = -2 * X.T @ Yj_b1_Yj @ invPsi @ S

            M5 = (X.T * ai) @ XX                          # m × m  (full rank)
            M6 = -np.outer(Xj_gj, b2) @ (Z.T @ XX)
            M7 = 2 * X.T @ Y @ (invPsi @ Y.T @ Yj_b2_Yj @ invPsi @ (Z.T @ XX))
            M8 = -2 * X.T @ Yj_b2_Yj @ invPsi @ (Z.T @ XX)

            V = np.linalg.solve(M5 + M6 + M7 + M8, M1 + M2 + M3 + M4)
            V = orthbasis(V)

            L = np.abs(V) - np.abs(oldV1)
            if np.trace(L.T @ L) < convlimit * p:
                convFlag[i] = "Converged"
                break
            if count > maxcount:
                convFlag[i] = "Not converged"
                break
            oldV1 = V.copy()

        kurtObj[i] = n * np.sum(np.diag(Z @ np.linalg.inv(Z.T @ Z) @ Z.T) ** 2)

        U_t, _, Vt_h = np.linalg.svd(X @ V, full_matrices=False)
        Vtem = V @ Vt_h.T
        Vall[i] = Vtem
        rall[i] = R @ Vtem @ Vtem.T  # saved as row vector

    ind = int(np.argmin(kurtObj))
    V = Vall[ind]
    R_out = rall[ind]
    T = X @ V
    K = float(kurtObj[ind])

    # Add mean
    T = T + Morig @ V
    R_out = R_out + Morig

    return T, V, R_out, K, Vall, kurtObj, convFlag


# ---------------------------------------------------------------------------
# Public dispatcher: projpursuit
# ---------------------------------------------------------------------------

def projpursuit(
    X: np.ndarray,
    p: int = 2,
    guess: int = 100,
    meth: str = "uni",
    opt: str = "ord",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Dispatch projection pursuit analysis to the appropriate algorithm.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_vars)
        Mean-centered (or raw) data matrix.
    p : int
        Number of projection dimensions to extract. Default 2.
    guess : int
        Number of random initial guesses for the optimisation. Default 100.
    meth : {'uni', 'mul'}
        'uni' — stepwise univariate kurtosis (default).
        'mul' — simultaneous multivariate kurtosis.
    opt : {'ord', 'rec'}
        'ord' — ordinary kurtosis (default).
        'rec' — recentered kurtosis (for unbalanced classes).

    Returns
    -------
    T : ndarray, shape (n_samples, p)
        Projection pursuit scores.
    V : ndarray, shape (n_vars, p)
        Projection vectors.
    ppout : dict with keys
        K         – scalar (or array of length p) best kurtosis value(s).
        kurtObj   – kurtosis values for all guesses.
        convFlag  – convergence status per guess.
        W         – W matrix (univariate SO only), else None.
        P         – P matrix (univariate SO only), else None.
        Mu        – recentered R vector (recentered methods only), else None.
    """
    meth = meth.lower()
    opt = opt.lower()

    ppout: dict = {"K": None, "kurtObj": None, "convFlag": None,
                   "W": None, "P": None, "Mu": None}

    if meth == "mul":
        if opt == "rec":
            T, V, R, K, Vall, kurtObj, convFlag = _rcmulkurtpp(X, p, guess)
            ppout["K"] = K
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["Mu"] = R
        else:  # ord
            T, V, Vall, kurtObj, convFlag = _mulkurtpp(X, p, guess)
            ppout["K"] = float(np.min(kurtObj))
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
    else:  # uni
        if opt == "rec":
            T, V, R, W, P, kurtObj, convFlag = _rckurtpp(X, p, guess)
            ppout["K"] = float(np.min(kurtObj))
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["W"] = W
            ppout["P"] = P
            ppout["Mu"] = R
        else:  # ord
            T, V, W, P, kurtObj, convFlag = _okurtpp(X, p, guess)
            ppout["K"] = float(np.min(kurtObj))
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["W"] = W
            ppout["P"] = P

    return T, V, ppout
