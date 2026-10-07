"""rul_pipeline: Version 0 probabilistic NDT -> remaining-useful-life chain.

Stages (each a sub-package, each a pure-function module):

    io.bristolfe            forward library  {a_i, F(a_i)}           (BristolFE export)
    synthetic               toy forward library + noisy measurement y (for testing)
    inversion.grid_bayes    y            -> p(a | y)                  (1-D grid Bayes)
    structural.mode_i       a^(i)        -> K_I^(i) = Y sigma sqrt(pi a)
    crack_growth.paris      a^(i)        -> N_f^(i)                   (Paris law)
    prognosis.monte_carlo   {a^(i)}      -> p(RUL | y)                (sampling)
    inspection.scheduling   {a^(i)}      -> next inspection interval  (risk <= alpha)
    update.bayes_update     p(a_k | y_1:k) -> p(a_k+1 | y_1:k+1)      (scaffold)

Units are SI throughout: metres, pascals, Pa*sqrt(m), seconds, cycles.
"""

__version__ = "0.1.0"
