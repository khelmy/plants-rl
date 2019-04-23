import numpy as np
import matplotlib.pyplot as plt
from bandit import *
from dist import *

mu = np.arange(1, 2.25, 0.01)
sigma = np.repeat(0.5, mu.size)

dist_n = NormalDist(mu, sigma)
ucb_n = UCB1Bandit(dist_n)
exp_n = EXP3Bandit(dist_n)
b_n = BenchmarkBandit(dist_n)

dist_u = UnifDist(mu, sigma)
ucb_u = UCB1Bandit(dist_u)
exp_u = EXP3Bandit(dist_u)
b_u = BenchmarkBandit(dist_u)

#for nrounds in (1500, 15000, 150000):
nrounds = 1500

def sim_all(warm_start = True):
    for ucb, exp, b, dname in ((ucb_n, exp_n, b_n, "Normal"),
            (ucb_u, exp_u, b_u, "Uniform")):
        # Threshold of 2.2 gets us top 5 observations, 2.0 gets us top 25
        for thresh, tname in ((2.2, "5"), (2.0, "25")):
            np.random.seed(6)
            xbar_hat, plays, losses = ucb.ucb1(nrounds, thresh, mae, mu,
                warm_start)
            e_hat, e_plays, e_losses = exp.exp3(nrounds, thresh, mae, mu, 0.5,
                warm_start)
            b_hat, b_plays, b_losses = b.random_benchmark(nrounds, thresh, mae, mu,
                warm_start)
            plt.plot(np.arange(1, losses.size + 1), losses)
            plt.plot(np.arange(1, e_losses.size + 1), e_losses)
            plt.plot(np.arange(1, b_losses.size + 1), b_losses)

            plt.title("MAE Loss versus Round for Top {tname} Observations ({dname})"
                .format(tname = tname, dname = dname))
            plt.legend(["UCB1", "EXP3 (0.5)", "Benchmark"])
            plt.show()

def sim_exp3(gammas = (0.8, 0.2)):
    for exp, b, dname in ((exp_n, b_n, "Normal"),
            (exp_u, b_u, "Uniform")):
        # Threshold of 2.2 gets us top 5 observations, 2.0 gets us top 25
        for thresh, tname in ((2.2, "5"), (2.0, "25")):
            np.random.seed(6)
            for gamma in gammas:
                e_losses = (exp.exp3(nrounds, thresh, mae, mu, gamma))[2]
                plt.plot(np.arange(1, e_losses.size + 1), e_losses)
            b_hat, b_plays, b_losses = b.random_benchmark(nrounds, thresh, mae, mu)
            plt.plot(np.arange(1, b_losses.size + 1), b_losses)

            plt.title("EXP3 MAE Loss versus Round for Top {tname} Observations ({dname})"
                .format(tname = tname, dname = dname))
            plt.legend(["gamma = 0.8", "gamma = 0.2", "Benchmark"])
            plt.show()

sim_all(False)
# sim_exp3()
