from distance import *
from dist import *
import numpy as np

class UCB1Bandit:
    def __init__(self, dist):
        self.dist = dist
        self.D = build_distance_matrix(self.dist.shape, self.dist.size)
        self.position = 0
        self.C = 500
        self.sigma_c = 0
        self.rewards = np.empty((0,))
        self.played = np.empty((0,))

    def draw(self):
        return self.dist.draw()

    def draw_i(self, i):
        return self.dist.draw_elem(i)

    # Take initial sample
    def initial_sample(self, warm_start):
        plays = np.ones(self.dist.size)
        if warm_start:
            xbar_hat = self.draw()
        else:
            xbar_hat = np.ones(self.dist.size) + 0.625
            xbar_hat = xbar_hat.reshape(int(np.sqrt(xbar_hat.shape[0])), int(np.sqrt(xbar_hat.shape[0])))
            plays /= 100
        return (xbar_hat, plays)

    # Update the mean with a new observation
    def adjust_mean(self, xbar_hat, pick_i, plays, draw_pick):
        return ((xbar_hat[pick_i % xbar_hat.shape[0]][pick_i // xbar_hat.shape[0]] * plays[pick_i] + draw_pick) /
            (plays[pick_i] + 1))

    # Based on specification from
    # https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/
    def ucb1(self, warm_start = True):
        (xbar_hat, plays) = self.initial_sample(warm_start)
        round = 1
        while self.sigma_c <= self.C:
            (xbar_hat, plays) = self.ucb1_update(round, xbar_hat, plays)
            round += 1
        top_ten_ests = np.flip(np.argsort(xbar_hat, None))[0:10]
        top_ten_true = np.flip(np.argsort(self.dist.k, None))[0:10]
        self.acc = np.mean(np.isin(top_ten_ests, top_ten_true))
        return (xbar_hat, plays)

    def ucb1_update(self, round, xbar_hat, plays):
        payoffs_hat = xbar_hat.flatten() + np.sqrt(2 * np.log(round) / plays)
        pick_i = np.argmax(payoffs_hat.flatten() / self.D[self.position])
        self.sigma_c += self.D[self.position][pick_i]
        self.position = pick_i
        draw_pick = self.draw_i(pick_i)
        xbar_hat[pick_i % xbar_hat.shape[0]][pick_i // xbar_hat.shape[0]] = self.adjust_mean(xbar_hat, pick_i, plays, draw_pick)
        plays[pick_i] += 1
        self.rewards = np.append(self.rewards, draw_pick)
        self.played = np.append(self.played, pick_i)
        return (xbar_hat, plays)

k = np.linspace(1.25, 2.25, num = 100).reshape((10,10))
np.random.shuffle(k)
theta = np.tile(0.125, k.shape)
gamma = GammaDist(k, theta)
ucb1 = UCB1Bandit(gamma)
ucb1.ucb1()
ucb1_naive = UCB1Bandit(gamma)
ucb1_naive.ucb1(False)
print(np.sum(ucb1.rewards))
print(ucb1.acc)
print(np.sum(ucb1_naive.rewards))
print(ucb1_naive.acc)
