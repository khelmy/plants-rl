import numpy as np
from dist import *

class Bandit:
    def __init__(self, dist):
        self.dist = dist

    def draw(self):
        return self.dist.draw()

    def draw_i(self, i):
        return self.dist.draw_i(i)

    # Take initial sample
    def initial_sample(self, nrounds, warm_start):
        plays = np.ones(self.dist.size)
        if warm_start:
            xbar_hat = self.draw()
        else:
            xbar_hat = np.ones(self.dist.size) + 0.625
            plays /= 100
        losses = np.zeros(nrounds)
        return (xbar_hat, plays, losses)

    # Update the mean with a new observation
    def adjust_mean(self, xbar_hat, pick_i, plays, draw_pick):
        return ((xbar_hat[pick_i] * plays[pick_i] + draw_pick) /
            (plays[pick_i] + 1))

    # Generic bandit process framework
    def bandit_process(self, nrounds, threshold, loss_metric, ref,
            update_f, warm_start = True):
        (xbar_hat, plays, losses) = self.initial_sample(nrounds, warm_start)
        for round in range(nrounds):
            (xbar_hat, plays, losses) = update_f(threshold,
                loss_metric, ref, round, xbar_hat, plays, losses)
        return (xbar_hat, plays, losses)

    # Generic update process
    def update_core(self, threshold, loss_metric, ref, round,
            xbar_hat, plays, losses, pick_i):
        draw_pick = self.draw_i(pick_i)
        xbar_hat[pick_i] = self.adjust_mean(xbar_hat, pick_i, plays, draw_pick)
        plays[pick_i] += 1
        losses[round] = loss_metric(xbar_hat, threshold, ref)
        return (xbar_hat, plays, losses)


class UCB1Bandit(Bandit):
    # Based on specification from
    # https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/

    def ucb1(self, nrounds, threshold, loss_metric, ref, warm_start = True):
        return self.bandit_process(nrounds, threshold, loss_metric, ref,
                self.ucb1_update, warm_start)

    def ucb1_update(self, threshold, loss_metric, ref, round,
            xbar_hat, plays, losses):
        payoffs_hat = xbar_hat + np.sqrt(2 * np.log(round) / plays)
        pick_i = np.argmax(payoffs_hat)
        return self.update_core(threshold, loss_metric, ref, round,
            xbar_hat, plays, losses, pick_i)

class EXP3Bandit(Bandit):
    # Based on specification from
    # https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
    def exp3(self, nrounds, threshold, loss_metric, ref,
            gamma, warm_start = True):
        self.weights = np.ones(self.dist.size)
        self.gamma = gamma
        return self.bandit_process(nrounds, threshold, loss_metric, ref,
                self.exp3_update, warm_start)

    def exp3_update(self, threshold, loss_metric, ref, round,
            xbar_hat, plays, losses):
        payoffs_hat = ((1 - self.gamma) * (self.weights / np.sum(self.weights)) +
            (self.gamma / xbar_hat.size))
        pick_i = np.random.choice(self.dist.size, p = payoffs_hat)
        (xbar_hat, plays, losses) = self.update_core(threshold, loss_metric, ref, round,
            xbar_hat, plays, losses, pick_i)
        self.weights[pick_i] = (self.weights[pick_i] *
            np.exp(self.gamma * xbar_hat[pick_i] / xbar_hat.size))
        return (xbar_hat, plays, losses)

class BenchmarkBandit(Bandit):
    def ordered_benchmark(self, nrounds, threshold, loss_metric, ref,
            warm_start = True):
        return self.bandit_process(nrounds, threshold, loss_metric, ref,
                self.ordered_update, warm_start)

    def ordered_update(self, threshold, loss_metric, ref, round,
            xbar_hat, plays, losses):
        pick_i = round % self.dist.size
        return self.update_core(threshold, loss_metric, ref, round,
            xbar_hat, plays, losses, pick_i)

    def random_benchmark(self, nrounds, threshold, loss_metric, ref,
            warm_start = True):
        return self.bandit_process(nrounds, threshold, loss_metric, ref,
                self.random_update, warm_start)

    def random_update(self, threshold, loss_metric, ref, round,
            xbar_hat, plays, losses):
        pick_i = np.random.randint(xbar_hat.size)
        return self.update_core(threshold, loss_metric, ref, round,
            xbar_hat, plays, losses, pick_i)

class StaticBandit(Bandit):
    def adaptive_bidding():
        return


def mae(xbar_hat, threshold, ref):
    return np.mean(np.abs(xbar_hat[ref > threshold] - ref[ref > threshold]))

def mse(xbar_hat, threshold, ref):
    return np.mean(np.square(xbar_hat[ref > threshold] - ref[ref > threshold]))
