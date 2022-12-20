from collections import defaultdict

from tqdm import tqdm
import numpy as np

from algorithms import metrics
from algorithms.model_selection import DEFAULT_METRICS


def rbf(X, Y=None, gamma=1):
    X_norm = np.sum(X ** 2, axis=-1)
    if Y is None:
        Y = X
        Y_norm = X_norm
    else:
        Y_norm = np.sum(Y ** 2, axis=-1)
    return np.exp(-gamma * (X_norm[:, np.newaxis] + Y_norm[np.newaxis, :] - 2 * X @ Y.T))


class SupportVectorRegressor:
    def __init__(self,
                 X,
                 y,
                 C,
                 epsilon,
                 tolerance,
                 kernel_type,
                 gamma):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        self.C = C
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.kernel = None

        self.b = 0
        self.W = np.zeros(X.shape[1])
        self.alpha = np.zeros(X.shape[0])

        self.calculate_kernel()

    def calculate_kernel(self):
        if self.kernel_type == 'linear':
            self.kernel = self.X @ self.X.T
        elif self.kernel_type == 'rbf':
            self.kernel = rbf(X=self.X, gamma=self.gamma)
        else:
            raise NotImplementedError('Only linear and rbf kernels are implemented.')

    def calculate_error(self, i):
        return self.b + np.expand_dims(self.alpha, axis=0) @ np.expand_dims(self.kernel[i], axis=1) - self.y[i]

    def check_KKT_condition_violations(self, i):
        Ai = self.alpha[i]
        Ei = self.calculate_error(i)

        violates = Ai == 0 and not (-self.epsilon <= Ei + self.tolerance and Ei <= self.epsilon + self.tolerance)
        violates = violates or ((-self.C < Ai < 0) and Ei != self.epsilon)
        violates = violates or (0 < Ai < self.C and Ei != -self.epsilon)
        violates = violates or (Ai == -self.C and not Ei >= self.epsilon - self.tolerance)
        violates = violates or (Ai == self.C and not Ei <= self.epsilon - self.tolerance)
        return violates

    def fit(self, X_valid, y_valid, max_iterations):
        history = defaultdict(list)

        for _ in (progress_bar := tqdm(range(max_iterations))):
            n_changes = 0
            for i in range(len(self.X)):
                if self.check_KKT_condition_violations(i):
                    possible_j = np.setdiff1d(np.arange(len(self.X)), [i])
                    j = np.random.choice(possible_j, size=1).item()
                    n_changes += self.update_alpha(i, j)

            train_avg_losses = {metric: np.mean(fn(self.y, self.predict(self.X)))
                                for metric, fn in DEFAULT_METRICS.items()}
            valid_avg_losses = {metric: np.mean(fn(y_valid, self.predict(X_valid)))
                                for metric, fn in DEFAULT_METRICS.items()}

            for metric in DEFAULT_METRICS.keys():
                history[f'train_{metric}'].append(train_avg_losses[metric])
                history[f'valid_{metric}'].append(valid_avg_losses[metric])

            progress_bar.set_description_str(f'C={self.C}, epsilon={self.epsilon}, tolerance={self.tolerance}')
            progress_bar.set_postfix_str(f'train_mse={train_avg_losses["MSE"]:.7f}, '
                                         f'valid_mse={valid_avg_losses["MSE"]:.7f}, '
                                         f'number of changes: {n_changes}')
            if n_changes == 0:
                break

        return history

    def predict(self, X):
        if self.kernel_type == 'linear':
            return self.b + X @ self.W.reshape(-1, 1)
        elif self.kernel_type == 'rbf':
            return (self.alpha @ rbf(self.X, X, self.gamma)).reshape(-1, 1) + self.b
        else:
            raise NotImplementedError('Only linear and rbf kernels are implemented.')

    def update_alpha(self, i, j):
        Ei = self.calculate_error(i)
        Ej = self.calculate_error(j)

        L = np.max([-self.C, self.alpha[i] + self.alpha[j] - self.C])
        H = np.min([self.C, self.alpha[i] + self.alpha[j] + self.C])

        eta = self.kernel[i][i] - 2 * self.kernel[i][j] + self.kernel[j][j]

        if L == H or eta <= 0:
            return False

        Aj_updated_positive = (Ei - Ej + 2 * self.epsilon) / eta + self.alpha[j]
        Aj_updated_zero = (Ei - Ej) / eta + self.alpha[j]
        Aj_updated_negative = (Ei - Ej - 2 * self.epsilon) / eta + self.alpha[j]

        if self.alpha[i] + self.alpha[j] <= -self.C:
            Aj_updated = Aj_updated_zero

        elif -self.C < self.alpha[i] + self.alpha[j] < 0:
            if Aj_updated_positive < self.alpha[i] + self.alpha[j]:
                Aj_updated = Aj_updated_positive
            elif Aj_updated_zero <= self.alpha[i] + self.alpha[j]:
                Aj_updated = self.alpha[i] + self.alpha[j]
            elif self.alpha[i] + self.alpha[j] < Aj_updated_zero < 0:
                Aj_updated = Aj_updated_zero
            elif 0 < Aj_updated_negative:
                Aj_updated = Aj_updated_negative
            else:
                Aj_updated = 0

        elif self.alpha[i] + self.alpha[j] == 0:
            if Aj_updated_positive <= L:
                Aj_updated = L
            elif L < Aj_updated_positive < 0:
                Aj_updated = Aj_updated_positive
            elif 0 < Aj_updated_negative:
                Aj_updated = Aj_updated_negative
            else:
                Aj_updated = 0

        elif 0 < self.alpha[i] + self.alpha[j] < self.C:
            if Aj_updated_positive < 0:
                Aj_updated = Aj_updated_positive
            elif Aj_updated_zero <= 0:
                Aj_updated = 0
            elif 0 < Aj_updated_zero < self.alpha[i] + self.alpha[j]:
                Aj_updated = Aj_updated_zero
            elif self.alpha[i] + self.alpha[j] < Aj_updated_negative:
                Aj_updated = Aj_updated_negative
            else:
                Aj_updated = self.alpha[i] + self.alpha[j]

        else:
            Aj_updated = Aj_updated_zero

        Aj_updated = np.clip(Aj_updated, L, H)
        Ai_updated = self.alpha[i] + self.alpha[j] - Aj_updated

        dAi = Ai_updated - self.alpha[i]
        dAj = Aj_updated - self.alpha[j]

        if self.kernel_type == 'linear':
            self.W += np.ravel(dAi * self.X[i] + dAj * self.X[j])

        bi = self.b - dAi * self.kernel[i][i] - dAj * self.kernel[i][j] - Ei
        bj = self.b - dAi * self.kernel[i][j] - dAj * self.kernel[j][j] - Ej

        self.b = (bi + bj) / 2
        if 0 < Ai_updated < self.C:
            self.b = bi
        if 0 < Aj_updated < self.C:
            self.b = bj

        self.alpha[i] = Ai_updated
        self.alpha[j] = Aj_updated
        return True
