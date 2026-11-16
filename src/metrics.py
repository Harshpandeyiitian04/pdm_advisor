# This file measures how wrong the life-left guesses are. For both scores, lower means a better guess.

# This brings in the number tools.
import numpy as np


# This score is the typical size of the mistake, in cycles. Zero means a perfect guess.
def rmse(y_true, y_pred):
    # Turn the real answers and the guesses into lists of numbers.
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    # Square each mistake, average them, then take the square root so the answer is back in cycles.
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# This is the score NASA uses for this problem.
# Guessing that an engine has more life than it really does is punished harder than guessing too little,
# because saying "it is fine" when it is about to fail is the dangerous mistake.
def nasa_score(y_true, y_pred):
    # The mistake is the guess minus the real life left. A positive number means we were too optimistic.
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    # Add up a bigger penalty when the guess is too high, and a smaller penalty when it is too low.
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))
