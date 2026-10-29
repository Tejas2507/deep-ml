import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    denom = sum(exp_scores)
    prob = [round(exp_score/denom,4) for exp_score in exp_scores]
    return prob
