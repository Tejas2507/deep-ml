import math

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:

    value = x
    derivative = 1.0

    for f in reversed(functions):

        if f == 'square':
            derivative *= 2 * value
            value = value ** 2

        elif f == 'sin':
            derivative *= math.cos(value)
            value = math.sin(value)

        elif f == 'exp':
            value = math.exp(value)
            derivative *= value

        elif f == 'log':
            derivative *= 1 / value
            value = math.log(value)

        else:
            raise ValueError(f"Unknown function: {f}")

    return derivative