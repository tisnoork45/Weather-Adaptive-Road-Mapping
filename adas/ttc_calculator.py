def calculate_ttc(depth, velocity):
    if velocity <= 0:
        return float("inf")
    return depth / velocity