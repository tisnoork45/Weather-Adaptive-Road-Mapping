def assess_risk(depth, ttc):
    if depth < 4 or ttc < 2:
        return "HIGH"
    elif depth < 10 or ttc < 5:
        return "MEDIUM"
    else:
        return "LOW"