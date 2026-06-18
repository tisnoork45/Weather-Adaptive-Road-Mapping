import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adas.relative_velocity import calculate_relative_velocity
from adas.ttc_calculator import calculate_ttc
from adas.risk_engine import assess_risk

prev_depth = 12
curr_depth = 10
fps = 30

velocity = calculate_relative_velocity(prev_depth, curr_depth, fps)
ttc = calculate_ttc(curr_depth, velocity)
risk = assess_risk(curr_depth, ttc)

print("Velocity:", velocity)
print("TTC:", ttc)
print("Risk:", risk)