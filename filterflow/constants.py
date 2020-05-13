import math

MIN_RELATIVE_LOG_WEIGHT = -4.
# for example if using 10 particles, we consider those with weight exp(-4*ln(10)) = 1e-4 to have died out
MIN_ABSOLUTE_LOG_WEIGHT = -13.8  # approx. -6 ln(10)

MIN_RELATIVE_WEIGHT = math.exp(MIN_RELATIVE_LOG_WEIGHT)
MIN_ABSOLUTE_WEIGHT = math.exp(MIN_ABSOLUTE_LOG_WEIGHT)
