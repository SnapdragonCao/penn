MODULE = 'penn'

# Configuration name
CONFIG = 'swipe'

# The decoder to use for postprocessing
DECODER = 'argmax'


# Distance between adjacent frames
# HOPSIZE = 160  # samples
HOPSIZE = 80  # samples
# The pitch estimation method to use
METHOD = 'swipe'

# Audio sample rate
SAMPLE_RATE = 16000  # hz
