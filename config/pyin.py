MODULE = 'penn'

# Configuration name
CONFIG = 'pyin'

# The decoder to use for postprocessing
# DECODER = 'argmax'
DECODER = 'viterbi'

# Distance between adjacent frames
# HOPSIZE = 160  # samples
HOPSIZE = 80  # samples
# The pitch estimation method to use
METHOD = 'pyin'

# Audio sample rate
SAMPLE_RATE = 16000  # hz
