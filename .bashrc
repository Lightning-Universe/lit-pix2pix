# Set the encoding for SSH since ssh can't inherit the ENV
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Set HOME
export HOME="/teamspace/studios/this_studio"

# >>> lightning managed. do not modify >>>
[ -f /settings/.lightningrc ] && source /settings/.lightningrc bash
# <<< lightning managed. do not modify <<<
