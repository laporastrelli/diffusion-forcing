#!/usr/bin/env bash

echo "---- inside train.sh ----"
which python
python -c "import sys; print(sys.executable); print('\n'.join(sys.path))" || true
echo "-------------------------"

# … rest of your script …
