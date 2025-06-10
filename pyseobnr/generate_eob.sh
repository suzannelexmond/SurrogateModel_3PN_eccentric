#!/bin/bash -l

# Set dummy TERM variable to suppress warnings
export TERM=dumb

# Fix for the unary operator error by properly formatting the condition
if [ -n "$BASH_VERSION" ]; then
    source /etc/profile.d/igwn-conda.sh
else
    source /etc/profile.d/igwn-conda.sh >/dev/null 2>&1
fi

# Source bashrc (needed for conda initialization)
source ~/.bashrc

# Activate conda environment
conda activate igwn || {
    echo "ERROR: Failed to activate igwn environment"
    exit 1
}

# Run your Python script
# python /home/suzanne.lexmond/Surrogate_model/Surrogate_model_repo/SurrogateModel_3PN_eccentric/pyseobnr/test.py > python_output.log 2>&1
python /home/suzanne.lexmond/Surrogate_model/Surrogate_model_repo/SurrogateModel_3PN_eccentric/pyseobnr/generate_eccentic_eob.py > python_output.log 2>&1

echo 'job done'
exit 0