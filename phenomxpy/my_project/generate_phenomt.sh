#!/bin/bash -l

# Memory management
# ulimit -v $((8 * 1024 * 1024))  # Hard limit at 8GB

# Clean previous runs
rm -f python_output.log *.png 2>/dev/null

# Get job name from argument
JOB_NAME="$1"

# Make folder for job files
mkdir ${JOB_NAME}

# Environment setup
export TERM=dumb
export MPLBACKEND='Agg'
export PYTHONUNBUFFERED=1

# Activate conda
source /etc/profile.d/igwn-conda.sh 2>/dev/null
source ~/.bashrc

conda activate phenomxpy_env || {
    echo "ERROR: Conda activation failed" > ${JOB_NAME}/conda_error.log
    exit 1
}

# Run with all output captured
python /home/suzanne.lexmond/Surrogate_model/Surrogate_model_repo/SurrogateModel_3PN_eccentric/phenomxpy/phenomxpy/my_project/test_phenomxpy.py > python_output.log 2>&1

# Move output file to job directory specified in .sub file
mv python_output.log ${JOB_NAME}

exit $?