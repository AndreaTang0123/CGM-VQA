#!/bin/bash
#SBATCH --job-name=cgm_plots
#SBATCH --output=logs/plots_%j.log
#SBATCH --error=logs/plots_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -e
unset PYTHONPATH

# Change to project root
cd /nas/longleaf/home/txueying/CGM-VQA

# Use a temporary virtual environment for plotting to avoid polluting main venvs
VENV_PLOT="/work/users/t/x/txueying/venv_plotting"
if [ ! -d "$VENV_PLOT" ]; then
  python3 -m venv "$VENV_PLOT"
fi
source "$VENV_PLOT/bin/activate"

# Install plotting dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install matplotlib numpy -q

# Run the plotting script
echo "Generating plots..."
python statistics/plot_charts.py

echo "Plots generated successfully."
ls -l statistics/*.png
