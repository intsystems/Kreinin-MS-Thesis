# Validate all regression models across GPUs.
set -e
cd "$(dirname "$0")"

# Round 1: 3 models in parallel
python validate_regression_models.py --gpu 1 --checkpoint run_ancient-brook-1_model_best.pth &
python validate_regression_models.py --gpu 2 --checkpoint run_sandy-moon-3_model_best.pth &
python validate_regression_models.py --gpu 3 --checkpoint run_grievous-senate-4_model_best.pth &
wait

# Round 2: UMambaBot
python validate_regression_models.py --gpu 1 --checkpoint run_ancient-bantha-5_model_best.pth
wait

# Aggregate results into summary.json
python validate_regression_models.py --aggregate
