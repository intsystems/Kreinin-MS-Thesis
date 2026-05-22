# Validate all flow matching models across GPUs.
# Each model runs on a separate GPU for parallelism.
set -e
cd "$(dirname "$0")"

# Round 1: 4 models in parallel
python validate_flow_models.py --gpu 0 --checkpoint run_lemon-shape-1_model_best.pth &
python validate_flow_models.py --gpu 1 --checkpoint run_vivid-lion-3_model_best.pth &
python validate_flow_models.py --gpu 2 --checkpoint run_royal-breeze-5_model_best.pth &
python validate_flow_models.py --gpu 3 --checkpoint run_misunderstood-cloud-7_model_best.pth &
wait

python validate_flow_models.py --gpu 0 --checkpoint run_polar-lion-8_model_best.pth &
python validate_flow_models.py --gpu 1 --checkpoint run_still-sky-6_model_best.pth &
wait

python validate_flow_models.py --aggregate
