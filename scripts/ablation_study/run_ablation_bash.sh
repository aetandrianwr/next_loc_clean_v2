#!/bin/bash
# =============================================================================
# Ablation Study Runner - Bash Script
# =============================================================================
# This script runs the comprehensive ablation study for Pointer Network V45.
# It manages 3 parallel training sessions with proper staggering.
#
# Usage:
#   ./scripts/ablation_study/run_ablation_bash.sh [dataset]
#   dataset: geolife, diy, or all (default: all)
# =============================================================================

set -e

# Configuration
CONDA_ENV="mlenv"
MAX_PARALLEL=3
DELAY_BETWEEN_JOBS=5
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Ablation experiments
ABLATIONS=(
    "full_model"
    "no_user_emb"
    "no_time_emb"
    "no_weekday_emb"
    "no_recency_emb"
    "no_duration_emb"
    "no_pos_from_end"
    "no_sinusoidal_pos"
    "no_temporal"
    "no_pointer"
    "no_generation"
    "no_gate"
    "single_layer"
)

# Parse arguments
DATASET="${1:-all}"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$PROJECT_ROOT"

echo "============================================================"
echo "ABLATION STUDY FOR POINTER NETWORK V45"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Dataset: $DATASET"
echo "Max Parallel Jobs: $MAX_PARALLEL"
echo "Number of Ablations: ${#ABLATIONS[@]}"
echo "============================================================"

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/experiments/ablation_study"
mkdir -p "$OUTPUT_DIR/logs"

# Function to run single ablation
run_ablation() {
    local config=$1
    local ablation=$2
    local log_file=$3
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $ablation"
    python scripts/ablation_study/train_ablation.py \
        --config "$config" \
        --ablation "$ablation" \
        > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $ablation"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $ablation"
    fi
}

# Function to run ablations for a dataset
run_dataset_ablations() {
    local dataset=$1
    local config=$2
    
    echo ""
    echo "============================================================"
    echo "Running ablations for $dataset"
    echo "Config: $config"
    echo "============================================================"
    
    # Track background PIDs
    local pids=()
    local job_count=0
    
    for ablation in "${ABLATIONS[@]}"; do
        log_file="$OUTPUT_DIR/logs/${dataset}_${ablation}.log"
        
        # Run in background
        run_ablation "$config" "$ablation" "$log_file" &
        pids+=($!)
        job_count=$((job_count + 1))
        
        # Stagger jobs
        sleep $DELAY_BETWEEN_JOBS
        
        # If we've reached max parallel, wait for one to finish
        if [ $job_count -ge $MAX_PARALLEL ]; then
            wait -n  # Wait for any job to finish
            job_count=$((job_count - 1))
        fi
    done
    
    # Wait for all remaining jobs
    echo "Waiting for remaining jobs to complete..."
    wait
    
    echo ""
    echo "All ablations for $dataset completed!"
}

# Main execution
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

if [ "$DATASET" = "all" ] || [ "$DATASET" = "geolife" ]; then
    run_dataset_ablations "geolife" "config/models/config_pointer_v45_geolife.yaml"
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "diy" ]; then
    run_dataset_ablations "diy" "config/models/config_pointer_v45_diy.yaml"
fi

# Collect results
echo ""
echo "============================================================"
echo "COLLECTING RESULTS"
echo "============================================================"

python -c "
import os
import json
from pathlib import Path

results = {'geolife': {}, 'diy': {}}
ablation_dir = Path('experiments/ablation_study')

for exp_dir in ablation_dir.iterdir():
    if not exp_dir.is_dir() or not exp_dir.name.startswith('ablation_'):
        continue
    
    test_file = exp_dir / 'test_results.json'
    info_file = exp_dir / 'ablation_info.json'
    
    if test_file.exists() and info_file.exists():
        with open(test_file) as f:
            test_results = json.load(f)
        with open(info_file) as f:
            info = json.load(f)
        
        ablation = info['ablation_name']
        if 'geolife' in exp_dir.name:
            ds = 'geolife'
        elif 'diy' in exp_dir.name:
            ds = 'diy'
        else:
            continue
        
        if ablation not in results[ds] or exp_dir.name > results[ds][ablation].get('dir', ''):
            results[ds][ablation] = {'results': test_results, 'dir': exp_dir.name}

# Print summary
for ds in ['geolife', 'diy']:
    if not results[ds]:
        continue
    print(f'\\n{\"=\"*80}')
    print(f'{ds.upper()} ABLATION RESULTS')
    print(f'{\"=\"*80}')
    print(f'{\"Ablation\":<25} {\"Acc@1\":>8} {\"Acc@5\":>8} {\"Acc@10\":>8} {\"MRR\":>8} {\"NDCG\":>8} {\"Δ\":>8}')
    print('-' * 80)
    
    full_acc = results[ds].get('full_model', {}).get('results', {}).get('acc@1', 0)
    
    for abl in ['full_model', 'no_user_emb', 'no_time_emb', 'no_weekday_emb', 
                'no_recency_emb', 'no_duration_emb', 'no_pos_from_end',
                'no_sinusoidal_pos', 'no_temporal', 'no_pointer', 
                'no_generation', 'no_gate', 'single_layer']:
        if abl in results[ds]:
            r = results[ds][abl]['results']
            delta = r['acc@1'] - full_acc if abl != 'full_model' else 0
            print(f'{abl:<25} {r[\"acc@1\"]:>8.2f} {r[\"acc@5\"]:>8.2f} {r[\"acc@10\"]:>8.2f} {r[\"mrr\"]:>8.2f} {r[\"ndcg\"]:>8.2f} {delta:>+8.2f}')

# Save JSON
with open(ablation_dir / 'ablation_results_${TIMESTAMP}.json', 'w') as f:
    json.dump({ds: {k: v['results'] for k, v in data.items()} for ds, data in results.items()}, f, indent=2)
print(f'\\nResults saved to: experiments/ablation_study/ablation_results_${TIMESTAMP}.json')
"

echo ""
echo "============================================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================================"
