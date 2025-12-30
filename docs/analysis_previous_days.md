
   Summary

   Created Files:

     - Automation Script: /data/next_loc_clean_v2/scripts/analysis_prev_days/run_prev_days_analysis.py
     - 66 Config Files: /data/next_loc_clean_v2/config/analysis_prev_days/*.yaml
     - Results CSV Files:
       - Val: /data/next_loc_clean_v2/results/analysis_prev_days/prev_days_analysis_val_20251228_191237.csv
       - Test: /data/next_loc_clean_v2/results/analysis_prev_days/prev_days_analysis_test_20251228_191237.csv

   What the Script Does:

     - Trains 66 models across 6 previous days (3, 7, 10, 14, 17, 21) × (6 GeoLife configs + 5 DIY configs)
     - Runs 3 parallel training sessions with 5-second staggered delays
     - Uses seed=42, patience=5
     - Generates config files dynamically for each combination
     - Appends results to CSV files in real-time as training completes

   Usage:

     # Test with 2 epochs (dry run)
     python scripts/analysis_prev_days/run_prev_days_analysis.py --dry_run

     # Test with 2 epochs (actual training)
     python scripts/analysis_prev_days/run_prev_days_analysis.py --max_epochs 2

     # Full training with 50 epochs
     python scripts/analysis_prev_days/run_prev_days_analysis.py --max_epochs 50

   CSV Columns:

   dataset, prev_days, config_name, d_model, nhead, num_layers, dim_feedforward, learning_rate, num_params, experiment_dir,
   correct@1, correct@3, correct@5, correct@10, total, rr, ndcg, f1, acc@1, acc@5, acc@10, mrr, loss