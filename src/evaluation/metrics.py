"""
Evaluation metrics for next location prediction.

This module provides reusable evaluation metrics that can be used across
all models (baseline and proposed). The implementation is based on the
reference code from location-prediction-ori-freeze.

Metrics included:
- correct@1, correct@3, correct@5, correct@10: Top-k accuracy counts
- acc@1, acc@5, acc@10: Top-k accuracy percentages
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks
- NDCG@10 (Normalized Discounted Cumulative Gain): Ranking quality metric
- F1: Weighted F1 score for top-1 predictions

Usage:
    import torch
    from src.evaluation.metrics import calculate_metrics
    
    # For batch evaluation
    logits = model(inputs)  # [batch_size, num_locations]
    targets = labels        # [batch_size]
    
    results = calculate_metrics(logits, targets)
    # Returns dict with: correct@1, correct@3, correct@5, correct@10,
    #                    rr, ndcg, total, acc@1, acc@5, acc@10, mrr, f1
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


def get_mrr(prediction, targets):
    """
    Calculates the MRR (Mean Reciprocal Rank) score for the given predictions and targets.
    
    The reciprocal rank is 1/rank where rank is the position of the correct target
    in the sorted prediction list (1-indexed). This function returns the sum of
    reciprocal ranks across all samples.
    
    Args:
        prediction (torch.Tensor): Shape [B, K] - the softmax output/logits of the model.
        targets (torch.Tensor): Shape [B] - actual target indices.
    
    Returns:
        float: The sum of reciprocal ranks across all samples.
    
    Example:
        >>> prediction = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.1, 0.7]])
        >>> targets = torch.tensor([1, 2])
        >>> rr_sum = get_mrr(prediction, targets)
        >>> # For first sample, target 1 is at rank 1, rr = 1/1 = 1.0
        >>> # For second sample, target 2 is at rank 1, rr = 1/1 = 1.0
        >>> # Sum = 2.0
    """
    # Sort predictions in descending order to get ranks
    index = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find where the target appears in the sorted predictions
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    
    # Get ranks (1-indexed)
    ranks = (hits[:, -1] + 1).float()
    
    # Calculate reciprocal ranks
    rranks = torch.reciprocal(ranks)
    
    return torch.sum(rranks).cpu().numpy()


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG@k (Normalized Discounted Cumulative Gain) score.
    
    NDCG measures the quality of ranking by considering the position of the
    correct target. Items at higher ranks contribute more to the score.
    The discount is logarithmic: 1/log2(rank + 1).
    
    Only predictions within top-k are considered. Predictions beyond rank k
    receive a score of 0.
    
    Args:
        prediction (torch.Tensor): Shape [B, K] - the softmax output/logits of the model.
        targets (torch.Tensor): Shape [B] - actual target indices.
        k (int): Consider only top-k predictions. Default is 10.
    
    Returns:
        float: The sum of NDCG scores across all samples.
    
    Example:
        >>> prediction = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.1, 0.7]])
        >>> targets = torch.tensor([1, 0])
        >>> ndcg_sum = get_ndcg(prediction, targets, k=10)
        >>> # For first sample, target 1 is at rank 1, ndcg = 1/log2(2) = 1.0
        >>> # For second sample, target 0 is at rank 2, ndcg = 1/log2(3) ≈ 0.631
    """
    # Sort predictions in descending order to get ranks
    index = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find where the target appears in the sorted predictions
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    
    # Get ranks (1-indexed) as numpy array
    ranks = (hits[:, -1] + 1).float().cpu().numpy()
    
    # Calculate NDCG with logarithmic discount
    ndcg = 1 / np.log2(ranks + 1)
    
    # Zero out scores for ranks beyond k
    not_considered_idx = ranks > k
    ndcg[not_considered_idx] = 0
    
    return np.sum(ndcg)


def calculate_correct_total_prediction(logits, true_y):
    """
    Calculate top-k accuracy counts, MRR, and NDCG for predictions.
    
    This is the core function that computes all the base metrics needed for
    evaluation. It calculates:
    - Top-k hits for k in [1, 3, 5, 10]
    - Sum of reciprocal ranks (for MRR calculation)
    - Sum of NDCG scores
    - Total number of samples
    
    Args:
        logits (torch.Tensor): Shape [B, num_locations] - model predictions/logits.
        true_y (torch.Tensor): Shape [B] - ground truth target indices.
    
    Returns:
        tuple: (result_array, true_labels, top1_predictions)
            - result_array (np.ndarray): Shape [7] containing:
                [correct@1, correct@3, correct@5, correct@10, rr_sum, ndcg_sum, total]
            - true_labels (torch.Tensor): True labels on CPU
            - top1_predictions (torch.Tensor): Top-1 predictions on CPU
    
    Example:
        >>> logits = torch.randn(32, 100)  # 32 samples, 100 locations
        >>> targets = torch.randint(0, 100, (32,))
        >>> results, true_y, top1 = calculate_correct_total_prediction(logits, targets)
        >>> print(f"Correct@1: {results[0]}, Total: {results[6]}")
    """
    top1 = []
    result_ls = []
    
    # Calculate top-k accuracy for k in [1, 3, 5, 10]
    for k in [1, 3, 5, 10]:
        # Handle case where number of classes is less than k
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        
        # Get top-k predictions
        prediction = torch.topk(logits, k=k, dim=-1).indices
        
        # Store top-1 predictions for F1 score calculation
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()
        
        # Count how many predictions contain the correct target
        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
    
    # Add MRR sum
    result_ls.append(get_mrr(logits, true_y))
    
    # Add NDCG sum
    result_ls.append(get_ndcg(logits, true_y))
    
    # Add total number of samples
    result_ls.append(true_y.shape[0])
    
    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1


def get_performance_dict(return_dict):
    """
    Convert raw metric counts to percentage-based performance metrics.
    
    This function takes the accumulated counts from evaluation and converts
    them into interpretable percentages and ratios.
    
    Args:
        return_dict (dict): Dictionary containing raw counts:
            - correct@1, correct@3, correct@5, correct@10: Hit counts
            - rr: Sum of reciprocal ranks
            - ndcg: Sum of NDCG scores
            - f1: F1 score (already computed)
            - total: Total number of samples
    
    Returns:
        dict: Performance metrics with percentages:
            - correct@1, correct@3, correct@5, correct@10: Hit counts (unchanged)
            - acc@1, acc@5, acc@10: Accuracy percentages
            - mrr: Mean Reciprocal Rank as percentage
            - ndcg: Mean NDCG as percentage
            - f1: F1 score (unchanged)
            - total: Total samples (unchanged)
    
    Example:
        >>> return_dict = {
        ...     'correct@1': 80, 'correct@3': 90, 'correct@5': 95, 'correct@10': 98,
        ...     'rr': 85.5, 'ndcg': 88.2, 'f1': 0.78, 'total': 100
        ... }
        >>> perf = get_performance_dict(return_dict)
        >>> print(f"Accuracy@1: {perf['acc@1']:.2f}%")  # 80.00%
    """
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "ndcg": return_dict["ndcg"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }
    
    # Calculate accuracy percentages
    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    
    # Calculate mean reciprocal rank as percentage
    perf["mrr"] = perf["rr"] / perf["total"] * 100
    
    # Calculate mean NDCG as percentage
    perf["ndcg"] = perf["ndcg"] / perf["total"] * 100
    
    return perf


def calculate_metrics(logits, targets, return_predictions=False):
    """
    High-level function to calculate all evaluation metrics.
    
    This is the main entry point for metric calculation. It computes all
    metrics including F1 score and returns a comprehensive dictionary.
    
    Args:
        logits (torch.Tensor): Shape [B, num_locations] - model predictions/logits.
        targets (torch.Tensor): Shape [B] - ground truth target indices.
        return_predictions (bool): If True, also return predictions. Default False.
    
    Returns:
        dict: Complete performance metrics including:
            - correct@1, correct@3, correct@5, correct@10: Hit counts
            - acc@1, acc@5, acc@10: Accuracy percentages
            - mrr: Mean Reciprocal Rank as percentage
            - ndcg: Mean NDCG as percentage
            - f1: Weighted F1 score as percentage
            - total: Total number of samples
        
        If return_predictions=True, returns tuple (metrics_dict, predictions_dict)
        where predictions_dict contains:
            - true_labels: Ground truth labels
            - top1_predictions: Top-1 predicted labels
    
    Example:
        >>> import torch
        >>> logits = torch.randn(100, 50)  # 100 samples, 50 locations
        >>> targets = torch.randint(0, 50, (100,))
        >>> metrics = calculate_metrics(logits, targets)
        >>> print(f"Accuracy@1: {metrics['acc@1']:.2f}%")
        >>> print(f"MRR: {metrics['mrr']:.2f}%")
        >>> print(f"F1 Score: {metrics['f1']:.2f}%")
    """
    # Calculate base metrics
    result_arr, true_labels, top1_preds = calculate_correct_total_prediction(logits, targets)
    
    # Calculate F1 score
    # Handle both single element and array cases for top1_preds
    if not top1_preds.shape:
        top1_list = [top1_preds.tolist()]
    else:
        top1_list = top1_preds.tolist()
    
    true_list = true_labels.tolist()
    f1 = f1_score(true_list, top1_list, average="weighted")
    
    # Build return dictionary with raw counts
    return_dict = {
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "rr": result_arr[4],
        "ndcg": result_arr[5],
        "f1": f1,
        "total": result_arr[6],
    }
    
    # Convert to performance metrics with percentages
    performance = get_performance_dict(return_dict)
    
    # Convert F1 to percentage
    performance["f1"] = performance["f1"] * 100
    
    if return_predictions:
        predictions = {
            "true_labels": true_labels,
            "top1_predictions": top1_preds,
        }
        return performance, predictions
    
    return performance


def accumulate_metrics(metrics_list):
    """
    Accumulate metrics across multiple batches or folds.
    
    Useful for combining results from multiple evaluation batches or
    cross-validation folds.
    
    Args:
        metrics_list (list): List of metric dictionaries from calculate_correct_total_prediction.
            Each dict should contain: correct@1, correct@3, correct@5, correct@10,
            rr, ndcg, total.
    
    Returns:
        dict: Accumulated metrics ready for get_performance_dict.
    
    Example:
        >>> batch_results = []
        >>> for batch in dataloader:
        ...     logits = model(batch)
        ...     result_arr, _, _ = calculate_correct_total_prediction(logits, targets)
        ...     batch_results.append({
        ...         'correct@1': result_arr[0], 'correct@3': result_arr[1],
        ...         'correct@5': result_arr[2], 'correct@10': result_arr[3],
        ...         'rr': result_arr[4], 'ndcg': result_arr[5], 'total': result_arr[6]
        ...     })
        >>> accumulated = accumulate_metrics(batch_results)
    """
    accumulated = {
        "correct@1": 0,
        "correct@3": 0,
        "correct@5": 0,
        "correct@10": 0,
        "rr": 0,
        "ndcg": 0,
        "total": 0,
    }
    
    for metrics in metrics_list:
        for key in accumulated.keys():
            accumulated[key] += metrics[key]
    
    return accumulated


def evaluate_model(model, dataloader, device, verbose=False):
    """
    Evaluate a model on a full dataset.
    
    This function handles the complete evaluation loop including:
    - Batch processing
    - Metric accumulation
    - F1 score calculation across all samples
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Data loader for evaluation data.
        device (torch.device): Device to run evaluation on.
        verbose (bool): Print progress information. Default True.
    
    Returns:
        dict: Complete performance metrics including:
            - correct@1, correct@3, correct@5, correct@10
            - acc@1, acc@5, acc@10
            - mrr, ndcg, f1 (all as percentages)
            - total
    
    Example:
        >>> import torch
        >>> model = MyModel()
        >>> test_loader = DataLoader(test_dataset, batch_size=32)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> metrics = evaluate_model(model, test_loader, device)
        >>> print(f"Test Accuracy: {metrics['acc@1']:.2f}%")
    """
    model.eval()
    
    # Initialize accumulators
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    true_ls = []
    top1_ls = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions
            logits = model(inputs)
            
            # Calculate metrics for this batch
            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(
                logits, targets
            )
            
            # Accumulate results
            result_arr += batch_result_arr
            true_ls.extend(batch_true.tolist())
            
            # Handle scalar vs array for top1 predictions
            if not batch_top1.shape:
                top1_ls.extend([batch_top1.tolist()])
            else:
                top1_ls.extend(batch_top1.tolist())
    
    # Calculate F1 score across all samples
    f1 = f1_score(true_ls, top1_ls, average="weighted")
    
    # Build return dictionary
    return_dict = {
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "rr": result_arr[4],
        "ndcg": result_arr[5],
        "f1": f1,
        "total": result_arr[6],
    }
    
    # Convert to performance metrics
    performance = get_performance_dict(return_dict)
    
    # Convert F1 to percentage
    performance["f1"] = performance["f1"] * 100
    
    if verbose:
        print(
            f"acc@1 = {performance['acc@1']:.2f}% "
            f"f1 = {performance['f1']:.2f}% "
            f"mrr = {performance['mrr']:.2f}% "
            f"ndcg = {performance['ndcg']:.2f}%"
        )
    
    return performance
