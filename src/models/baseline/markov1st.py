"""
1st-Order Markov Chain Model for Next Location Prediction.

This module implements a first-order Markov chain model that predicts the next 
location based on transition probabilities learned from training data.

The model learns per-user transition probabilities:
    P(next_location | current_location, user)

For each user, the model builds a transition matrix where each entry (i, j) 
represents the count of transitions from location i to location j.

Reference:
    Original implementation from location-prediction-ori-freeze/baselines/markov.py

Usage:
    from src.models.baseline.markov1st import Markov1stModel
    
    model = Markov1stModel(num_locations=1000, random_seed=42)
    model.fit(train_data)
    predictions = model.predict(test_sequences, user_ids)
"""

import numpy as np
import torch
from collections import defaultdict


class Markov1stModel:
    """
    First-order Markov Chain model for next location prediction.
    
    The model maintains per-user transition counts:
        transition_counts[user_id][from_loc][to_loc] = count
    
    Prediction is done by looking up the most frequently visited next location
    given the current location and user.
    
    Attributes:
        num_locations (int): Total number of unique locations.
        random_seed (int): Random seed for reproducibility.
        transition_counts (dict): Nested dict of transition counts per user.
        global_transition_counts (dict): Fallback transition counts across all users.
        location_counts (dict): Count of visits per location (for fallback).
    """
    
    def __init__(self, num_locations, random_seed=42):
        """
        Initialize the Markov model.
        
        Args:
            num_locations (int): Total number of unique locations in the dataset.
            random_seed (int): Random seed for reproducibility. Default is 42.
        """
        self.num_locations = num_locations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Per-user transition counts: user -> from_loc -> to_loc -> count
        self.transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Global transition counts (fallback): from_loc -> to_loc -> count
        self.global_transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Location frequency (for fallback when no transition data available)
        self.location_counts = defaultdict(int)
        
        # Store sorted locations by frequency for fallback
        self.sorted_locations = None
        
        # Total parameters (number of unique transitions)
        self.total_parameters = 0
    
    def fit(self, train_data):
        """
        Fit the model on training data.
        
        Learns transition probabilities from sequences in the training data.
        
        Args:
            train_data (list): List of dictionaries, each containing:
                - 'X': numpy array of location sequence
                - 'user_X': numpy array of user IDs (uses first element)
                - 'Y': target location ID
        
        Returns:
            self: The fitted model.
        """
        # Group data by user
        user_sequences = defaultdict(list)
        
        for sample in train_data:
            user_id = sample['user_X'][0]
            locations = list(sample['X'])
            target = sample['Y']
            
            # Add the full sequence including target
            full_sequence = locations + [target]
            user_sequences[user_id].append(full_sequence)
        
        # Build transition counts for each user
        for user_id, sequences in user_sequences.items():
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    from_loc = sequence[i]
                    to_loc = sequence[i + 1]
                    
                    # User-specific transitions
                    self.transition_counts[user_id][from_loc][to_loc] += 1
                    
                    # Global transitions (fallback)
                    self.global_transition_counts[from_loc][to_loc] += 1
                    
                    # Location counts
                    self.location_counts[to_loc] += 1
        
        # Compute total parameters (number of unique transitions per user)
        for user_id in self.transition_counts:
            num_from_locs = len(self.transition_counts[user_id])
            # Approximate: num_from_locs squared for potential transitions
            self.total_parameters += sum(
                len(to_locs) for to_locs in self.transition_counts[user_id].values()
            )
        
        # Sort locations by global frequency for fallback
        self.sorted_locations = sorted(
            self.location_counts.keys(), 
            key=lambda x: self.location_counts[x], 
            reverse=True
        )
        
        return self
    
    def predict_single(self, current_loc, user_id, top_k=10):
        """
        Predict next location for a single sample.
        
        Args:
            current_loc (int): Current location ID.
            user_id (int): User ID.
            top_k (int): Number of top predictions to return.
        
        Returns:
            numpy.ndarray: Array of top-k predicted location IDs.
        """
        # Try user-specific transitions first
        if user_id in self.transition_counts:
            user_transitions = self.transition_counts[user_id]
            
            if current_loc in user_transitions:
                # Get transitions from current location, sorted by count
                transitions = user_transitions[current_loc]
                sorted_locs = sorted(
                    transitions.keys(), 
                    key=lambda x: transitions[x], 
                    reverse=True
                )
                
                if len(sorted_locs) >= top_k:
                    return np.array(sorted_locs[:top_k])
                else:
                    # Pad with global fallback
                    result = list(sorted_locs)
                    if self.sorted_locations:
                        for loc in self.sorted_locations:
                            if loc not in result:
                                result.append(loc)
                                if len(result) >= top_k:
                                    break
                    # If still not enough, pad with zeros
                    while len(result) < top_k:
                        result.append(0)
                    return np.array(result[:top_k])
        
        # Fallback to global transitions
        if current_loc in self.global_transition_counts:
            transitions = self.global_transition_counts[current_loc]
            sorted_locs = sorted(
                transitions.keys(), 
                key=lambda x: transitions[x], 
                reverse=True
            )
            
            if len(sorted_locs) >= top_k:
                return np.array(sorted_locs[:top_k])
            else:
                result = list(sorted_locs)
                if self.sorted_locations:
                    for loc in self.sorted_locations:
                        if loc not in result:
                            result.append(loc)
                            if len(result) >= top_k:
                                break
                # If still not enough, pad with zeros
                while len(result) < top_k:
                    result.append(0)
                return np.array(result[:top_k])
        
        # Ultimate fallback: most frequent locations
        if self.sorted_locations:
            result = list(self.sorted_locations[:min(len(self.sorted_locations), top_k)])
            while len(result) < top_k:
                result.append(0)
            return np.array(result[:top_k])
        
        # If no data at all, return zeros
        return np.zeros(top_k, dtype=np.int64)
    
    def predict(self, test_data, top_k=10):
        """
        Predict next locations for all test samples.
        
        Args:
            test_data (list): List of dictionaries, each containing:
                - 'X': numpy array of location sequence
                - 'user_X': numpy array of user IDs
                - 'Y': target location ID (ground truth)
            top_k (int): Number of top predictions to return.
        
        Returns:
            tuple: (predictions, targets)
                - predictions: list of numpy arrays, each of shape [top_k]
                - targets: list of ground truth location IDs
        """
        predictions = []
        targets = []
        
        for sample in test_data:
            user_id = sample['user_X'][0]
            current_loc = sample['X'][-1]  # Use last location in sequence
            target = sample['Y']
            
            pred = self.predict_single(current_loc, user_id, top_k)
            predictions.append(pred)
            targets.append(target)
        
        return predictions, targets
    
    def predict_as_logits(self, test_data):
        """
        Generate logit-like scores for all test samples.
        
        Creates a score matrix compatible with the evaluation metrics module.
        For Markov model, scores are based on transition counts normalized.
        
        Args:
            test_data (list): List of dictionaries with test samples.
        
        Returns:
            tuple: (logits_tensor, targets_tensor)
                - logits_tensor: torch.Tensor of shape [N, num_locations]
                - targets_tensor: torch.Tensor of shape [N]
        """
        all_logits = []
        all_targets = []
        
        for sample in test_data:
            user_id = sample['user_X'][0]
            current_loc = sample['X'][-1]
            target = sample['Y']
            
            # Initialize logits with small value
            logits = np.full(self.num_locations, -1e9, dtype=np.float32)
            
            # Try user-specific transitions first
            if user_id in self.transition_counts:
                user_transitions = self.transition_counts[user_id]
                
                if current_loc in user_transitions:
                    transitions = user_transitions[current_loc]
                    total_count = sum(transitions.values())
                    
                    for to_loc, count in transitions.items():
                        if to_loc < self.num_locations:
                            # Log probability as logit
                            logits[to_loc] = np.log(count / total_count + 1e-10)
                else:
                    # Use all user's transitions as fallback
                    all_transitions = defaultdict(int)
                    for from_loc, to_locs in user_transitions.items():
                        for to_loc, count in to_locs.items():
                            all_transitions[to_loc] += count
                    
                    if all_transitions:
                        total_count = sum(all_transitions.values())
                        for to_loc, count in all_transitions.items():
                            if to_loc < self.num_locations:
                                logits[to_loc] = np.log(count / total_count + 1e-10)
            
            # If still no data, use global transition or frequency
            if np.max(logits) < -1e8:
                if current_loc in self.global_transition_counts:
                    transitions = self.global_transition_counts[current_loc]
                    total_count = sum(transitions.values())
                    
                    for to_loc, count in transitions.items():
                        if to_loc < self.num_locations:
                            logits[to_loc] = np.log(count / total_count + 1e-10)
                else:
                    # Use location frequency
                    total_count = sum(self.location_counts.values())
                    for loc, count in self.location_counts.items():
                        if loc < self.num_locations:
                            logits[loc] = np.log(count / total_count + 1e-10)
            
            all_logits.append(logits)
            all_targets.append(target)
        
        logits_tensor = torch.tensor(np.array(all_logits), dtype=torch.float32)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        
        return logits_tensor, targets_tensor
    
    def get_total_parameters(self):
        """
        Get the total number of parameters (unique transitions) in the model.
        
        Returns:
            int: Total number of stored transitions.
        """
        return self.total_parameters
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        import pickle
        
        model_data = {
            'num_locations': self.num_locations,
            'random_seed': self.random_seed,
            'transition_counts': dict(self.transition_counts),
            'global_transition_counts': dict(self.global_transition_counts),
            'location_counts': dict(self.location_counts),
            'sorted_locations': self.sorted_locations,
            'total_parameters': self.total_parameters,
        }
        
        # Convert nested defaultdicts to regular dicts
        model_data['transition_counts'] = {
            user: {from_loc: dict(to_locs) for from_loc, to_locs in user_trans.items()}
            for user, user_trans in self.transition_counts.items()
        }
        model_data['global_transition_counts'] = {
            from_loc: dict(to_locs) 
            for from_loc, to_locs in self.global_transition_counts.items()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a model from a file.
        
        Args:
            path (str): Path to the saved model.
        
        Returns:
            Markov1stModel: The loaded model.
        """
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            num_locations=model_data['num_locations'],
            random_seed=model_data['random_seed']
        )
        
        # Restore transition counts
        for user, user_trans in model_data['transition_counts'].items():
            for from_loc, to_locs in user_trans.items():
                for to_loc, count in to_locs.items():
                    model.transition_counts[user][from_loc][to_loc] = count
        
        for from_loc, to_locs in model_data['global_transition_counts'].items():
            for to_loc, count in to_locs.items():
                model.global_transition_counts[from_loc][to_loc] = count
        
        for loc, count in model_data['location_counts'].items():
            model.location_counts[loc] = count
        
        model.sorted_locations = model_data['sorted_locations']
        model.total_parameters = model_data['total_parameters']
        
        return model
