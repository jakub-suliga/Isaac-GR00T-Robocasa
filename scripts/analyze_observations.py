#!/usr/bin/env python3
"""
Analyze GR-1 observation files saved during simulation.

Usage:
    python scripts/analyze_observations.py <obs_dir>
    python scripts/analyze_observations.py output/gr1_new/nvidia_GR00T-N1.5-3B/gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env/observations
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_observation(filepath):
    """Load a single observation file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_observation_structure(obs_dict):
    """Analyze the structure of a single observation."""
    info = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            info[key] = {
                'shape': value.shape,
                'dtype': str(value.dtype),
                'size': value.size,
                'min': float(np.min(value)) if value.size > 0 else None,
                'max': float(np.max(value)) if value.size > 0 else None,
                'mean': float(np.mean(value)) if value.size > 0 else None,
            }
        elif isinstance(value, (list, tuple)):
            info[key] = {
                'type': type(value).__name__,
                'length': len(value),
                'element_types': [type(v).__name__ for v in value[:5]] if len(value) > 0 else [],
            }
        else:
            info[key] = {
                'type': type(value).__name__,
                'value': str(value)[:100] if not isinstance(value, (dict, list)) else '...',
            }
    return info

def analyze_observations(obs_dir):
    """Analyze all observation files in a directory."""
    obs_dir = Path(obs_dir)
    if not obs_dir.exists():
        print(f"Error: Directory {obs_dir} does not exist")
        return
    
    # Find all pkl files
    pkl_files = sorted(obs_dir.glob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"Error: No .pkl files found in {obs_dir}")
        return
    
    print(f"Found {len(pkl_files)} observation files in {obs_dir}")
    print("=" * 80)
    
    # Load first file to understand structure
    print("\n1. Observation Structure Analysis")
    print("-" * 80)
    first_obs = load_observation(pkl_files[0])
    
    if isinstance(first_obs, list):
        print(f"Observation is a list with {len(first_obs)} items")
        if len(first_obs) > 0:
            print(f"\nFirst item structure:")
            first_item = first_obs[0]
            if isinstance(first_item, dict):
                struct = analyze_observation_structure(first_item)
                for key, info in struct.items():
                    print(f"  {key}: {info}")
            else:
                print(f"  Type: {type(first_item)}")
                print(f"  Value: {first_item}")
    elif isinstance(first_obs, dict):
        print("Observation is a dictionary")
        struct = analyze_observation_structure(first_obs)
        for key, info in struct.items():
            print(f"  {key}: {info}")
    else:
        print(f"Observation type: {type(first_obs)}")
    
    # Analyze all files
    print("\n2. Episode Statistics")
    print("-" * 80)
    episode_lengths = []
    episode_keys = defaultdict(list)
    
    for pkl_file in pkl_files:
        try:
            obs = load_observation(pkl_file)
            if isinstance(obs, list):
                episode_lengths.append(len(obs))
                if len(obs) > 0 and isinstance(obs[0], dict):
                    episode_keys[pkl_file.name].extend(obs[0].keys())
            elif isinstance(obs, dict):
                episode_lengths.append(1)
                episode_keys[pkl_file.name].extend(obs.keys())
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
    
    if episode_lengths:
        episode_lengths = np.array(episode_lengths)
        print(f"Total episodes: {len(episode_lengths)}")
        print(f"Episode length - Mean: {np.mean(episode_lengths):.1f}, Std: {np.std(episode_lengths):.1f}")
        print(f"Episode length - Min: {np.min(episode_lengths)}, Max: {np.max(episode_lengths)}")
        print(f"Total steps: {np.sum(episode_lengths)}")
    
    # Check for common keys across episodes
    print("\n3. Observation Keys Across Episodes")
    print("-" * 80)
    if episode_keys:
        all_keys = set()
        for keys in episode_keys.values():
            all_keys.update(keys)
        print(f"Unique keys found: {sorted(all_keys)}")
        
        # Check consistency
        key_counts = defaultdict(int)
        for keys in episode_keys.values():
            for key in set(keys):
                key_counts[key] += 1
        
        print(f"\nKey frequency (how many episodes contain each key):")
        for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
            print(f"  {key}: {count}/{len(episode_keys)} episodes")
    
    # Sample a few observations for detailed analysis
    print("\n4. Sample Observation Details")
    print("-" * 80)
    for i, pkl_file in enumerate(pkl_files[:3]):  # Analyze first 3 episodes
        print(f"\nEpisode {i} ({pkl_file.name}):")
        try:
            obs = load_observation(pkl_file)
            if isinstance(obs, list):
                print(f"  Length: {len(obs)} steps")
                if len(obs) > 0:
                    print(f"  First step keys: {list(obs[0].keys()) if isinstance(obs[0], dict) else 'N/A'}")
                    if len(obs) > 1:
                        print(f"  Last step keys: {list(obs[-1].keys()) if isinstance(obs[-1], dict) else 'N/A'}")
            elif isinstance(obs, dict):
                print(f"  Single observation with keys: {list(obs.keys())}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # File sizes
    print("\n5. File Size Statistics")
    print("-" * 80)
    file_sizes = [f.stat().st_size / (1024*1024) for f in pkl_files]  # MB
    print(f"Total size: {sum(file_sizes):.2f} MB")
    print(f"Mean size: {np.mean(file_sizes):.2f} MB")
    print(f"Size range: {np.min(file_sizes):.2f} - {np.max(file_sizes):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GR-1 observation files")
    parser.add_argument(
        "obs_dir",
        type=str,
        help="Directory containing observation .pkl files",
    )
    args = parser.parse_args()
    
    analyze_observations(args.obs_dir)

