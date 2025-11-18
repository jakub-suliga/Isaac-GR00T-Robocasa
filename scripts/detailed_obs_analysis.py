#!/usr/bin/env python3
"""
Detailed analysis of GR-1 observations including statistics and visualization.

Usage:
    python scripts/detailed_obs_analysis.py <obs_dir> [--episodes <num>]
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_observation(filepath):
    """Load a single observation file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_state_statistics(obs_list, state_key):
    """Analyze statistics for a state observation across all steps."""
    if not obs_list or state_key not in obs_list[0]:
        return None
    
    all_values = []
    for obs in obs_list:
        if state_key in obs:
            value = obs[state_key]
            if isinstance(value, np.ndarray):
                # Flatten the array
                all_values.append(value.flatten())
    
    if not all_values:
        return None
    
    all_values = np.concatenate(all_values)
    return {
        'mean': np.mean(all_values),
        'std': np.std(all_values),
        'min': np.min(all_values),
        'max': np.max(all_values),
        'median': np.median(all_values),
        'values': all_values,
    }

def analyze_all_tasks(base_dir):
    """Analyze observations across all tasks."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all task directories
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(task_dirs)} task directories")
    print("=" * 80)
    
    task_stats = {}
    
    for task_dir in sorted(task_dirs):
        obs_dir = task_dir / "observations"
        if not obs_dir.exists():
            continue
        
        pkl_files = sorted(obs_dir.glob("*.pkl"))
        if len(pkl_files) == 0:
            continue
        
        task_name = task_dir.name
        print(f"\nAnalyzing task: {task_name}")
        print("-" * 80)
        
        # Load first episode to understand structure
        first_obs = load_observation(pkl_files[0])
        if not isinstance(first_obs, list) or len(first_obs) == 0:
            continue
        
        episode_lengths = []
        for pkl_file in pkl_files:
            try:
                obs = load_observation(pkl_file)
                if isinstance(obs, list):
                    episode_lengths.append(len(obs))
            except:
                continue
        
        if episode_lengths:
            task_stats[task_name] = {
                'num_episodes': len(episode_lengths),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'total_steps': np.sum(episode_lengths),
            }
            
            print(f"  Episodes: {len(episode_lengths)}")
            print(f"  Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"  Total steps: {np.sum(episode_lengths)}")
            
            # Analyze state statistics from first episode
            state_keys = [k for k in first_obs[0].keys() if k.startswith('state.')]
            if state_keys:
                print(f"  State keys: {state_keys}")
                for state_key in state_keys[:3]:  # Show first 3
                    stats = analyze_state_statistics(first_obs, state_key)
                    if stats:
                        print(f"    {state_key}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Summary across all tasks
    print("\n" + "=" * 80)
    print("Summary Across All Tasks")
    print("=" * 80)
    if task_stats:
        total_episodes = sum(s['num_episodes'] for s in task_stats.values())
        total_steps = sum(s['total_steps'] for s in task_stats.values())
        print(f"Total tasks: {len(task_stats)}")
        print(f"Total episodes: {total_episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Average episodes per task: {total_episodes / len(task_stats):.1f}")
        print(f"Average steps per task: {total_steps / len(task_stats):.1f}")

def analyze_single_task(obs_dir, num_episodes=5):
    """Detailed analysis of a single task's observations."""
    obs_path = Path(obs_dir)
    if not obs_path.exists():
        print(f"Error: Directory {obs_dir} does not exist")
        return
    
    pkl_files = sorted(obs_path.glob("*.pkl"))[:num_episodes]
    if len(pkl_files) == 0:
        print(f"Error: No .pkl files found in {obs_dir}")
        return
    
    print(f"Analyzing {len(pkl_files)} episodes from {obs_dir}")
    print("=" * 80)
    
    # Collect all observations
    all_observations = []
    for pkl_file in pkl_files:
        try:
            obs = load_observation(pkl_file)
            if isinstance(obs, list):
                all_observations.extend(obs)
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
    
    if not all_observations:
        print("No observations loaded")
        return
    
    print(f"\nTotal observations: {len(all_observations)}")
    print(f"Observation keys: {list(all_observations[0].keys())}")
    
    # Analyze state observations
    print("\n" + "-" * 80)
    print("State Observation Statistics")
    print("-" * 80)
    
    state_keys = [k for k in all_observations[0].keys() if k.startswith('state.')]
    for state_key in state_keys:
        stats = analyze_state_statistics(all_observations, state_key)
        if stats:
            print(f"\n{state_key}:")
            print(f"  Shape: {all_observations[0][state_key].shape if isinstance(all_observations[0][state_key], np.ndarray) else 'N/A'}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Median: {stats['median']:.6f}")
    
    # Analyze video observations
    print("\n" + "-" * 80)
    print("Video Observation Statistics")
    print("-" * 80)
    
    video_keys = [k for k in all_observations[0].keys() if k.startswith('video.')]
    for video_key in video_keys:
        if video_key in all_observations[0]:
            value = all_observations[0][video_key]
            if isinstance(value, np.ndarray):
                print(f"\n{video_key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                
                # Sample a few frames
                sample_values = []
                for obs in all_observations[:10]:
                    if video_key in obs:
                        sample_values.append(obs[video_key].flatten())
                
                if sample_values:
                    sample_values = np.concatenate(sample_values)
                    print(f"  Pixel value range: [{np.min(sample_values)}, {np.max(sample_values)}]")
                    print(f"  Pixel mean: {np.mean(sample_values):.2f}")
    
    # Analyze annotations
    print("\n" + "-" * 80)
    print("Annotation Analysis")
    print("-" * 80)
    
    annotation_keys = [k for k in all_observations[0].keys() if k.startswith('annotation.')]
    for ann_key in annotation_keys:
        if ann_key in all_observations[0]:
            sample_annotations = [obs[ann_key] for obs in all_observations[:10] if ann_key in obs]
            if sample_annotations:
                print(f"\n{ann_key}:")
                print(f"  Sample (first): {sample_annotations[0]}")
                if len(sample_annotations) > 1:
                    unique = len(set(sample_annotations))
                    print(f"  Unique values in first 10: {unique}/{len(sample_annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed analysis of GR-1 observations")
    parser.add_argument(
        "obs_dir",
        type=str,
        help="Directory containing observation .pkl files, or base directory with multiple task directories",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to analyze in detail (default: 5)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Analyze all tasks in the base directory",
    )
    args = parser.parse_args()
    
    if args.all_tasks:
        analyze_all_tasks(args.obs_dir)
    else:
        analyze_single_task(args.obs_dir, args.episodes)

