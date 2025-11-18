#!/usr/bin/env python3
"""
Generate a summary report of all GR-1 observations.

Usage:
    python scripts/generate_obs_summary.py <base_dir>
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import json

def load_observation(filepath):
    """Load a single observation file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_task(obs_dir):
    """Analyze a single task's observations."""
    obs_path = Path(obs_dir)
    if not obs_path.exists():
        return None
    
    pkl_files = sorted(obs_path.glob("*.pkl"))
    if len(pkl_files) == 0:
        return None
    
    # Load first episode to get structure
    first_obs = load_observation(pkl_files[0])
    if first_obs is None:
        return None
    
    episode_lengths = []
    total_size = 0
    
    for pkl_file in pkl_files:
        try:
            obs = load_observation(pkl_file)
            if isinstance(obs, list):
                episode_lengths.append(len(obs))
            total_size += pkl_file.stat().st_size
        except:
            continue
    
    if not episode_lengths:
        return None
    
    # Get observation keys from first episode
    obs_keys = []
    if isinstance(first_obs, list) and len(first_obs) > 0:
        obs_keys = list(first_obs[0].keys())
    elif isinstance(first_obs, dict):
        obs_keys = list(first_obs.keys())
    
    return {
        'task_name': obs_path.parent.name,
        'num_episodes': len(episode_lengths),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        'min_episode_length': int(np.min(episode_lengths)),
        'max_episode_length': int(np.max(episode_lengths)),
        'total_steps': int(np.sum(episode_lengths)),
        'total_size_mb': total_size / (1024 * 1024),
        'mean_size_mb': (total_size / len(pkl_files)) / (1024 * 1024),
        'observation_keys': obs_keys,
    }

def generate_summary(base_dir):
    """Generate summary report for all tasks."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all observation directories
    obs_dirs = list(base_path.rglob("observations"))
    
    if len(obs_dirs) == 0:
        print(f"Error: No observation directories found in {base_dir}")
        return
    
    print(f"Found {len(obs_dirs)} observation directories")
    print("=" * 100)
    
    task_results = []
    for obs_dir in sorted(obs_dirs):
        result = analyze_task(obs_dir)
        if result:
            task_results.append(result)
    
    if not task_results:
        print("No valid observation data found")
        return
    
    # Print summary table
    print("\n" + "=" * 100)
    print("GR-1 Observation Summary Report")
    print("=" * 100)
    print(f"\nTotal tasks analyzed: {len(task_results)}")
    
    # Overall statistics
    total_episodes = sum(r['num_episodes'] for r in task_results)
    total_steps = sum(r['total_steps'] for r in task_results)
    total_size = sum(r['total_size_mb'] for r in task_results)
    
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print(f"Average episodes per task: {total_episodes / len(task_results):.1f}")
    print(f"Average steps per task: {total_steps / len(task_results):.1f}")
    
    # Per-task breakdown
    print("\n" + "-" * 100)
    print("Per-Task Breakdown")
    print("-" * 100)
    print(f"{'Task Name':<60} {'Episodes':<10} {'Steps':<10} {'Size (MB)':<12} {'Mean Len':<10}")
    print("-" * 100)
    
    for result in sorted(task_results, key=lambda x: x['task_name']):
        task_name = result['task_name']
        if len(task_name) > 60:
            task_name = task_name[:57] + "..."
        
        print(f"{task_name:<60} {result['num_episodes']:<10} {result['total_steps']:<10} "
              f"{result['total_size_mb']:<12.2f} {result['mean_episode_length']:<10.1f}")
    
    # Observation keys consistency
    print("\n" + "-" * 100)
    print("Observation Keys")
    print("-" * 100)
    if task_results:
        all_keys = set()
        for result in task_results:
            all_keys.update(result['observation_keys'])
        
        print(f"Total unique keys: {len(all_keys)}")
        print(f"Keys: {sorted(all_keys)}")
        
        # Check consistency
        key_counts = {}
        for result in task_results:
            for key in result['observation_keys']:
                key_counts[key] = key_counts.get(key, 0) + 1
        
        print(f"\nKey frequency across tasks:")
        for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
            print(f"  {key}: {count}/{len(task_results)} tasks")
    
    # Save to JSON
    output_file = base_path / "observation_summary.json"
    summary_data = {
        'total_tasks': len(task_results),
        'total_episodes': total_episodes,
        'total_steps': total_steps,
        'total_size_mb': total_size,
        'tasks': task_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary report of GR-1 observations")
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing observation directories",
    )
    args = parser.parse_args()
    
    generate_summary(args.base_dir)

