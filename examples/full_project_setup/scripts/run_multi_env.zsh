#!/bin/zsh

# Path to the scene file
# SCENE="./scenes/arena_obstacles_middleSceneOnly.ttt"
SCENE="./scenes/arena_approach_v2.ttt"

# Path to the script that starts CoppeliaSim
SCRIPT="./scripts/start_coppelia_sim.zsh"

# Number of instances to run
NUM_INSTANCES=1

# Base port number
BASE_PORT=20000

echo "Launching $NUM_INSTANCES instances of CoppeliaSim..."

for ((i=0; i<NUM_INSTANCES; i++)); do
  PORT=$((BASE_PORT + i))
  echo "Starting instance $i on port $PORT..."
  zsh "$SCRIPT" "$SCENE" "$PORT" &
  sleep 0.2
done

echo "All instances started in background."

# Optional: Wait for all background processes to finish
# wait
