#!/bin/bash

# Wrapper script to run speedrun.sh completely detached from terminal

# Create a new session and run speedrun.sh
setsid bash speedrun_2h100.sh > speedrun_2h100.log 2>&1 < /dev/null &

# Get the PID
SPEEDRUN_PID=$!

echo "Started speedrun.sh with PID: $SPEEDRUN_PID"
echo "Log file: speedrun_kimberly.log"
echo ""
echo "To check if it's running:"
echo "  ps aux | grep speedrun.sh"
echo ""
echo "To view logs in real-time:"
echo "  tail -f speedrun_kimberly.log"
echo ""
echo "To stop the process:"
echo "  kill $SPEEDRUN_PID  # Stop training"

