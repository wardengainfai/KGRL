#!/usr/bin/env bash
conda activate kgrl

Files="data/trials/minigrid/final_training"

perform_trial(){
  config_file=$1
  file=$(basename -- "$config_file")
  file_name="${file%.*}"
  output_file="$Files/training_output_$file_name.txt"

  tmux new-window -n "$file_name" bash
  tmux send -t trials:"$file_name" "conda activate kgrl;
  python -m kgrl trial $config_file -hls -mg -ci --use-best --opt-storage sqlite:///optuna.db -fp n_episodes 1 -fp deterministic True" ENTER

  echo "done with window creation $file_name"
}

#create a session to store the windows and start top
echo "Starting top session..."
tmux new-session -d -s trials -n htop htop
for f in $Files/*plain.yml
do
  echo $f
  perform_trial $f
done