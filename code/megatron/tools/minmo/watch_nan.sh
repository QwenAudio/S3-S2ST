#!/usr/bin/env bash

ACCESS_TOKEN=60ca6a6f146df435499e7dae2bf3cc155bfdc1279c1402263a295b38e9628a3b   # FunBase
# ACCESS_TOKEN=1a8a11e164e608ca4a312216890c327a7f9c7f2122414575d2b6ccc8adbb03cc   # Test
NUM_LINES=1000
NUM_LINES_WINDOW=100
WARNING_NAN_THRESHOLD=10

exp_dir=$1
exp_name=$2
batch_size=$3
log_index=$4

echo "exp_name: $exp_name"
echo "batch_size: $batch_size"
echo "exp_dir: $exp_dir"
echo "log_index: $log_index"

dir=$(dirname $0)

while true
do
    echo "exp_dir: $exp_dir"
    log_file=$(ls -thr $exp_dir/logs/*/log_${log_index}.txt | tail -1)
    echo "log_file: ${log_file}"
    # num_nan_lines=$(tail -$NUM_LINES "$log_file" | fgrep "loss is nan, set is to 0.0" | wc -l)
    # echo "num_nan_lines: $num_nan_lines"
    current_step=$(tail -$NUM_LINES "$log_file" | fgrep "number of nan iterations:" | tail -$NUM_LINES_WINDOW | grep -Po "(?<=iteration) +\d+" | sed "s/ //g" | sort -n | tail -1)
    current_w_hrs=$(python -c "print(round($current_step * $batch_size * 0.000007678441368384, 2))")

    current_nan_iterations=$(tail -$NUM_LINES "$log_file" | fgrep "number of nan iterations:" | tail -$NUM_LINES_WINDOW | grep -Po "(?<=number of nan iterations:) +\d+" | sed "s/ //g" | sort -n | tail -1)

    lm_loss_sorted=$(tail -$NUM_LINES "$log_file" | fgrep "number of nan iterations:" | tail -$NUM_LINES_WINDOW | grep -Po "(?<=lm_loss:) +[\d\.]+" | sed "s/ //g" | sort -n)
    num_lm_losses=$(echo "$lm_loss_sorted" | wc -l)
    lm_loss_min=$(echo "$lm_loss_sorted" | head -1 | head -c 6)
    lm_loss_max=$(echo "$lm_loss_sorted" | tail -1 | head -c 6)
    lm_loss_avg=$(echo "$lm_loss_sorted" | awk '{sum+=$1} END {print sum/NR}' | head -c 6)

    message="**$exp_name**\n
$(date)\n
**exp_dir** $exp_dir\n
**log_file** ${log_file#$exp_dir/}\n
**current_step** $current_step\n
**current_w_hrs** $current_w_hrs\n
**current_nan_iterations** $current_nan_iterations\n
**lm_loss_min (last $NUM_LINES_WINDOW lines)** $lm_loss_min\n
**lm_loss_max (last $NUM_LINES_WINDOW lines)** $lm_loss_max\n
**lm_loss_avg (last $NUM_LINES_WINDOW lines)** $lm_loss_avg\n"
    if [ $current_nan_iterations -ge $WARNING_NAN_THRESHOLD ]; then
        # add warning message, add 
        message="$message\n# ⚠️WARNING⚠️\nPossible nan loss!!!\nencountered **$current_nan_iterations** nan lines in the last $NUM_LINES lines of the log file"
        title="⚠️WARNING⚠️"
    else
        message="$message\n# ✅OK✅\nNo obvious nan loss in the log file"
        title="✅OK✅"
    fi

    python $dir/send_custom_robot_group_message.py --access_token $ACCESS_TOKEN --title "$title - FunBase training status" --msg "$message"

    sleep 3600
done