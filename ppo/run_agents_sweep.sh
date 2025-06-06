#!/bin/bash

# ここに 出力されたスイープIDを設定
SWEEP_ID="あなたのプロジェクト名/あなたのスイープID"


echo "スイープID: $SWEEP_ID を使用してエージェントを nohup で起動します。"

# GPU 0, 1, 2, 3 を使用してエージェントを起動
# 必要に応じてGPUのリストは変更してください (例: for GPU in 0 1)
for GPU in 0 1 2 3; do
  LOG_FILE="wandb_agent_gpu_${GPU}.log" # 各エージェント用のログファイル名
  echo "GPU $GPU でエージェントを起動します。ログは $LOG_FILE に出力されます。"

  # nohup を使ってバックグラウンドでエージェントを実行し、出力をログファイルにリダイレクト
  nohup CUDA_VISIBLE_DEVICES=$GPU wandb agent $SWEEP_ID > "$LOG_FILE" 2>&1 &

  # nohupが正しくプロセスを開始したか確認するための短い待機（任意）
  sleep 1
done

echo "全てのエージェントの起動を試みました。"
echo "各エージェントのログファイルを確認してください (例: wandb_agent_gpu_0.log)。"
echo "実行中のプロセスを確認するには 'ps aux | grep wandb' を使用してください。"
echo "エージェントを停止するには、上記のコマンドでプロセスIDを特定し、'kill <プロセスID>' を実行してください。"
