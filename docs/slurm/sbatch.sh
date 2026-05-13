#!/bin/bash
#SBATCH --job-name=SPAS
#SBATCH --output=revise/mps_%j.out
#SBATCH --error=revise/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2 # Không khai báo GPU; --gres=mps:l40:2 (card L40); --gres=mps:a100:2 (card A100, ưu tiên các job dùng > 40GB vRAM)
#SBATCH --mem=4G
#SBATCH --time=72:00:00
REQUIRED_VRAM=72000  # Quan trọng - Số vRAM cần dùng (để tìm GPU phù hợp)
# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
# *** Kích hoạt venv (Sửa đường dẫn / môi trường theo user)
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate deepseek_vl2
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES
# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM, nếu không tìn thấy GPU đủ vRAM thì hàm CHECK_OUT sẽ đưa job vào lại hàng đợi để chờ tìm slot khác; sau 5 lần requeue mà vẫn chưa tìm được slot thì sẽ trả về mã lỗi để kết thúc job)
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"


export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU 

# =========================================================
# CHẠY CODE
# =========================================================


# Run NSGA-II based attack
python run_attack.py \
    --sample_path run_half_1.txt \              # Sample IDs to attack
    --result_clean_dir result_clip \ # Clean retrieval results directory
    --reader_name llava-one \               # Model options: llava-one, deepseekvl2, qwenvl2.5
    --retriever_name blip \                 # Model options: clip, blip
    --w 312 --h 312 \                      # Image dimensions
    --pop_size 20 \                        # NSGA-II population size
    --F 0.9 \                              # Mutation weight
    --n_k 1 \                              # Target position to attack (top-k)
    --max_iter 100 \                       # Maximum iterations
    --std 0.05                             # Perturbation magnitude



# python run_score_for_each_topk.py \
#     --n_k 5 \
#     --retriever_name clip \
#     --reader_name deepseek-vl2-tiny \
#     --train_reader llava-one \
#     --std 0.05 \
#     --run_path run.txt \
#     --sample_path run.txt \
#     --attack_result_path "attack_result_usingquestion=1/clip_llava-one_0.05" \
#     --result_clean_dir "result_usingquery=0_clip" \
#     --using_question 1 \
#     --method nsga2 \
#     --llm gemini \
#     --target_answer "golden_answer" \
#     --mode "all"



# python run_score_for_each_topk.py \
#     --n_k 5 \
#     --retriever_name clip \
#     --reader_name deepseek-vl2-tiny \
#     --train_reader llava-next \
#     --std 0.05 \
#     --run_path run_half_2.txt \
#     --sample_path run_half_2.txt \
#     --attack_result_path "attack_result_usingquestion=1/clip_llava-next_0.05" \
#     --result_clean_dir "result_usingquery=0_clip" \
#     --using_question 1 \
#     --method nsga2 \
#     --llm gemini \
#     --target_answer "golden_answer" \
#     --mode "all"


# python run_score_for_each_topk.py \
#     --n_k 5 \
#     --retriever_name clip \
#     --reader_name deepseek-vl2-tiny \
#     --train_reader qwenvl2.5 \
#     --std 0.05 \
#     --run_path run.txt \
#     --sample_path run.txt \
#     --attack_result_path "attack_result_usingquestion=1/clip_qwenvl2.5_0.05" \
#     --result_clean_dir "result_usingquery=0_clip" \
#     --using_question 1 \
#     --method nsga2 \
#     --llm gemini \
#     --target_answer "golden_answer" \
#     --mode "all"