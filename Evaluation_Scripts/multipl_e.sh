for lang in "d" "cpp" "go" "py" "rb" "rs" "swift"
do
    for temperature in 0.2 0.8
    do
        python /kaggle/working/ircoder97/Evaluation_Scripts/code-harness/main.py \
            --model "/kaggle/input/modelgo/" \
            --max_length_generation 1024 \
            --tasks "multiple-$lang" \
            --temperature $temperature \
            --n_samples 50 \
            --precision "fp16" \
            --allow_code_execution \
            --continuous_batching_size 32 \
            --swap_space 128 \
            --save_references_path "/kaggle/working/$modelname/multipl-e/$lang/$temperature/references.json" \
            --save_generations_path "/kaggle/working/$modelname/multipl-e/$lang/$temperature/generations.json" \
            --metric_output_path "/kaggle/working/$modelname/multipl-e/$lang/$temperature/metrics.json" 
    done
done
