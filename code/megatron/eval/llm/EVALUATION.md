## LLM评测

### CEVAL

```Shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir -p data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../
# Default Qwen2.5-7B
python evaluate_ceval.py -d data/ceval/ --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}

# Default Qwen2.5-7B-Instruct (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_ceval.py -d data/ceval/ --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B-Instruct}
```

### MMLU

```Shell
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

# Default Qwen2.5-7B
python evaluate_mmlu.py -d data/mmlu/data/ --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}

# Default Qwen2.5-7B-Instruct (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_mmlu.py -d data/mmlu/data/ --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B-Instruct}
```

### CMMLU

```Shell 
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir -p data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../

# Default Qwen2.5-7B
python evaluate_cmmlu.py -d data/cmmlu/ --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}
```

### HumanEval 

Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

```Shell
git clone https://github.com/openai/human-eval
pip install -e human-eval

pip install jsonlines

cp human-eval/data/HumanEval.jsonl.gz .
gunzip HumanEval.jsonl.gz
# Default Qwen2.5-7B
python evaluate_humaneval.py -f HumanEval.jsonl -o HumanEval_res.jsonl --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}
evaluate_functional_correctness HumanEval_res.jsonl
# Default Qwen2.5-7B-Instruct
python evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res_chat.jsonl  --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}
evaluate_functional_correctness HumanEval_res_chat.jsonl
```

When installing package human-eval, please note its following disclaimer:

This program exists to run untrusted model-generated code. Users are strongly encouraged not to do so outside of a robust security sandbox. The execution call in execution.py is deliberately commented out to ensure users read this disclaimer before running code in a potentially unsafe manner. See the comment in execution.py for more information and instructions.

### GSM8K

```Shell
# Default Qwen2.5-7B
# 从github自动拉取gsm8k的数据进行评测
python evaluate_gsm8k.py --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B}
# 使用git仓库中的gsm8k数据进行评测
python evaluate_gsm8k.py --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B} --sample-input-file gsm8k-local

# Default Qwen2.5-7B-Instruct (5-shot.)
# 从github自动拉取gsm8k的数据进行评测
python evaluate_chat_gsm8k.py --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B-Instruct}
# 使用git仓库中的gsm8k数据进行评测
python evaluate_chat_gsm8k.py --checkpoint-path ${CHECKPOINT_PATH:Qwen/Qwen2.5-7B-Instruct} --sample-input-file gsm8k-local
```