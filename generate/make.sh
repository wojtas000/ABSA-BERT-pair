# generate datasets
if [ "$1" == 'apa' ]; then
    python generate_${1}_NLI_M.py
    exit 0
fi

python generate_${1}_NLI_M.py
python generate_${1}_QA_M.py
python generate_${1}_NLI_B_QA_B.py
python generate_${1}_BERT_single.py
