#!/bin/bash

set -e

CONFIG_FILE="/configs/dream.yaml"


BASE_DIR=""


# # Select axioms
python $BASE_DIR/axiom_based_strategy/select_axioms.py --config_file $CONFIG_FILE 

sleep 10

################### 1-4

# direct prove 
python $BASE_DIR/fast_inference_repeated.py --config_file $CONFIG_FILE 


for i in {1..2}
do
    python $BASE_DIR/sub_proposition_label.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/sub_proposition_analysis.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/prove_sp_all_errors.py --config_file $CONFIG_FILE 

done

sleep 10

python $BASE_DIR/sub_proposition_label.py --config_file $CONFIG_FILE

################################ time 5
Alternative strategy
python $BASE_DIR/axiom_based_strategy/generate_strategies.py --config_file $CONFIG_FILE

sleep 10

# Prove with strategy
python $BASE_DIR/proof_with_alt_strategy.py --config_file $CONFIG_FILE

sleep 10
################################# time 6 7

for i in {1..2}
do
    python $BASE_DIR/sub_proposition_label.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/sub_proposition_analysis.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/prove_sp_all_errors.py --config_file $CONFIG_FILE 
    
    sleep 10
done


################################# time 8

python $BASE_DIR/sub_proposition_label.py --config_file $CONFIG_FILE

sleep 10 

# alternative strategy
python $BASE_DIR/axiom_based_strategy/generate_strategies.py --config_file $CONFIG_FILE

sleep 10

python $BASE_DIR/proof_with_alt_strategy.py --config_file $CONFIG_FILE

sleep 10
################################### time 9 
for i in {1..2}
do
    python $BASE_DIR/sub_proposition_label.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/sub_proposition_analysis.py --config_file $CONFIG_FILE

    sleep 10

    python $BASE_DIR/prove_sp_all_errors.py --config_file $CONFIG_FILE 

    sleep 10

done
