# Towards Advanced Mathematical Reasoning for LLMs via First-Order Logic Theorem Proving

[**📖 Paper**](https://arxiv.org/abs/2506.17104) 


DREAM is a self-adaptive solution that enhances the Diversity and REAsonability of LLMs' generation strategies. DREAM incorporates an Axiom-Driven Strategy Diversification mechanism to promote varied strategic outcomes and a Sub-Proposition Error Feedback to help LLMs reflect on and correct their proofs.

## **What's New?** 

- **🎉 `2024/08/21`:** Our paper has been accepted by EMNLP 2025.


## Setup

Install Lean4 compiler:
```
lake init <Project_name>
lake build
```

## Run the Experiment

Script to run our method: 
```
bash script_dream.sh
```
Note: all scripts that has applied Lean4Verifier.py must be put in the first-order folder. 




## Citation
BibTeX:
```
@misc{cao2025advancedmathematicalreasoningllms,
      title={Towards Advanced Mathematical Reasoning for LLMs via First-Order Logic Theorem Proving}, 
      author={Chuxue Cao and Mengze Li and Juntao Dai and Jinluan Yang and Zijian Zhao and Shengyu Zhang and Weijie Shi and Chengzhong Liu and Sirui Han and Yike Guo},
      year={2025},
      eprint={2506.17104},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.17104}, 
}
```