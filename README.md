# MAFE: Multi-Agent Fair Environments for Decision-Making Systems

**Accepted to ICML 2026**

<img width="1036" alt="Image" src="[https://github.com/user-attachments/assets/cdd84e20-182b-4820-9965-1595c80a8598](https://github.com/mast-team/MAFE-Multi-agent-Fair-Environments/blob/main/MAFE_Overview.png)" />

## Description

This repository contains the code for the Loan, Healthcare, and Education MAFEs discussed in our paper:

**MAFE: Multi-Agent Fair Environments for Decision-Making Systems**  
Accepted to **ICML 2026**.

## Requirements

Below is a list of the libraries and Python version used to develop this code.

- torch 2.2.2  
- python 3.8.6  
- seaborn  
- matplotlib  
- numpy  
- pandas  
- scipy  
- sklearn

## Training on a MAFE

All training code is contained within the `main.py` file of the respective MAFE. To view options, run:

```bash
python main.py --help
```

For example, run the following command for the Education MAFE:

```bash
python main.py "Education_Environment"
```

## Citation

If you use this work in your research, please cite our preprint available on arXiv:

```bibtex
@article{lazri2025mafe,
  title   = {MAFE: Multi-Agent Fair Environments for Decision-Making Systems},
  author  = {Lazri, Zachary McBride and Nakra, Anirudh and Brugere, Ivan and Dervovic, Danial and Polychroniadou, Antigoni and Huang, Furong and Dachman-Soled, Dana and Wu, Min},
  journal = {arXiv preprint arXiv:2502.18534},
  year    = {2025}
}
```
