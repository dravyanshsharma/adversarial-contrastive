# Experimental supplement to [An Analysis of Robustness of Non-Lipschitz Networks](https://arxiv.org/abs/2010.06154)

Code used in the experiments in JMLR 2023 paper "An Analysis of Robustness of Non-Lipschitz Networks" by Nina Balcan (CMU), Avrim Blum (TTIC), Dravyansh Sharma (CMU), and Hongyang Zhang (U Waterloo), authors ordered alphabetically by last name.


## Pre-requisites
- Python
- Pytorch
- CUDA
- numpy

## Overview

We propose the **Random Feature Subspace threat model** to better understand the robustness of non-Lipschitz networks. Our code includes
- Implementation of an exact attack under our threat model (Algorithm 3 in [the paper](https://arxiv.org/abs/2010.06154)).
- Implementation and evaluation of robust error and abstention rates of our proposed abstention-based defense algorithm (Algorithm 1 in [the paper](https://arxiv.org/abs/2010.06154)).

We study our attack and defense in the supervised and self-supervised contrastive learning frameworks, i.e. the feature space under attack corresponds to feature embedding obtained under these frameworks.

Our algorithm is able to simultaneously obtain small robust error as well as small abstention rates, in both supervised and self-supervised contrastive learning settings.

<p align="center">
<img width="700" alt="Screenshot 2023-03-16 at 11 09 25 AM" src="https://user-images.githubusercontent.com/2097750/225660693-04d1702a-5686-4850-9fd1-79508a4885b2.png">
</p>

Furthermore, our theory and experiments suggest that abstention is necessary to defend in our feature space attack model, and standard techniques without abstention are indefensible.

<p align="center">
<img width="700" alt="Screenshot 2023-03-16 at 11 11 07 AM" src="https://user-images.githubusercontent.com/2097750/225661160-9aa79749-9f7e-4f2f-96b4-2e12b4ece33a.png">
</p>


## Summary of files in the repository

- `filter_outliers.py`
	Performs pre-processing step 2 in Algorithm 1.

- `n3_1_robust_accuracy.py`, `dontknow.py`
	Compute robust error and abstention rates for Algorithm 1.

- `n3_1__optimal_attack.py`
	Optimal attack (Algorithm 3) is implemented for n3=1.
	
- `train_simclr.py`
	to train a SimCLR representation

- `contrast\`
	defines the loss, learning rate scheduler, and other details.
	
	
## Reference
For full technical details and experimental results, please check [the paper](https://arxiv.org/abs/2010.06154).

```
@article{balcan2023analysis, 
	author = {Maria-Florina Balcan and Avrim Blum and Dravyansh Sharma and Hongyang Zhang}, 
	title = {An Analysis of Robustness of Non-Lipschitz Networks}, 
	journal = {Journal of Machine Learning Research},
	year = {2023}
}
```


## Contact
dravyans@cs.cmu.edu for further questions related to the paper.
dravyans@cs.cmu.edu or hongyang.zhang@uwaterloo.ca for further questions related to code.
