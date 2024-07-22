# Learning-firmly-nonexpansive-operators

This repository contains the experimental source code to reproduce the numerical experiments in:

* K. Bredies, J. Chirinos Rodriguez, E. Naldi. Learning Firmly Nonexpansive Operators (2024). [ArXiv preprint](https://arxiv.org/abs/2407.14156)

The file "Train_with_ADMM" serves to train the operator and "Denoising_BUTTERFLY" and "Denoising_CIRCLES" perform the experiments in Section 4.5.

The user must install the packages 'yalmp', 'sedumi', 'mptdoc', 'hysdel', 'lcp', 'mpt', 'cddmex' and 'glpkmex' from https://www.tbxmanager.com/.


If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@misc{BCN2024,
      title={Learning Firmly Nonexpansive Operators}, 
      author={Kristian Bredies and Jonathan Chirinos-Rodriguez and Emanuele Naldi},
      year={2024},
      eprint={2407.14156},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2407.14156}, 
}
```


## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
