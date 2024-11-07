<h2 align="center">DeFT-net</h2>
<p align="center">Dual-Window Extended Frequency Transformer for Rhythmic Motion Prediction rhymthic data period timing annonations of the Human3.6M dataset and dual-windewed attention model source code are provided open source from the <a href="https://carouseldancing.org">CAROUSEL+</a> EU funded FET PROACT project #101017779</p>
<div align="center">

[![Carousel Dancing Discord](https://dcbadge.vercel.app/api/server/eMcjUHN8rQ?style=flat)](https://discord.gg/eMcjUHN8rQ)
[![Twitter Follow](https://img.shields.io/twitter/follow/CarouselDancing.svg?style=social&label=Follow)](https://twitter.com/CarouselDancing)
[![Youtube Subscribe](https://img.shields.io/youtube/channel/subscribers/UCz2rCoDtFlJ4K1yOExu0AWQ?style=social)](https://www.youtube.com/channel/UCz2rCoDtFlJ4K1yOExu0AWQ?sub_confirmation=1)
[![Github Stars](https://img.shields.io/github/stars/CarouselDancing/DeFT-net?style=social)](https://github.com/CarouselDancing/dancegraph/stargazers)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/CarouselDancing/DeFT-net/graphs/commit-activity)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Open in Visual Studio Code](https://img.shields.io/badge/-Open%20in%20VSCode-007acc?logo=Visual+Studio+Code&logoColor=FFFFFF)](https://vscode.dev/github/CarouselDancing/DeFT-net)
[![https://github.com/CarouselDancing/DeFT-net/actions?query=workflow%3AVerify+branch%3Amain)](https://img.shields.io/github/actions/workflow/status/CarouselDancing/DeFT-net/verify.yml?branch=main&logo=github&label=tests)]()
<!--![Lines of code](https://tokei.rs/b1/github/CarouselDancing/DeFT-net)-->
<!--[![Github Downloads (total)](https://img.shields.io/github/downloads/CarouselDancing/DeFT-net/total.svg)](https://github.com/CarouselDancing/DeFT-net/releases)-->
</div>

## Overview

This is the code repo for our paper accepted at CGVC 2024.

## Dependencies 

[![Python](https://img.shields.io/pypi/pyversions/sixteen.svg)](https://badge.fury.io/py/nine)


## Getting the Data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map format can be downloaded from [here](https://drive.google.com/drive/folders/1zTghPRXPl5XTXdJa-L51O67RbpMkUB2Q?usp=sharing).

After downloading, extract actions walking and walking together for S1...11. 

Our re-timed interpolated version of H3.6m dataset in exponential map format for actions walking and walking together can be downloaded from [here](https://drive.google.com/file/d/18FWWw734UyeZJHrP5RHLMiJJ9nCq-oY3/view?usp=sharing).


Dataset Directory Structure 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

```shell script
OurRetimedInterpolated
|-- S1
|   |-- walking_1.txt  
|   |-- walking_2.txt
|   |-- walkingtogether_1.txt  
|   |-- walkingtogether_2.txt
|-- |-- ...
`-- S11
```

##  Training Configuration

All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on Human3.6m datasets and representations.

To train,
## HisRepItselfDCT
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --dataset ./path to H3.6M dataset/
```
## OurRe-timedDCT
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --dataset ./path to OurRetimedInterpolated/
```  
## Dual-windowDCT
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n_run 140 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --dataset ./path to OurRetimedInterpolated/ --model_fold
```

##  Evaluation

## References

D. Sinclair, A. Ademola, B. Koniaris, K. Mitchell: _[DanceGraph: A Complementary Architecture for Synchronous Dancing Online](https://farpeek.com/DanceGraph.pdf)_, 2023 36th International Computer Animation & Social Agents (CASA) . 

Wei Mao, Miaomiao Liu, Mathieu Salzmann. 
[_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://arxiv.org/abs/2007.11755). In ECCV 20.

Ademola, A., Sinclair, D., Koniaris, B., Hannah, S., Mitchell, K. [DeFT-Net: Dual-Window Extended Frequency Transformer for Rhythmic Motion Prediction](https://diglib.eg.org/server/api/core/bitstreams/c5743b51-cdb3-402e-aed8-5788f3a45760/content). In CGVC 2024




