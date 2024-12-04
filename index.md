---
layout: page
title: GrouPConCeption
subtitle: "Who Walks With You Matters: Perceiving Social Interactions with Groups for Pedestrian Trajectory Prediction"
cover-img: /subassets/img/head_pic.png
---

## Information

This is the homepage of our paper "Who Walks With You Matters: Perceiving Social Interactions with Groups for Pedestrian Trajectory Prediction".

In this work, we propose the [GrouPConCeption](https://github.com/livepoolq/groupconception) model (GPCC) handling group relations and social interactions among agents.
The paper is available at [arXiv](http://arxiv.org/abs/2412.02395).
Click the buttons below for more information.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="http://arxiv.org/abs/2412.02395">📖 Paper</a>
    <!-- <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/SocialCirclePlus">📖 Supplemental Materials (TBA)</a>
    <br><br> -->
    <a class="btn btn-colorful btn-lg" href="https://github.com/livepoolq/groupconception">🛠️ Codes (PyTorch)</a>
    <a class="btn btn-colorful btn-lg" href="/guidelines">💡 Codes Guidelines</a>
    <br><br>
</div>

## Abstract

Understanding and anticipating human movement has become more critical and challenging in diverse applications such as autonomous driving and surveillance. The complex interactions brought by different relations between agents are a crucial reason that poses challenges to this task. Researchers have put much effort into designing a system using rule-based or data-based models to extract and validate the patterns between pedestrian trajectories and these interactions, which has not been adequately addressed yet. Inspired by how humans perceive social interactions with different level of relations to themself, this work proposes the GrouP ConCeption (short for GPCC) model composed of the Group method, which categorizes nearby agents into either group members or non-group members based on a long-term distance kernel function, and the Conception module, which perceives both visual and acoustic information surrounding the target agent. Evaluated across multiple datasets, the GPCC model demonstrates significant improvements in trajectory prediction accuracy, validating its effectiveness in modeling both social and individual dynamics. The qualitative analysis also indicates that the GPCC framework successfully leverages grouping and perception cues human-like intuitively to validate the proposed model's explainability in pedestrian trajectory forecasting.

## Highlights

![groupconception](./subassets/img/fig_method_long.pdf)

- The Group method that divides neighboring agents into the grouping members and ones out of this group based on long-term distance kernel function;
- The Conception module that perceives visual and acoustic information from different regions around the target agent based on vision range of human eyes;
- The validation of the effectiveness and explainability of the proposed GPCC model by evaluating it on different datasets and analyzing how its components function to co-contribute to improve the model's trajectory prediction performance.

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{zou2024who,
  title={Who Walks With You Matters: Perceiving Social Interactions with Groups for Pedestrian Trajectory Prediction},
  author={Zou, Ziqian and Wong, Conghao and Xia, Beihao and Peng, Qinmu and You, Xinge},
  journal={arXiv preprint arXiv:2412.02395},
  year={2024}
}
```

## Contact us

Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com
Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn  
