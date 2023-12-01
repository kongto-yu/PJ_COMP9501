# PJ_COMP9501

## Claim

This repo is the final project of the course COMP9501 open-source code of The University of Hong Kong.   
The creator is Luca Jiangtao Yu, and the collaborator is Niehao Chen.
This repo will be turned private after the grading of the course.

## Introduction
Welcome to the Radio2Speech Project! It is a groundbreaking system that utilizes radio frequency (RF) signals for reconstructing high-quality speech from loudspeakers using mmWave Radar technology. Unlike conventional microphones, this innovative approach is uniquely resistant to noise and capable of penetrating soundproof materials.

## Key features
Advanced Technology: Radio2Speech represents a major leap forward compared to previous techniques, which were confined to simple single-tone music or unclear speech recovery.

RadioGAN: At the core of this system is RadioGAN, adept at accurately reconstructing speech in the time-frequency domain from RF signals, even within a limited frequency range.

Neural Vocoder Integration: The system incorporates a neural vocoder for speech waveform synthesis from the estimated time-frequency representation, effectively bypassing the need to rely on phase data contaminated by noise.

## Performance:
Versatile Efficacy: Extensive evaluations of Radio2Speech have showcased its impressive performance across various settings – from serene to noisy environments and even in soundproof scenarios.

Comparable to Microphones: In quiet environments, Radio2Speech has demonstrated performance on par with traditional microphones, marking a significant achievement.

## How to Run the Code
To begin training the Radio2Speech model, simply execute the following command in your terminal:

```bash
bash train_cgan.sh
```

This command will initiate the model's training process using the predefined configurations and datasets.
