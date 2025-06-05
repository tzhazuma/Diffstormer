# Diffstormer
 Combined controlled LDM (CLDM) with Restormer/SwinIR/SCUnet/VIT/Resnet/CNN for imaging restoration 
 ###
 A enhanced version (multi stage 1 model) of DiffBIR
 Still under construction
 ***
A modified version of code of the Paper:    
***
## MoCo-Diff: Adaptive Conditional Prior on Diffusion Network for MRI Motion Correction
https://doi.org/10.1007/978-3-031-72089-5_39   

@InProceedings{Li_MoCoDiff_MICCAI2024,
        author = { Li, Feng and Zhou, Zijian and Fang, Yu and Cai, Jiangdong and Wang, Qian},
        title = { { MoCo-Diff: Adaptive Conditional Prior on Diffusion Network for MRI Motion Correction } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15006},
        month = {October},
        page = {411 -- 421}
}
Which based on these codes:
Mainly based on DiffBIR
***
## Controled LDM:
https://github.com/lllyasviel/ControlNet  

@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
  booktitle={IEEE International Conference on Computer Vision (ICCV)}
  year={2023},
}
***
## Restormer
https://github.com/swz30/Restormer  

@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
***
## SwinIR
https://github.com/JingyunLiang/SwinIR  
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}
***
## DiffBIR
https://github.com/XPixelGroup/DiffBIR
@misc{lin2024diffbir,
      title={DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior}, 
      author={Xinqi Lin and Jingwen He and Ziyan Chen and Zhaoyang Lyu and Bo Dai and Fanghua Yu and Wanli Ouyang and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2308.15070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
***
## Real-ESRGAN
https://github.com/xinntao/Real-ESRGAN
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
*** 
## KAIR
https://github.com/cszn/KAIR
***
## SCUNet
https://github.com/cszn/SCUNet
@article{zhang2023practical,
   author = {Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Fan, Deng-Ping and Timofte, Radu and Gool, Luc Van},
   title = {Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis},
   journal = {Machine Intelligence Research},
   DOI = {10.1007/s11633-023-1466-0},
   url = {https://doi.org/10.1007/s11633-023-1466-0},
   volume={20},
   number={6},
   pages={822--836},
   year={2023},
   publisher={Springer}
}
***
## BSRGAN
https://github.com/cszn/BSRGAN
@inproceedings{zhang2021designing,
    title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
    author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
    booktitle={IEEE International Conference on Computer Vision},
    pages={4791--4800},
    year={2021}
}
