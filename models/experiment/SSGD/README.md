# SSGD

origin paper: Single-Side Domain Generalization for Face Anti-Spoofing.

原论文应用图像Deepfake中，下面尝试在音频领域进行实验，看看效果如何。

两种实验设置，分别对域划分进行了不同的定义：
1. 将不同语言的伪造样本划分为一个域
2. 将不同声码器的伪造样本划分为不同的域

## Acroos language domain
数据来源：
1. Wavefake：英文、日语伪造数据
2. FAFMCC：中文伪造数据


## Across vocoder domain
基于Wavefake数据集对模型进行训练，通过对模型做跨库测试验证模型性能。 在这个实验中，保持语言的单一性，即只是用英语的伪造样本。 
因此，Wavefake中有7个伪造样本域，对应于不同的声码器。

目标是让这7个伪造域域能够得到不变域表示空间，而真实域在表示空间自身之间更加紧凑。

使用对抗训练的方式使这7伪造域得到不变域表示空间。

![图片](D:\Code\CollectionModel\N801\models\experiment\SSGD\SSGD.png "总体框架图")





