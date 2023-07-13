# README
## 实验介绍
这是一个多模态情感分析实验，包括4000条训练集数据以及511条测试集数据，需要自行划分训练集和验证集。本项目代码参照VistaNet的框架，使用了pytorch搭建了一个文本-图像模态融合情感分析的深度学习模型，验证集准确率达到62.5%
## 文件结构
```
├─checkpoint
├─config_settings
│  ├─config_reducenet.json
│  ├─config_textnet.json
│  ├─config_vggnet.json
│  └─config_vistanet.json
├─data
├─dataloader
│  ├─abstract_dataloader.py
│  └─dataloader.py
├─log
├─models
│  ├─vistanet.py
│  ├─reducenet.py
│  ├─model_text_side.py
│  └─model_img_side.py
├─trainer
│  ├─abstract_trainer.py
│  └─supervised_trainer.py
├─dataset.py
├─evaluator.py
├─quickstart.py
├─README.md
├─specialtokens.py
├─test_without_label.txt
├─train.txt
└─__pycache__
```
## 实验环境
Nvidia RTX2060显卡 5.0GB显存
## 依赖环境安装
可以通过pip install -r requirement.txt安装所有依赖的非python标准库<br>
同时需要安装Cuda11.3版本以适配本项目使用的GPU计算，如果无法使用GPU计算，则可以选择不安装，使用CPU以较慢速度计算。<br>
如在安装依赖环境方面存在问题如`torch`无法使用GPU进行计算，可以查看我的个人博客获取帮助。[^1]
## 使用方法
请在`quickstart.py`文件中输入你希望训练的模型名称，并且修改`config_settings`文件夹下对应的模型参数文件，然后运行`quickstart.py`即可。<br>
已经提供了四个可运行的模型，分别是`vistanet`,`reducenet`,`textnet`,`vggnet`<br>
**注意**：请尽量不要打乱文件的结构，这会影响一些参数的运行。
## 模型介绍
1. `vistanet`：思路来源于文章*VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis*[^2]，但是对其内容进行了修改，保持了图像提取部分使用预训练的VGG不变，语义提取部分引入了预训练的BERT模型，然后使用VistaNet的多模态交叉注意力进行语义提取，最后输出。
2. `reducenet`：思路比较简单，通过一个CNN类模型提取图像信息，再训练一个embedding来提取语义，将两个向量拼接后输出。
3. `textnet`：消融实验使用，仅仅使用BERT预训练模型提取了语义，然后直接将其`flatten`后通过全连接层输出。
4. `vggnet`：消融实验使用，仅仅使用了预训练的VGG模型提取图像特征进行分类。
## 作者
赵艺博 10203330408@stu.ecnu.edu.cn<br>
也可联系 ZhaoYibo624@Gmail.com

[^1]:https://flowus.cn/liltry/28a814b4-9468-44f7-9602-b8f536eed439
[^2]:https://ojs.aaai.org/index.php/AAAI/article/view/3799