## 维数灾难
在高维情形下出现的数据样本稀疏、距离计算困难等问题，是所有机器学习方法共同面临的严重障碍， 被称为"维数灾难" (curse ofdimensionality) .  
缓解维数灾难的一个重要途径是降维(dimension red uction) ， 亦称" 维数约简 ，即通过某种数学变换将原始高维属性空间转变为一个低维"子空间" (subspace) ，在这个子空间中样本密度大幅提高， 距离计算也变得更为容易为什么能进行降维?这是因为在很多时候， 人们观测或收集到的数据样本
虽是高维的?但与学习任务密切相关的也许仅是某个低维分布，即高维空间中的一个低维"嵌入" (embedding) . 
***
**对降维效果的评估**，通常是比较降维前后学习器的性能?若性能有所提高
则认为降维起到了作用.若将维数降至二维或三维，则可通过可视化技术来直
观地判断降维效果.

***经典的降维方法:***
+ 1. 多维缩放(Mult iple Dimensional Scaling，简称MDS)
+ 2. 主成分分析(Principal Component Analysis ，简称PCA)_____是一种无监督的线性降维方法
+ 3. 核主成分分析(Kernelized PCA ，简称KPCA)
+ 4. 流形学习(manifold learning)是一类借鉴了拓扑流形概念的降维方法.
     + a. 等度量映射(Isometric Mapping，简称Isomap) 
     + b. 局部线性嵌入(Locally Linear Embedding，简称LLE)
+ 5. 度量学习(metric learning) 
