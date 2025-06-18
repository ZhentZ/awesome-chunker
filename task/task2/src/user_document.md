**加载container：**

![](data:image/png;base64...)

备注：

1. dockerrun为浪潮公司参照nvidia-docker开发的基于GPU的docker命令。注意没有空格
2. -i tensorflow 为加载映像(images)的名字
3. -v /home/wangleiquan/:/home/wangleiquan/ 将原始路径（前）映射到虚拟环境（后）下
4. -d 0，1 加载GPU。共有4块GPU（0，1，2，3）.注意此-d区别于原始 docker run 命令的-d，完全不是一个东西。根据自己的需求选择0-3，尽可能方便他人少用资源。
5. Permission denied。是ssh相关，没有关系。
6. 镜像是继承了系统的用户。原始用户名为wangleiquan，docker虚拟环境的用户名也为wangleiquan

**重命名containerID**

![](data:image/png;base64...)

备注：建议把containerID重命名下自己可以记住的。docker rename containerid newid

**打开一个已有的docker container**

![](data:image/png;base64...)

备注：

1. 使用docker start -i tfwlq，注意此时用户变成了root
2. 使用nvidia-smi查看可以看到当前container有两块可用的GPU

**如需保存，Commit images**

![](data:image/png;base64...)