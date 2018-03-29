
本工程封装MXNet Rec接口，实现视频I/O读取速度优化.

优化前后性能对比
-----
*mxnet*: 1.0.1\
*network*: BN-Inception\
*machine*: GTX 1080ti\
*data classes*: 101\
*number of frames*: 3f/video\
*frame resolution*: 256x320

| 读取方式            |             单个gpu              | 4个gpu                                  |
| --------------- | :----------------------------: | -------------------------------------- |
| cv2.imread，单线程 |  速度：23.44samples/sec   显存: 9491MB  | 速度： 26.81samples/sec   显存: 12274MB    |
| cv2.imread，多线程（10） |  速度：23.44samples/sec   显存: 9481MB  | 速度： 26.81samples/sec   显存: 12274MB    |
| 优化后    |  速度：120.50 samples/sec   显存: 9497MB | 速度： 464.67 samples/sec   显存: 12298MB |


接口使用姿势
-----

1:本工程未提供截帧程序，视频需要提前截帧，每个视频对应的帧图像存放在同一个文件夹

2:遍历文件夹，生成训练列表文件，文件每行描述一个视频信息，格式为：‘文件夹绝对路径 ，截取的帧图像数，视频所属类别’

3:运行video2rec.py读取txt文件，将图像打包成rec文件

4:运行dataloader.py读取rec文件，并打包成 VideoDataIter


