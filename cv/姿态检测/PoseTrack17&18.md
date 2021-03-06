# PoseTrack

* 第一部分中内容摘自PoseTrack17&18官网LeaderBoard结果。
* 第二部分中会不定期追加一些近期的仍使用该数据集的较好的工作。

## 原LeaderBoard

### [PoseTrack2017](https://posetrack.net/leaderboard.php)

* 有三个任务：单帧检测、多帧检测、多人跟随。这里摘录后两个的部分。

#### Multi-frame Person Pose Estimation

* 1.DCPose：Dual Consecutiveness Pose Estimator(no paper),2020/8/11.
* 2.TAPose:Temporal Adapter Pose(no paper),2020/9/9
* 3.PoseWarper:Learning Temporal Pose Estimation from Sparsely-Labeled Videos(no paper),2019/7/27.
* 4.HRNet:[Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch),2018/11/7.
* 5.FlowTrack:[FlowTrack](https://arxiv.org/abs/1804.06208[),2018/3/14.
* 6.CorrTrack:CorrTrack(no paper),2020/3/2.
* 7.DetTrack:[Combining detection and tracking for human pose estimation in videos](http://arxiv.org/abs/2003.13743),2020/3/18.
* 8.KeyTrack:[Keypoints Is All You Need](https://arxiv.org/abs/1912.02323),2019/11/21.
* 9.PGPT:[Pose-Guided Tracking-by-Detection: Robust Multi-Person Pose Tracking](https://ieeexplore.ieee.org/document/9034193),2019/3/21.
* 10.POINet:[POINet: Pose-Guided Ovonic Insight Network for Multi-Person Pose Tracking](https://dl_acm.xilesou.top/citation.cfm?id=3350984),2019/4/8.

#### Multi-Person Pose Tracking

* 1.DetTrack:[Combining detection and tracking for human pose estimation in videos](http://arxiv.org/abs/2003.13743),2020/3/18.
* 2.KeyTrack:[Keypoints Is All You Need](https://arxiv.org/abs/1912.02323),2019/11/21.
* 3.Composite Fields(no paper),2020/5/26.
* 4.PGPT:[Pose-Guided Tracking-by-Detection: Robust Multi-Person Pose Tracking](https://ieeexplore.ieee.org/document/9034193),2019/3/21.
* 5.CorrTrack:CorrTrack(no paper),2020/3/2.
* 6.Tracj1(no paper),2020/8/11.
* 7.CNN-based Tracker(no paper),2020.8.10.
* 8.Net1111(no paper),2020/8/11.
* 9.[POINet: Pose-Guided Ovonic Insight Network for Multi-Person Pose Tracking](https://dl_acm.xilesou.top/citation.cfm?id=3350984),2019/4/8.
* 10.[LightTrack online pose tracking](https://github.com/Guanghan/lighttrack),2019/4/14.

### [PoseTrack2018](https://posetrack.net/workshops/eccv2018/posetrack_eccv_2018_results.html)

* 共有两种任务：多人检测和多人跟随。

#### Multi-Person Pose Estimation

* 1.DGDBQ:Bidirectional Pyramid Structure-Aware Network(no paper),2018/8/31.
* 2.ByteCV:Multi-Domain Pose Network(no paper),2018/8/31.
* 3.ALG:Track with re-id(no paper),2018/8/31.
* 4.MSRA:[Simple Baseline for Human Pose Estimation and Tracking](https://github.com/leoxiaobin/pose.pytorch),2018/8/31.
* 5.Miracle:[Multi-person Pose Estimation for Pose Tracking with Enhanced Cascaded Pyramid Network](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Yu_Multi-Person_Pose_Estimation_for_Pose_Tracking_with_Enhanced_Cascaded_Pyramid_ECCVW_2018_paper.pdf),2018.
* 6.openSVAI-S*:[openSVAI: single frame pose estimation](http://guanghan.info/projects/End-to-End-System/),2018/9/25.
* 7.Loomo:nothing at all，2019/5/31.
* 8.E2ES:End-to-End Human Pose Estimation(no link),2018/8/31.
* 9.[MIPAL](https://arxiv.org/abs/1905.09500),2018/8/31.
* 10.SHG:modified stacked hourglass(no link),2018/8/31.

#### Multi-Person Pose Tracking

* 1.MSRA:[Simple Baseline for Human Pose Estimation and Tracking](https://github.com/leoxiaobin/pose.pytorch),2018/8/31.
* 2.ALG:Track with re-id(no paper),2018/8/31.
* 3.Miracle:[Multi-person Pose Estimation for Pose Tracking with Enhanced Cascaded Pyramid Network](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Yu_Multi-Person_Pose_Estimation_for_Pose_Tracking_with_Enhanced_Cascaded_Pyramid_ECCVW_2018_paper.pdf),2018.
* 4.[MIPAL](https://arxiv.org/abs/1905.09500),2018/8/31.
* 5.CV-Human:CV for human pose estimation and tracking(no paper),2018/8/31.
* 6.openSVAI-S*:[openSVAI: single frame pose estimation](http://guanghan.info/projects/End-to-End-System/),2018/9/25.
* 7.CMP:Convolution Machine Pose(no paper),2018/9/1.
* 8.E2E:Online End-to-End Human Pose Estimation and Tracking,2018/8/31
* 9.PR:PoseRefine(no paper),2018.8.31
* 10.Loomo:nothing at all，2019/5/31.

## 新的工作

* PoseWarper：Learning Temporal Pose Estimation from Sparsely-Labeled Videos，NIPS，2019
* Efficient Online Multi-Person 2D Pose Tracking with Recurrent Spatio-Temporal Affinity Fields，CVPR，2019
* Combining detection and tracking for human pose estimation in videos，CVPR，2020
* CorrTrack：Temporal Keypoint Matching and Reﬁnement Network for Pose Estimation and Tracking，ECCV，2020
* Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos，ECCV，2020