1. 训练分割模型
    下载 https://cloud.189.cn/web/share?code=UJrmYrFZBzIn
    放入datasets/vesselseg

    # python preprocess_for_segmentation.py # 不用运行，因为代码里有预处理

    cd segmentation

    python train.py

    python test.py

3. 训练分类模型
    下载 https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess?resource=download
    放入datasets/kaggle_eyepacs
    
    python preprocess_for_classification.py

    cd classification

    python train.py -c configs/eyepacs_binary_laddernet_sharpen.yaml 
    # *rgb.yaml为用rgb图训练， *laddernet.yaml为用分割结果训练， *sharpen.yaml为用锐化结果训练, *rgba.yaml为用四通道图训练

    python test.py -c configs/eyepacs_binary_laddernet_sharpen.yaml

4. 检测
    python detection.py
    # 配置参数来自对应的*.yaml