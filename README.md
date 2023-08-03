## 初始化python virutalenv
```shell
$ python -m venv venv
$ source venv/bin/activate
```

## 安装依赖
```shell
(venv) $ pip install -r requirements.txt
```

## 运行jupyter lab
```shell
(venv) $ export PYTHONPATH=$PWD:$PYTHONPATH
(venv) $ jupyter lab
```

然后在Jupyter lab打开侧边栏的`File Browser`，找到`notebooks`目录，打开`notebooks/dl-data.ipynb`，运行即可。