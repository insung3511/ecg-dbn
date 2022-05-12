## ECG DBN Project
ECG Singal data를 받아 DBN 으로 각 Feature를 뽑아내는 인공지능 모델 프로젝트이다. 현 프로젝트는 아래 리뷰 논문에 <a href="https://www.sciencedirect.com/science/article/pii/S2590188520300123?via%3Dihub">GB-DBN을 활용한 논문</a>을 기반으로 진행되고 있다. 

GB-DBN은 RBM를 여러겹 쌓은 Deep Learning Machine이기에 학습을 시키기 위해서는 RBM 모델이 여러개가 필요하다. 즉, 여러개의 RBM 모델을 쌓아 DBN을 만드는 것이기에 여기서 중요한 부분은 확률 분포 모델을 어떤 것을 활용하냐이다. Bernouli 확률 분포는 가장 기본적인 RBM의 확률 분포 모델이고 GB-RBM은 Gaussian 확률 분포를 활용한 모델이다. GB-DBN도 이름에서 알 수 있듯 Gaussian 확률 분포를 활용하였기에 GB-RBM과 BB-RBM를 같이 활용하여 구성된다. 

# Directory
Last Updated : May. 4th. 2022
```
..
├── CHANGES.txt
├── LICENSE
├── RBM.py
├── README.md
├── SVM.py
├── __pycache__
│   ├── RBM.cpython-38.pyc
│   ├── RBM.cpython-39.pyc
│   ├── SVM.cpython-39.pyc
│   └── new_RBM.cpython-39.pyc
├── allnew_model.ipynb
├── arc
│   ├── BBRBM.py
│   ├── GBRBM.py
│   ├── RBM.py
│   └── model.py
├── data
│   ├── __pycache__
│   │   ├── again_median.cpython-39.pyc
│   │   ├── medain_filtering_class.cpython-38.pyc
│   │   ├── medain_filtering_class.cpython-39.pyc
│   │   ├── medfilt.cpython-39.pyc
│   │   ├── read_samples.cpython-39.pyc
│   │   └── so_again_median.cpython-39.pyc
│   ├── db1
│   │   ├── 100.atr
│   │   ├── ...
│   │   ├── SHA256SUMS.txt
│   │   ├── mitdbdir
│   │   │   ├── samples
│   │   │   │   ├── 1001103.xws
│   │   │   │   ├── ...
│   │   │   │   └── 2342317.xws
│   │   │   └── src
│   │   │       ├── makefile
│   │   │       └── printdir.bat
│   │   └── x_mitdb
│   │       ├── ANNOTATORS
│   │       ├── ...
│   │       └── x_234.hea
│   ├── db2
│   │   ├── 200.atr
│   │   ├── ...
│   │   ├── SHA256SUMS.txt
│   │   ├── mitdbdir
│   │   │   ├── samples
│   │   │   │   ├── 1001103.xws
│   │   │   │   ├── ...
│   │   │   │   └── 2342317.xws
│   │   │   └── src
│   │   │       ├── makefile
│   │   │       └── printdir.bat
│   │   ├── record_converter.py
│   │   └── x_mitdb
│   │       ├── ANNOTATORS
│   │       ├── RECORDS
│   │       ├── ...
│   │       └── x_234.hea
│   ├── db3
│   │   ├── svdb
│   │   │   ├── 800.atr
│   │   │   ├── ...
│   │   │   ├── SHA256SUMS.txt
│   │   │   └── result_nst_old
│   │   └── svdb_result
│   │       └── svdb
│   │           ├── rdann_svdb800.csv
│   │           ├── ...
│   │           └── rdsamp_svdb894.csv
│   ├── final_db1
│   │   ├── 101.csv
│   │   ├── ...
│   │   └── 230.csv
│   ├── final_db2
│   │   ├── 100.csv
│   │   ├── ...
│   │   └── 234.csv
│   ├── final_db3
│   │   ├── rdann_svdb800.csv
│   │   ├── ...
│   │   └── rdsamp_svdb894.csv
│   ├── medain_filtering_class.py
│   ├── read_samples.py
│   ├── test
│   │   ├── 800.atr
│   │   ├── ...
│   │   └── 800.xws
│   └── unify
│       ├── 0001.atr
│       ├── 0001.dat
│       ├── 0001.hea
│       ├── aha
│       │   ├── 0001.atr
│       │   ├── ...
│       │   └── 0201.hea
│       ├── convert.py
│       ├── converter.py
│       ├── just_unify.py
│       ├── offical_rdann.py
│       ├── rdann
│       ├── rdann.c
│       ├── rdsamp
│       ├── rdsamp.c
│       ├── record_convert_old.py
│       ├── result
│       │   ├── 0001_rdann.csv
│       │   ├── 0201_rdann.csv
│       │   ├── converted_csv0001.csv
│       │   ├── converted_csv0201.csv
│       │   ├── converted_txt0001.txt
│       │   └── converted_txt0201.txt
│       └── result.zip
├── model.ipynb
├── model.py
├── model_from_ipynb.py
├── new_RBM.py
├── new_mode_fuck.py
├── new_model.ipynb
├── outfile_filtered.csv
├── record_all
│   ├── ahadb
│   │   └── 1.0.0
│   │       ├── 0001.atr
│   │       ├── ...
│   │       └── SHA256SUMS.txt
│   ├── cudb
│   │   └── 1.0.0
│   │       ├── ANNOTATORS
│   │       ├── ...
│   │       └── cu35.xws
│   ├── edb
│   │   └── 1.0.0
│   │       ├── ANNOTATORS
│   │       ├── RECORDS
│   │       ├── ...
│   │       └── edb.txt
│   ├── nstdb
│   │   └── 1.0.0
│   │       ├── 118e00.atr
│   │       ├── ...
│   │       ├── nstdbgen-
│   │       └── old
│   │           ├── 118_02.dat
│   │           ├── ...
│   │           └── oldnstdb.txt
│   ├── rdann
│   ├── ...
│   └── record_converter.py
└── tree.txt

36 directories, 2547 files
```

# Code description
## ./record_all/record_converter.py
ecg-csv repo에 있는 코드를 활용한 것으로 ECG Database 들을 활용하기 위해서는 wfdb 를 활용해야한다. wfdb wave에서 제공해주는 application을 파이썬에서 실행시켜 csv 파일로 변환을 한다. 자세한 코드는 [여기를](https://github.com/insung3511/ecg-csv) 통해 확인이 가능하다.

Update. (May. 12th)
기존에 작성하고 있던 코드에서 아주 크나큰 문제가 있는데 그것은 바로 annotation 즉, 라벨을 하나도 처리하지 않았다는 것이다. 지금까지 받아왔던 Input들은 모두 그저 하나의 Feature들의 집합 밖에 불과 한 것이다. 심지어 지금 생각해보니 Linear SVM은 Test 이후에 들어갈게 아니라 GB-DBN 테스트 이후에 Test로 들어갔어야 했다... 하 모든게 잘못된거 같다. 최대한 빨리 수정을 할 예정이며 코드 수정사항 및 여러 문제 등은 CHANGES.txt에 Logging 할 예정

## ./data/medain_filtering_class.py
지정된 dataset을 pre-processing을 거쳐 신호처리를 하는 코드이다. record_all 디렉토리에 record database가 모두 있으며 앞서 말한 record_converter 파이썬 코드를 통해서 csv로 바꿨다. median_filtering_class (이하 mf.) 는 추출한 csv 파일을 읽어와 논문에서 말한 Pre-processing 방식으로 신호처리를 진행한다.

## ./model.py
파일명에서 추측할 수 있듯이 이는 모델코드이다. 모델 설게를 위해서 쓰이는 코드로 train, test에 활용될 예정이다. 
BBRBM, GBRBM 레이어 쌓음. 학습 진행 후 결과를 보고 마져 진행 예정

## ./RBM.py
이전에는 RBM, GBRBM, BBRBM 을 나누어서 활용했으나 지금은 RBM 모델을 하나 만들고 Bernoulli 확률 분포로 넘겨주어 BBRBM 연산을 할려고 한다. Gaussian 확률 분포로 넘겨주면 GBRBM 으로 연산이 되기에 RBM하나로 왔다갔다 할 예정이다.

# Status
현재 모든 진행 상황은 CHANGES.txt에 기록을 하고 있으며 기록 후 변경 내용이 있을때마다 커밋을 하고 있다. 그렇기에 과거 지워진 내용이나 달라진 내용을 확인하고 싶다면 commit history를 보면 알 수 있다. 
