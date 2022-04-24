## ECG DBN Project
ECG Singal data를 받아 DBN 으로 각 Feature를 뽑아내는 인공지능 모델 프로젝트이다. 현 프로젝트는 아래 리뷰 논문에 <a href="https://www.sciencedirect.com/science/article/pii/S2590188520300123?via%3Dihub">GB-DBN을 활용한 논문</a>을 기반으로 진행되고 있다. 

GB-DBN은 RBM를 여러겹 쌓은 Deep Learning Machine이기에 학습을 시키기 위해서는 RBM 모델이 여러개가 필요하다. 즉, 여러개의 RBM 모델을 쌓아 DBN을 만드는 것이기에 여기서 중요한 부분은 확률 분포 모델을 어떤 것을 활용하냐이다. Bernouli 확률 분포는 가장 기본적인 RBM의 확률 분포 모델이고 GB-RBM은 Gaussian 확률 분포를 활용한 모델이다. GB-DBN도 이름에서 알 수 있듯 Gaussian 확률 분포를 활용하였기에 GB-RBM과 BB-RBM를 같이 활용하여 구성된다. 

# Directory


# Code description
## ./record_all/record_converter.py
ecg-csv repo에 있는 코드를 활용한 것으로 ECG Database 들을 활용하기 위해서는 wfdb 를 활용해야한다. wfdb wave에서 제공해주는 application을 파이썬에서 실행시켜 csv 파일로 변환을 한다. 자세한 코드는 [여기를](https://github.com/insung3511/ecg-csv) 통해 확인이 가능하다.

## ./data/medain_filtering_class.py
지정된 dataset을 pre-processing을 거쳐 신호처리를 하는 코드이다. record_all 디렉토리에 record database가 모두 있으며 앞서 말한 record_converter 파이썬 코드를 통해서 csv로 바꿨다. median_filtering_class (이하 mf.) 는 추출한 csv 파일을 읽어와 논문에서 말한 Pre-processing 방식으로 신호처리를 진행한다.

## ./model.py
파일명에서 추측할 수 있듯이 이는 모델코드이다. 모델 설게를 위해서 쓰이는 코드로 train, test에 활용될 예정이다. 현재는 아직 개발중

## ./RBM.py ./BBRBM.py ./GBRBM.py
RBM을 구성하기 위한 코드로 RBM을 기반으로 하기에 RBM.py가 RBM 레이어를 만들고 이를 객체로 갖고와 BBRBM에서는 Bernouil 확률 분포로 연산, GBRBM에서는 Gaussian 확률 분포로 연산을 한다. 해당 코드는 <a href="https://github.com/xukai92/pytorch-rbm/blob/ea88786dc8352dae59a4e306ad8fe4d274e13c14/rbm.py">여기에서</a> 갖고 왔으며 개발자가 명시한 라이센스 표준 정책에 따르고 있다.

# Status
현재 모든 진행 상황은 CHANGES.txt에 기록을 하고 있으며 기록 후 변경 내용이 있을때마다 커밋을 하고 있다. 그렇기에 과거 지워진 내용이나 달라진 내용을 확인하고 싶다면 commit history를 보면 알 수 있다. 
