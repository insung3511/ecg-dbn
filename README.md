## ECG DBN Project
ECG Singal data를 받아 DBN 으로 각 Feature를 뽑아내는 인공지능 모델 프로젝트이다. 현 프로젝트는 아래 리뷰 논문에 <a href="https://www.sciencedirect.com/science/article/pii/S2590188520300123?via%3Dihub">GB-DBN을 활용한 논문</a>을 기반으로 진행되고 있다. 

GB-DBN은 RBM를 여러겹 쌓은 Deep Learning Machine이기에 학습을 시키기 위해서는 RBM 모델이 여러개가 필요하다. 즉, 여러개의 RBM 모델을 쌓아 DBN을 만드는 것이기에 여기서 중요한 부분은 확률 분포 모델을 어떤 것을 활용하냐이다. Bernouli 확률 분포는 가장 기본적인 RBM의 확률 분포 모델이고 GB-RBM은 Gaussian 확률 분포를 활용한 모델이다. GB-DBN도 이름에서 알 수 있듯 Gaussian 확률 분포를 활용하였기에 GB-RBM과 BB-RBM를 같이 활용하여 구성된다. 
# Code description
## ./record_all/record_converter.py
ecg-csv repo에 있는 코드를 활용한 것으로 ECG Database 들을 활용하기 위해서는 wfdb 를 활용해야한다. wfdb wave에서 제공해주는 application을 파이썬에서 실행시켜 csv 파일로 변환을 한다. 자세한 코드는 [여기를](https://github.com/insung3511/ecg-csv) 통해 확인이 가능하다.

Update. (May. 12th)
기존에 작성하고 있던 코드에서 아주 크나큰 문제가 있는데 그것은 바로 annotation 즉, 라벨을 하나도 처리하지 않았다는 것이다. 지금까지 받아왔던 Input들은 모두 그저 하나의 Feature들의 집합 밖에 불과 한 것이다. 심지어 지금 생각해보니 Linear SVM은 Test 이후에 들어갈게 아니라 GB-DBN 테스트 이후에 Test로 들어갔어야 했다... 하 모든게 잘못된거 같다. 최대한 빨리 수정을 할 예정이며 코드 수정사항 및 여러 문제 등은 CHANGES.txt에 Logging 할 예정

## ./data/read_sample.py
기존 medain_filtering_class.py (a.k.a. mf) 코드에서는 csv 파일을 읽어와 이를 결과를 보여주었지만, 문제가 3번째 Database (SVDB, Test database) 가 제대로 입력되지 못한다는 문제가 있다. 이를 다시 변환하고 다시 읽어오기 위해서 wfdb module를 활용하여 읽어온다. 

## ./allnew_model.ipynb
allnew_model.ipynb 는 다시 새롭게 작성한 모델 코드이다. 거의 대다수의 모델 코드 내용은 이전의 내용과 비슷하지만, 다른 점이 있다면 read_sample (a.k.a. rs) 로 부터 받아온 데이터를 변환을 하는 과정에서 문제가 생겨 이를 다시 맞춰주는 과정이 필요 했다. 또한 라벨링이 되지 않는다는 이슈 등의 문제를 고치고자 새로 제작을 진행하였다.

## ./new_RBM.py
new_RBM.py 는 allnew_model.ipynb에 맞는 새로운 모델이라고 생각하면 쉽다. 기존의 모델은 mf 에 맞춰져 데이터를 읽어왔는데 이 과정에서 Batch size로 변환하는 코드로 인해 다시 맞춰줘야하여 이를 변환하는 과정을 다시 새롭게 작성하였다.

## ./arc/
기존에 사용했던 model, sample reader 등과 같이 쓰이지 않는 코드 들을 모두 넣어둔 아카이브이다. 불필요하나 참고하는데에는 쓰이고 있다.

# Status
현재 모든 진행 상황은 CHANGES.txt에 기록을 하고 있으며 기록 후 변경 내용이 있을때마다 커밋을 하고 있다. 그렇기에 과거 지워진 내용이나 달라진 내용을 확인하고 싶다면 commit history를 보면 알 수 있다. 
