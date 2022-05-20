## ECG DBN Project
ECG Singal data를 받아 DBN 으로 각 Feature를 뽑아내는 인공지능 모델 프로젝트이다. 현 프로젝트는 아래 리뷰 논문에 <a href="https://www.sciencedirect.com/science/article/pii/S2590188520300123?via%3Dihub">GB-DBN을 활용한 논문</a>을 기반으로 진행되고 있다. 

GB-DBN은 RBM를 여러겹 쌓은 Deep Learning Machine이기에 학습을 시키기 위해서는 RBM 모델이 여러개가 필요하다. 즉, 여러개의 RBM 모델을 쌓아 DBN을 만드는 것이기에 여기서 중요한 부분은 확률 분포 모델을 어떤 것을 활용하냐이다. Bernouli 확률 분포는 가장 기본적인 RBM의 확률 분포 모델이고 GB-RBM은 Gaussian 확률 분포를 활용한 모델이다. GB-DBN도 이름에서 알 수 있듯 Gaussian 확률 분포를 활용하였기에 GB-RBM과 BB-RBM를 같이 활용하여 구성된다. 
# Code description
## Model codes
빠르게 작업을 하다보니 현재 디렉토리가 제대로 정리가 되어 있지가 않다. 아마 6월 전에는 한번 정리를 할 것 같으나 현재는 result를 남겨둬야 한다는 정메서 *.ipynb 파일들은 모두 현재 모델 코드이다. 다만 몇몇 설정 값과 모델 구조의 차이가 있으니 참고하길 바란다. 정 안된다면 CHANGES.txt 를 참고하거나 Issue 를 남기면 이에 대해 답변 하겠다. 추가적으로 Input data는 Matlab 추출값으로 진행하고 있다. 같은 연구원 분 께서 정리를 해주어 이를 적용하고 있다. 결과 값은 추후에 한번에 정리 할 예정이다.
## ./arc/
기존에 사용했던 model, sample reader 등과 같이 쓰이지 않는 코드 들을 모두 넣어둔 아카이브이다. 불필요하나 참고하는데에는 쓰이고 있다.

# Status
현재 모든 진행 상황은 CHANGES.txt에 기록을 하고 있으며 기록 후 변경 내용이 있을때마다 커밋을 하고 있다. 그렇기에 과거 지워진 내용이나 달라진 내용을 확인하고 싶다면 commit history를 보면 알 수 있다. 
