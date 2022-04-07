## ECG DBN Project
ECG Singal data를 받아 DBN 으로 각 Feature를 뽑아내는 인공지능 모델 프로젝트이다. 현 프로젝트는 아래 리뷰 논문에 GB-DBN을 활용한 논문을 기반으로 진행되고 있다. <a href="https://www.sciencedirect.com/science/article/pii/S2590188520300123?via%3Dihub">이거요</a>

무튼 현재 진행 중인건 신호 처리 파트이다. 이에 더 궁금한 점이 있다면 직접 논문을 읽어라... 저자는 지금 굉장히 피곤하고 배고픈 상태이다. 왠만해서는 건들지말자. 물거다.

## Code description
median_filter 로 시작하는 파이썬 코드가 굉장히 많은데 이유는 뭐 사연이 깊고 깊다... ~~최종, 진짜찐최종, 찐찐찐찐최종,진짜진심최종, 아몰라최종이겟지.py~~.

무튼 일단 so_again_median_filter.py 가 진짜 최종으로 거의 완성이 되었고 이제 RBM 모델을 쌓아 올려 DBN을 만들어야한다.
