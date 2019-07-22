-Make 8 Type Model
1.FC
2.CONV
3.LSTM
4.CONV+LSTM
5.FC+ADLOSS
6.CONV+ADLOSS
7.LSTM+ADLOSS
8.CONV+LSTM+ADLOSS

-File Information
module.py: header file. It has define value and model(fc, conv, lstm, adloss) function
conv.py: 2.CONV file. It is constructed by CONV+FC

7월 22일 commit 정리
1. adv_conv, adv_conv_lstm, adv_fc, adv_lstm 파일 추가 
2. 파이썬3 에서는 같은 디렉토리안의 모듈을 import할때 name 앞에 . 을 붙여 참조합니다
3. module 에 discriminator model 추가, 마지막 레이어는 sigmoid입니다.
4. 일단 지금 따로 짠거는 각 모델의 인풋값이, X랑 C[0] 랑 C[CELL_SIZE-1]랑 다달라서 이대로 가는게 나을듯
5. adv가 은근 헷갈려서 실행 단계에서 실수할까봐 아예 하위 디렉토리로 옮겼음