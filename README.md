## Make 8 Type Model
1. FC: input_type(101)
2. CONV: input_type(011)
3. LSTM: input_type(101)
4. CONV+LSTM: input_type(011)
5. FC+ADLOSS: input_type(101)
6. CONV+ADLOSS: input_type(111)
7. LSTM+ADLOSS: input_type(101)
8. CONV+LSTM+ADLOSS: input_type(111)

## File Information
- module.py: header file. It has define value and model(fc, conv, lstm, adloss) function
- conv.py: 2.CONV file. It is constructed by CONV+FC

## data type
1. S: FC의 only speed 와 LSTM의 only speed(cell size 추가됨)
2. C
3. E

## 7월 22일 commit 정리
1. adv_conv, adv_conv_lstm, adv_fc, adv_lstm 파일 추가 
2. 파이썬3 에서는 같은 디렉토리안의 모듈을 import할때 name 앞에 . 을 붙여 참조합니다
3. module 에 discriminator model 추가, 마지막 레이어는 sigmoid입니다.
4. 일단 지금 따로 짠거는 각 모델의 인풋값이, X랑 C[0] 랑 C[CELL_SIZE-1]랑 다달라서 이대로 가는게 나을듯
5. adv가 은근 헷갈려서 실행 단계에서 실수할까봐 아예 하위 디렉토리로 옮겼음

## 7월 23일 고려사항 정리
1. 텐서보드 등 비쥬얼라이징
2. 오토 튜닝
3. lstm도 그래프 업데이트 해주면 좋은거 아닌가?
4. conv lstm 은 fc 없이 가기로 한거?
5. C 마지막에 1은 뭐지?

## 7월 25일 commit 정리
1. 텐서보드 연동
2. 아웃풋 파일 만들기 (csv)
3. 데이터 자동 저장 및 선택 복구