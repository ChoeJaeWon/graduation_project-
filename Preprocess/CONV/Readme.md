we need only speed for conv itself
선정된 직전, 직후 m 개의 도로를 각각 입력하시고 xFile에 타겟 도로를 입력하시면 됩니다. 다 only speed로만 하시면 됩니다.
저장되는 것은 타겟 링크의 경로에 맞게 저장경로로 이동됩니다.
sequence가 1이면 conv만 쓰이는 모델을 위한 preprocessing 으로써 preprocess_conv로 이동하고
sequence가 2이상이면 conv와 lstm이 함께 쓰이는 모델을 위한 preprocessing 으로써 preprocess_conv_lstm으로 이동합니다.
입력된 도로의 갯수가 안맞으면 indexing error가 뜹니다.
결과는 입력 순서에 sensitive하니 순서에 주의하세요

errno 는 import 해야하는 python 버전이 있고 아닌 버전이 있습니다.

현제 preprocess version 0.9.3입니다
 - 컨펌되지 않은 버전
 - matrix size 확인 요망
 - 현재 매트릭스 사이즈는 5*timestamp(12) 로 되어있습니다.
 - lstm을 위한 sequence size 는 3으로 되어있습니다.
