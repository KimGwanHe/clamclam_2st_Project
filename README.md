# 🚍 하루의 마무리를 위한 clamcalm

![image](https://github.com/user-attachments/assets/6ff1b219-841b-4f82-839f-dedfad6464ca)


<br>

## 프로젝트 소개

- 자연어처리 및 앱 개발 프로젝트
- 하루를 돌아보며 의미있게 마무리하기 위한 그림 일기장
- 정신적인 안정과 만족감을 향상시키기 위한 사용자 일기내용에서 감정에 맞는 ASMR
- 고민상담을 해줄 챗봇 캄캄이
- 눈의 피로를 덜어주기 위한 책을 읽어주는 리딩북

<br>

## 1. 활용 라이브러리

- 문장요약  |  eenzeenee/t5-base-korean-summarization
- 감정분석  |  hun3359/klue-bert-base-sentiment
- 그림생성  |  Bingsu/my-k-anything-v3-0
- 유사도 분석  |  ko-sentence-transformers
- 텍스트 추출  |  NAVER Clova OCR API
- 음성변환  |  pyttsx3 module 

<br>

## 2. 개발 기간 및 작업 관리

### 개발 기간

- 전체 개발 기간 : 2024-06-03 ~ 2024-08-02
- UI 구현 : 2024-06-10 ~ 2024-06-23
- 기능 구현 : 2024-06-24 ~ 2024-07-23
- 테스트 및 수정 : 2024-07-23 ~ 2024-08-01

<br>

## 3. 그림일기장 구성

![image](https://github.com/user-attachments/assets/58396a35-1aa1-4331-be26-bc6049d0fcd9)

## 4. 챗봇 구성

![image](https://github.com/user-attachments/assets/20539919-8cb9-467d-8339-20d32fdeb13e)

## 5. 리딩북 구성

![image](https://github.com/user-attachments/assets/0c7f43e2-4d59-49a8-aa23-7ff3713605ae)


<br>

## 6. 트러블 슈팅(Trouble Shooting)

### 이슈:
- 동화책 이미지에서 텍스트를 추출할 때 발생하는 텍스트 인식률 저하 문제를 해결하기 위해 Clova OCR API를 사용했으나, 배경이 복잡하고 많은 노이즈가 포함된 이미지 특성상 인식률이 낮은 문제점이 있었습니다.
  
### 해결방안:
- HSV 이미지에서 밝기 정보로 명도 채널을 추출하고, OTSU 이진화, 모폴로지 닫기 연산, 컨투어 RETR_EXTERNAL 활용으로 복잡한 배경에서 텍스트 인식률을 높일 수 있었습니다.

<br>

## 7. 실제 화면

<div align="center">
  
### 메인화면
  
![image](https://github.com/user-attachments/assets/588050b8-3aab-4c3e-88ee-f921644688f8)
</div>

<hr>
<div align="center">
  
### 마이페이지

![image](https://github.com/user-attachments/assets/74f2f804-7de4-4333-8ddc-a1fab6636ae5)

그동안 작성했던 일기장 리스트를 작성일과 한줄 요약으로 확인할 수 있습니다.
</div>

<hr>
<div align="center">
  
### 그림일기장

![image](https://github.com/user-attachments/assets/df9a4ba2-844d-44e7-afa4-081f6c07d2aa)

일기 작성 -> 한문장 요약 -> 감정분석 및 이미지생성 -> 감정에 매칭되는 날씨와 멘트 및 asmr -> 일기완성
</div>

<hr>
<div align="center">
  
### 리딩북

![image](https://github.com/user-attachments/assets/bf31943a-41b7-40c3-8c03-2d7992270973)

책 pdf를 업로드 -> 각각의 페이지를 이미지로 인식 -> ocr적용 텍스트 추출 -> 음성파일 변환 -> 재생
</div>

<hr>
<div align="center">
  
### 챗봇

![image](https://github.com/user-attachments/assets/d7891d93-8308-4eed-b365-08c15d386240)

감성공감 챗봇
</div>

<br>
