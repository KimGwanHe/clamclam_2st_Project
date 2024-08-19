import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import requests
import base64
import re
import aiosmtplib
import sqlalchemy
import datetime
import json
import jwt
import hashlib
import numpy as np
import pandas as pd
import torch
import nltk
import pdfplumber # book
import uuid
import cv2
import logging
import time
import pyttsx3


from fastapi import FastAPI,Request, Depends, HTTPException,Response, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sklearn.metrics.pairwise import cosine_similarity
from flask_mail import Message
from mail_config import app_mail, mail
from email.mime.text import MIMEText
from sqlalchemy import Column, String, Integer, create_engine,ForeignKey,Text,DateTime,func,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from databases import Database
# from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from PIL import Image
from io import BytesIO
from typing import Optional


nltk.download('punkt')
app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
DATABASE_URL = "mySQL 주소"

database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    async with async_session() as session:
        request.state.session = session
        response = await call_next(request)
    return response

Base = declarative_base()

class Member(Base):
    __tablename__ = 'member'
    email = Column(String(length=50), primary_key=True, index=True)
    userpw = Column(String(length=100),nullable=False)
    birth = Column(String(length=50))

class Intro(Base):
    __tablename__ = 'intro'
    email = Column(String(100), ForeignKey('member.email'), primary_key=True, index=True)
    name = Column(String(length=30),nullable=False)
    keywords = Column(String(length=50),nullable=False)
    img = Column(Text, nullable=False)

class Chat(Base):
    __tablename__ = 'chat'
    email = Column(String(100), ForeignKey('member.email'), primary_key=True, index=True)
    chatting = Column(Text, nullable=False)
    createdat = Column(DateTime, server_default=func.now())
    updatedat = Column(DateTime, server_default=func.now(), onupdate=func.now())

class Diary(Base):
    __tablename__ = 'diary'
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), ForeignKey('member.email'), index=True)
    content = Column(Text, nullable=False)
    ment = Column(Text, nullable=False)
    asmr = Column(Text, nullable=False)
    img = Column(Text, nullable=False)
    createdat = Column(DateTime, server_default=func.now())

engine = create_async_engine(DATABASE_URL, echo=True, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
session = async_session()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

#----------------------------------------------------------------------------------------
# 동화책 읽어주기
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
IMAGE_FOLDER = 'public/img'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CLOVA OCR API 정보
api_url = 'CLOVA OCR API 주소'
secret_key = 'CLOVA OCR API 키'

def clova_ocr(image):
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    with BytesIO() as output:
        image.save(output, format='JPEG')
        image_data = output.getvalue()

    files = [
        ('file', ('image.jpg', image_data, 'image/jpeg'))
    ]

    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
        
    if response.status_code != 200:
        logging.error(f"OCR 요청 실패: {response.status_code} {response.text}")
        response.raise_for_status()
    
    result = response.json()
    if 'images' not in result:
        raise HTTPException(status_code=500, detail="OCR 응답에 이미지 정보가 없습니다.")
    return result

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    blurred_v = cv2.GaussianBlur(v, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 10 and cv2.contourArea(contour) > 100:
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))

def text_extraction(ocr_result):
    extracted_text = []
    for image in ocr_result.get('images', []):
        for field in image.get('fields', []):
            text = field.get('inferText')
            # text = re.sub(r'\s+', ' ', text)
            korean_text = re.sub(r'[^가-힣\s\.\?\!]', '', text)
            korean_text = re.sub(r'\s+', ' ', korean_text)
            extracted_text.append(korean_text)
    return extracted_text

def pdf_to_images(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pil_image = page.to_image().original
            images.append(pil_image)
    return images




#----------------------------------------------------------------------------------------
# 일기장에 필요한 라벨링 및 멘트들
sen_lableling ={
    "0": "분노",
    "1": "툴툴대는",
    "2": "좌절한",
    "3": "짜증내는",
    "4": "방어적인",
    "5": "악의적인",
    "6": "안달하는",
    "7": "구역질 나는",
    "8": "노여워하는",
    "9": "성가신",
    "10": "슬픔",
    "11": "실망한",
    "12": "비통한",
    "13": "후회되는",
    "14": "우울한",
    "15": "마비된",
    "16": "염세적인",
    "17": "눈물이 나는",
    "18": "낙담한",
    "19": "환멸을 느끼는",
    "20": "불안",
    "21": "두려운",
    "22": "스트레스 받는",
    "23": "취약한",
    "24": "혼란스러운",
    "25": "당혹스러운",
    "26": "회의적인",
    "27": "걱정스러운",
    "28": "조심스러운",
    "29": "초조한",
    "30": "상처",
    "31": "질투하는",
    "32": "배신당한",
    "33": "고립된",
    "34": "충격 받은",
    "35": "가난한 불우한",
    "36": "희생된",
    "37": "억울한",
    "38": "괴로워하는",
    "39": "버려진",
    "40": "당황",
    "41": "고립된(당황한)",
    "42": "남의 시선을 의식하는",
    "43": "외로운",
    "44": "열등감",
    "45": "죄책감의",
    "46": "부끄러운",
    "47": "혐오스러운",
    "48": "한심한",
    "49": "혼란스러운(당황한)",
    "50": "기쁨",
    "51": "감사하는",
    "52": "신뢰하는",
    "53": "편안한",
    "54": "만족스러운",
    "55": "흥분",
    "56": "느긋",
    "57": "안도",
    "58": "신이 난",
    "59": "자신하는"
  }

weather_lablelings ={
    0: "천둥번개치는",
    1: "건조한",
    2: "천둥번개치는",
    3: "천둥번개치는",
    4: "쌀쌀한",
    5: "고온다습한",
    6: "고온다습한",
    7: "천둥번개치는",
    8: "천둥번개치는",
    9: "고온다습한",
    10: "비 내리는",
    11: "비 내리는",
    12: "비 내리는",
    13: "흐린",
    14: "비 내리는",
    15: "쌀쌀한",
    16: "건조한",
    17: "비 내리는",
    18: "비 내리는",
    19: "고온다습한",
    20: "바람부는",
    21: "바람부는",
    22: "고온다습한",
    23: "눈 내리는",
    24: "바람부는",
    25: "바람부는",
    26: "건조한",
    27: "바람부는",
    28: "바람부는",
    29: "바람부는",
    30: "쌀쌀한",
    31: "쌀쌀한",
    32: "쌀쌀한",
    33: "건조한",
    34: "천둥번개치는",
    35: "비 내리는",
    36: "비 내리는",
    37: "비 내리는",
    38: "쌀쌀한",
    39: "건조한",
    40: "바람부는",
    41: "건조한",
    42: "흐린",
    43: "눈 내리는",
    44: "고온다습한",
    45: "흐린",
    46: "흐린",
    47: "고온다습한",
    48: "건조한",
    49: "바람부는",
    50: "맑은",
    51: "맑은",
    52: "맑은",
    53: "선선한",
    54: "선선한",
    55: "맑은",
    56: "선선한",
    57: "선선한",
    58: "맑은",
    59: "맑은"
  }
asmr = {
    '맑은':'https://www.youtube.com/watch?v=1bQk7uSfg-w',
    '흐린':' https://www.youtube.com/watch?v=SGj0AZcicPA',
    '비 내리는' : 'https://www.youtube.com/watch?v=xZbjgsE-Vlo',
    '눈 내리는' : 'https://www.youtube.com/watch?v=pOq7DvkGPWA',
    '천둥번개치는' : 'https://www.youtube.com/watch?v=AUjrHzjoxXs',
    '건조한' : 'https://www.youtube.com/watch?v=dulUH5RAHGg',
    '바람부는' : 'https://www.youtube.com/watch?v=p2fxv3PAtLU',
    '선선한' : 'https://www.youtube.com/watch?v=595mTG7KyRs',
    '고온다습한' : 'https://www.youtube.com/watch?v=lQ0fS2meTYQ',
    '쌀쌀한' : 'https://www.youtube.com/watch?v=N_g3AiXF-q8'
    }

ments = {
    '맑은':'오늘은 활기찬 하루를 보냈군요! 수면여행 비행기 ASMR을 들으면서 잠에 들어보세요.',
    '흐린':' 심란한 하루를 빨리 잊고 편안히 취침할 수 있도록 부드러운 ASMR 추천드립니다',
    '비 내리는' : '오늘은 슬픈 날이었군요. 잔잔한 재즈음악을 들으면서 조금이나마 마음이 편안하지길 바랍니다.',
    '눈 내리는' : '오늘같은 외로웠던 날은 부드럽게 다가오는 파도 소리를 들으며 마음의 위안을 찾아보세요',
    '천둥번개치는' : '힘든 하루를 보낸 오늘, 부드러운 피아노 선율이 당신의 마음을 편안하게 해줄 거예요',
    '건조한' : '건조한 하루를 보낸 당신을 위한 물소리가 마음을 촉촉하게 적셔줄거예요',
    '바람부는' : '불안했던 하루를 잘 마무리하기 위해 잔잔한 수면음악을 추천드립니다. ',
    '선선한' : '오늘 같은 날에는 선선한 하루의 연장선으로 기분 좋게 잘 수 있는 바람 부는 나뭇잎 ASMR을 추천드립니다',
    '고온다습한' : '마음이 답답했던 하루엔 시원한 빗소리로 마음의 답답함을 해소해보세요',
    '쌀쌀한' : '얼어붙은 날엔 장작불 ASMR이 당신의 마음에 따뜻함을 가져다줄 거예요'
}

#----------------------------------------------------------------------------------------
#일기장에 필요한 모델 및 함수들

# repo = "Bingsu/my-korean-stable-diffusion-v1-5"
repo = "Bingsu/my-k-anything-v3-0"
euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float16
)
pipe.to(device)
pipe.safety_checker = None

sum_model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
sum_tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')

sen_model_name = "hun3359/klue-bert-base-sentiment"
sen_tokenizer = BertTokenizer.from_pretrained(sen_model_name)
sen_model = BertForSequenceClassification.from_pretrained(sen_model_name)

# def mk_img(prompt):
#   seed = 23957
#   generator = torch.Generator("cuda").manual_seed(seed)
#   image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
#   return image

def mk_img(
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    scale: float = 7.5,
    steps: int = 30,
):
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        guidance_scale=scale,
        num_inference_steps=steps,
    ).images[0]

    return image

def summarize(text):
  prefix = "summarize: "
  inputs = [prefix + text]
  inputs = sum_tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
  output = sum_model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=64)
  decoded_output = sum_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
  result = nltk.sent_tokenize(decoded_output.strip())[0]
  return result

def sentiment(text):
  sentiment_model = pipeline("sentiment-analysis", model=sen_model, tokenizer=sen_tokenizer)
  result = sentiment_model(text)
  return result

def mk_diary(text,sen_lableling):
  summary = summarize(text)
  img = mk_img(summary)
  senti = sentiment(summary)
  sen_lablelings = {v: k for k, v in sen_lableling.items()}
  sent = senti[0]['label']
  label = sen_lablelings[sent]
  weather = weather_lablelings[int(label)]
  asmr_url = asmr[weather]
  ment = ments[weather]
  return text,weather,asmr_url,ment,img
#----------------------------------------------------------------------------------------

# SentenceBERT 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# CSV 파일 경로 설정
csv_file_path = 'total_data_clean1.csv'

# 데이터 로드 및 전처리
data = pd.read_csv(csv_file_path)

# data['embedding']=data['embedding'].astype('Float64')

# 저장된 임베딩이 리스트 형태로 저장된 경우, 배열로 변환
data['embedding'] = data['embedding'].apply(lambda x: np.array(eval(x)))


class SentenceRequest(BaseModel):
    sentence: str
    history: list

class DiaryEntry(BaseModel):
    email: str
    content: str
    weather: str
    ment: str
    asmr: str
    img: str

@app.post("/chat")
def calculate_similarity(request: SentenceRequest):
    input_sentence = request.sentence
    input_embedding = model.encode(input_sentence)   
    # 유사도 계산
    data['similarity'] = data['embedding'].map(lambda x: cosine_similarity( [input_embedding],[x]).squeeze())
    
    # 가장 유사한 문장 찾기
    similar_sentence = data.loc[data['similarity'].idxmax()]
    result = {
        'question': similar_sentence['question'],
        'answer': similar_sentence['answer'],
        'similarity': similar_sentence['similarity'],
        'history': request.history  # 최신 대화 히스토리 반환
    }
    return result
    

#----------------------------------------------------------------------------------------


async def send_email(recipient, subject, body):
    message = MIMEText(body)
    message["From"] = "보낼 이메일"
    message["To"] = recipient
    message["Subject"] = subject

    smtp_client = aiosmtplib.SMTP(
        hostname="smtp.gmail.com",
        port=587,
        start_tls=True,
        username="이메일",
        password="패스워드",
    )

    await smtp_client.connect()
    await smtp_client.send_message(message)
    await smtp_client.quit()


#----------------------------------------------------------------------------------------

#http://127.0.0.1
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("index.html")

@app.get("/index.html", response_class=FileResponse)
async def root():
    return FileResponse("index.html")

#-----------------------------------------------------------------------------------------

@app.get('/user/{address}/{num}', response_class=HTMLResponse)
async def send_mail(address: str, num: str):
    subject = '캄캄 인증메일입니다'
    body = f'인증번호는 {num}입니다 입력란에 입력해주세요.'
    await send_email(address, subject, body)
    
    return HTMLResponse(
        content="<script>alert('인증번호 전송이 완료되었습니다.');</script>",
        status_code=200
    )

@app.post('/user/regist', response_class=HTMLResponse)
async def regist(request: Request):
    body = await request.body()
    body_ = body.decode('utf-8').split('&')
    email = body_[0].replace('email=','')
    password = body_[2].replace('password=','')
    birth = body_[4].replace('birth=','')
    query = Member.__table__.insert().values(email=email, userpw=password,birth=birth)
    last_record_id = await database.execute(query)
    return HTMLResponse(
        content="<script>alert('회원가입이 완료되었습니다.');location.href='../../index.html';</script>",
        status_code=200
    )

@app.get('/user/{email}' , response_class=JSONResponse)
async def dul_e(email:str):
    async with async_session() as session:
        email_ = email.replace('@', '%40')
        query = text('SELECT * FROM member WHERE email = :email')
        result = await session.execute(query, {"email": email_})
        member = result.scalar_one_or_none()
        if member == None:
            return JSONResponse({"exists": True})
        else:
            return JSONResponse({"exists": False})
#-----------------------------------------------------------------------------------------    
# 일기장 함수

@app.get('/diary/{text}',response_class=JSONResponse)   
async def make_diary(text:str):
    print(text)
    text,weather,asmr_url,ment,img = mk_diary(text,sen_lableling)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    encoded_img = base64.b64encode(img_byte_arr).decode("utf-8")
    data = {'text':text,'weather':weather,'asmr_url':asmr_url,'ment':ment,'img':encoded_img}
    return JSONResponse(content=data)

#일기저장 함수
@app.post("/diaries", response_class=HTMLResponse)
async def create_diary(entry: DiaryEntry):
    # body = await request.body()
    # body = json.loads(body)
    email = entry.email
    content = entry.content
    ment = entry.ment
    asmr = entry.asmr
    img = entry.img
    query = Diary.__table__.insert().values(email=email, content=content,ment=ment,asmr=asmr,img=img)
    last_record_id = await database.execute(query)
    return HTMLResponse(
        content="<script>alert('일기저장이 완료되었습니다.');location.href='../../index.html';</script>",
        status_code=200
    )

#이메일로 일기 탐색
@app.get("/diaries/email/{email}", response_class=JSONResponse)
async def read_diary_all(email: str):
    email_ = email.replace('@','%40')
    print(email_)
    query = Diary.__table__.select().where(Diary.email == email_)
    db_diary = await database.fetch_all(query)
    if db_diary is None:
        raise HTTPException(status_code=404, detail="Diary not found")
    return db_diary


#아이디로 일기탐색함수
@app.get("/diaries/id/{diary_id}", response_class=JSONResponse)
async def read_diary_one(diary_id: int):
    print(diary_id)
    query = Diary.__table__.select().where(Diary.id == diary_id)
    db_diary = await database.fetch_one(query)
    if db_diary is None:
        raise HTTPException(status_code=404, detail="Diary not found")
    return db_diary

#-----------------------------------------------------------------------------------------

# 비밀번호 해시 검증을 위한 객체 생성
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "사용할 패스워드"  # 이 비밀 키는 안전하게 보관하세요
ALGORITHM = "HS256"  # 사용할 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 토큰의 만료 시간 (분 단위)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post('/user/login', response_class=HTMLResponse)
async def login(request: Request):
    body = await request.body()
    # JSON 문자열 파싱
    data = json.loads(body.decode('utf-8').strip('[]'))
    email = data.get('email', '').replace('@', '%40')
    password = data.get('password', '')
    print(email,password)

    # 데이터베이스에서 사용자 정보 조회
    async with async_session() as session:
        # query = Member.__table__.select().where(Member.email == email)
        result = await session.execute(
                sqlalchemy.select(Member).where(Member.email == email)
            )
        member = result.scalar_one_or_none()
      
        if member:
        # 회원이 존재하고 비밀번호가 일치하는지 확인
        # if member and password == member.userpw:
            if member and (password == member.userpw):
                # 로그인 성공 - JWT 토큰 생성
                access_token = create_access_token(data={"sub": email})
                print(access_token)
                return JSONResponse(content={
                    "message": "로그인 성공",
                    "access_token": access_token,
                    "token_type": "bearer"
                })
        # 로그인 실패
        # raise HTTPException(status_code=401, detail="아이디 또는 비밀번호를 확인해주세요")  
        return HTMLResponse(
        content="<script>alert('아이디 또는 비밀번호를 확인해주세요.');</script>",
        status_code=401)

# 로그아웃 엔드포인트
@app.post('/user/logout', response_class=JSONResponse)
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}   

# 회원 탈퇴 엔드포인트
@app.delete('/user/withdraw', response_class=JSONResponse)
async def withdraw(request: Request):
    data = await request.json()
    email = data.get('email', '').replace('@', '%40')
    async with async_session() as session:
        # 사용자 조회
        async with async_session() as session:
            query = delete(Member).where(Member.email == email)
            result = await session.execute(query)
            await session.commit()
        return {"message": f"사용자 {email}이(가) 성공적으로 탈퇴되었습니다."}

#---------------------------------------------------------------------------

# 리딩북
@app.post('/upload', response_class=JSONResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        logger.error("파일 타입이 맞지 않습니다.")
        return JSONResponse({'error': '파일 타입이 맞지 않습니다.'}, status_code=400)

    filename = file.filename
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        logger.info(f"PDF 파일 저장 경로: {pdf_path}")
        with open(pdf_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        logger.info("파일이 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"파일 저장 중 오류 발생: {str(e)}")
        return JSONResponse({'error': f'파일 저장 중 오류 발생: {str(e)}'}, status_code=500)

    try:
        images = pdf_to_images(pdf_path)
        logger.info(f"PDF에서 추출된 이미지 수: {len(images)}")
        full_text = ""

        # 첫 페이지 이미지를 저장
        first_page_image = images[0]
        first_page_image_path = os.path.join(IMAGE_FOLDER, f"{uuid.uuid4()}.jpg")
        first_page_image.save(first_page_image_path)

        for i, image in enumerate(images[2:], start=2):
            processed_image = preprocess_image(np.array(image))
            ocr_result = clova_ocr(processed_image)
            extracted_text = text_extraction(ocr_result)
            full_text += " ".join(extracted_text) + " "

        # 여러 개의 공백을 하나로
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        text_filename = os.path.splitext(filename)[0] + '.txt'

        # 텍스트를 음성으로 변환 (pyttsx3)
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)  # 속도 설정
        voices = engine.getProperty('voices')
        # 목소리 목록 출력
        voices = engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"Voice {i}: Name: {voice.name}, ID: {voice.id}, Gender: {'male' if 'male' in voice.name.lower() else 'female'}")
        for voice in voices:
            if 'male' in voice.name.lower():  # 목소리 선택
                engine.setProperty('voice', voice.id)
                break
        audio_filename = os.path.splitext(filename)[0] + '.mp3'
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        engine.save_to_file(full_text, audio_path)
        engine.runAndWait()

        logger.info(f"오디오 파일 저장 경로: {audio_path}")
        return JSONResponse({
            'filename': text_filename,
            'text': full_text,
            'audio_filename': audio_filename,
            'image_filename': os.path.basename(first_page_image_path)  # 이미지 파일명 반환
        }, status_code=200)

    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {str(e)}")
        return JSONResponse({'error': f'파일 처리 중 오류 발생: {str(e)}'}, status_code=500)
    
@app.get("/audio/{audio_filename}", response_class=FileResponse)
async def get_audio(audio_filename: str):
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(audio_path)



#-----------------------------------------------------------------------------------------



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)