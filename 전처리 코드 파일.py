%config InlineBackend.figure_format = "retina"
import warnings
warnings.filterwarnings(action='ignore') #경고 메시지 무시
from IPython.display import display #print가 아닌 display()로 연속 출력
from IPython.display import HTML #출력 결과를 HTML로 생성

import pandas as pd
from collections import OrderedDict
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import shapely
from shapely.geometry import Point
import sys
from pyproj import Proj, transform
%matplotlib inline
import folium
import os
import geopandas as gpd

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')

# 데이터 불러오기

df = pd.read_csv("/content/drive/MyDrive/통계분석및실습/2014.10.05~2024.10.05 날씨데이터 V2.csv", encoding='cp949')

df.head()

# 전처리 및 EDA

칼럼 괄호 및 단위 제거

df.columns = df.columns.str.replace(r'\(.*?\)', '', regex=True)  # 괄호와 그 안의 내용을 제거
df.columns = df.columns.str.replace(' ', '')  # 띄어쓰기 제거

df

시도, 시군구 칼럼 추가

# 시도 정보 딕셔너리
sido_dict = {
    '속초': '강원도',
    '북춘천': '강원도',
    '철원': '강원도',
    '동두천': '경기도',
    '파주': '경기도',
    '대관령': '강원도',
    '춘천': '강원도',
    '백령도': '인천광역시',
    '북강릉': '강원도',
    '강릉': '강원도',
    '동해': '강원도',
    '서울': '서울특별시',
    '인천': '인천광역시',
    '원주': '강원도',
    '울릉도': '경상북도',
    '수원': '경기도',
    '영월': '강원도',
    '충주': '충청북도',
    '서산': '충청남도',
    '울진': '경상북도',
    '청주': '충청북도',
    '대전': '대전광역시',
    '추풍령': '충청북도',
    '안동': '경상북도',
    '상주': '경상북도',
    '포항': '경상북도',
    '군산': '전라북도',
    '대구': '대구광역시',
    '전주': '전라북도',
    '울산': '울산광역시',
    '창원': '경상남도',
    '광주': '광주광역시',
    '부산': '부산광역시',
    '통영': '경상남도',
    '목포': '전라남도',
    '여수': '전라남도',
    '흑산도': '전라남도',
    '완도': '전라남도',
    '고창': '전라북도',
    '순천': '전라남도',
    '진도(첨찰산)': '전라남도',
    '대구(기)': '대구광역시',
    '홍성': '충청남도',
    '서청주': '충청북도',
    '제주': '제주특별자치도',
    '고산': '제주특별자치도',
    '성산': '제주특별자치도',
    '서귀포': '제주특별자치도',
    '진주': '경상남도',
    '강화': '인천광역시',
    '양평': '경기도',
    '이천': '경기도',
    '인제': '강원도',
    '홍천': '강원도',
    '태백': '강원도',
    '정선군': '강원도',
    '제천': '충청북도',
    '보은': '충청북도',
    '천안': '충청남도',
    '보령': '충청남도',
    '부여': '충청남도',
    '금산': '충청남도',
    '세종': '세종특별자치시',
    '부안': '전라북도',
    '임실': '전라북도',
    '정읍': '전라북도',
    '남원': '전라북도',
    '장수': '전라북도',
    '고창군': '전라북도',
    '영광군': '전라남도',
    '김해시': '경상남도',
    '순창군': '전라북도',
    '북창원': '경상남도',
    '양산시': '경상남도',
    '보성군': '전라남도',
    '강진군': '전라남도',
    '장흥': '전라남도',
    '해남': '전라남도',
    '고흥': '전라남도',
    '의령군': '경상남도',
    '함양군': '경상남도',
    '광양시': '전라남도',
    '진도군': '전라남도',
    '봉화': '경상북도',
    '영주': '경상북도',
    '문경': '경상북도',
    '청송군': '경상북도',
    '영덕': '경상북도',
    '의성': '경상북도',
    '구미': '경상북도',
    '영천': '경상북도',
    '경주시': '경상북도',
    '거창': '경상남도',
    '합천': '경상남도',
    '밀양': '경상남도',
    '산청': '경상남도',
    '거제': '경상남도',
    '남해': '경상남도',
    '북부산': '부산광역시',
}

# 시군구 정보 딕셔너리
sigungu_dict = {
    '속초': '속초시',
    '북춘천': '춘천시',
    '철원': '철원군',
    '동두천': '동두천시',
    '파주': '파주시',
    '대관령': '평창군',
    '춘천': '춘천시',
    '백령도': '옹진군',
    '북강릉': '강릉시',
    '강릉': '강릉시',
    '동해': '동해시',
    '서울': '서울특별시',
    '인천': '인천광역시',
    '원주': '원주시',
    '울릉도': '울릉군',
    '수원': '수원시',
    '영월': '영월군',
    '충주': '충주시',
    '서산': '서산시',
    '울진': '울진군',
    '청주': '청주시',
    '대전': '대전광역시',
    '추풍령': '상주시',
    '안동': '안동시',
    '상주': '상주시',
    '포항': '포항시',
    '군산': '군산시',
    '대구': '대구광역시',
    '전주': '전주시',
    '울산': '울산광역시',
    '창원': '창원시',
    '광주': '광주광역시',
    '부산': '부산광역시',
    '통영': '통영시',
    '목포': '목포시',
    '여수': '여수시',
    '흑산도': '신안군',
    '완도': '완도군',
    '고창': '고창군',
    '순천': '순천시',
    '진도(첨찰산)': '진도군',
    '대구(기)': '대구광역시',
    '홍성': '홍성군',
    '서청주': '청주시',
    '제주': '제주시',
    '고산': '서귀포시',
    '성산': '서귀포시',
    '서귀포': '서귀포시',
    '진주': '진주시',
    '강화': '강화군',
    '양평': '양평군',
    '이천': '이천시',
    '인제': '인제군',
    '홍천': '홍천군',
    '태백': '태백시',
    '정선군': '정선군',
    '제천': '제천시',
    '보은': '보은군',
    '천안': '천안시',
    '보령': '보령시',
    '부여': '부여군',
    '금산': '금산군',
    '세종': '세종특별자치시',
    '부안': '부안군',
    '임실': '임실군',
    '정읍': '정읍시',
    '남원': '남원시',
    '장수': '장수군',
    '고창군': '고창군',
    '영광군': '영광군',
    '김해시': '김해시',
    '순창군': '순창군',
    '북창원': '창원시',
    '양산시': '양산시',
    '보성군': '보성군',
    '강진군': '강진군',
    '장흥': '장흥군',
    '해남': '해남군',
    '고흥': '고흥군',
    '의령군': '의령군',
    '함양군': '함양군',
    '광양시': '광양시',
    '진도군': '진도군',
    '봉화': '봉화군',
    '영주': '영주시',
    '문경': '문경시',
    '청송군': '청송군',
    '영덕': '영덕군',
    '의성': '의성군',
    '구미': '구미시',
    '영천': '영천시',
    '경주시': '경주시',
    '거창': '거창군',
    '합천': '합천군',
    '밀양': '밀양시',
    '산청': '산청군',
    '거제': '거제시',
    '남해': '남해군',
    '북부산': '부산광역시',
}

# 시도와 시군구 칼럼 추가
df['시도'] = df['지점명'].map(sido_dict)
df['시군구'] = df['지점명'].map(sigungu_dict)

월 칼럼 생성

# 일시 칼럼을 datetime 형식으로 변환
df['일시'] = pd.to_datetime(df['일시'])

# 월 칼럼 추가
df['월'] = df['일시'].dt.month

칼럼 순서 변경

cols = list(df.columns)
col_to_move = cols.pop(14)
cols.insert(3, col_to_move)
df = df[cols]

cols = list(df.columns)
col_to_move = cols.pop(13)
cols.insert(2, col_to_move)
df = df[cols]

cols = list(df.columns)
col_to_move = cols.pop(14)
cols.insert(3, col_to_move)
df = df[cols]

df

결측치 확인

df.isnull().sum()

df[df['지점명']=='진도군']

df[df['평균기온'].isnull()]

df = df.sort_values(by=['지점명', '일시']).reset_index(drop=True)

df

df['지점명'].value_counts().tail(10)

# 지점명별 행의 개수 계산
count_per_station = df['지점명'].value_counts()

# 막대 그래프 그리기
plt.figure(figsize=(16, 6))
sns.barplot(x=count_per_station.index, y=count_per_station.values, palette="viridis")
plt.xlabel("지점명")
plt.ylabel("행 개수")
plt.title("지점명별 행 개수")
plt.xticks(rotation=90)
plt.show()

행의 개수가 9개년 미만인 지역들을 삭제

# 삭제할 지점명 리스트
remove_stations = ['고창', '홍성', '북춘천', '세종', '진도(첨찰산)', '북부산', '서청주', '대구(기)']

# 지점명 칼럼에서 해당 지점명이 아닌 행들만 남김
df = df[~df['지점명'].isin(remove_stations)].reset_index(drop=True)

df

평균기온에 대한 선형보간

# 지점명별로 평균기온에 대해 일시 기준으로 선형 보간 적용
df['평균기온'] = df.groupby('지점명')['평균기온'].transform(lambda x: x.interpolate(method='linear'))

df.isnull().sum()

최고기온 및 최저기온에 대한 선형보간

df['최고기온'] = df.groupby('지점명')['최고기온'].transform(lambda x: x.interpolate(method='linear'))
df['최저기온'] = df.groupby('지점명')['최저기온'].transform(lambda x: x.interpolate(method='linear'))

평균풍속, 평균상대습도, 평균증기압, 평균지면운도에 대한 선형보간

df['평균풍속'] = df.groupby('지점명')['평균풍속'].transform(lambda x: x.interpolate(method='linear'))
df['평균상대습도'] = df.groupby('지점명')['평균상대습도'].transform(lambda x: x.interpolate(method='linear'))
df['평균증기압'] = df.groupby('지점명')['평균증기압'].transform(lambda x: x.interpolate(method='linear'))
df['평균지면온도'] = df.groupby('지점명')['평균지면온도'].transform(lambda x: x.interpolate(method='linear'))

df.isnull().sum()

습구온도 칼럼 및 체감온도 칼럼 생성

# Stull의 습구온도 추정식 함수 정의
def calculate_wet_bulb_temperature(T, RH):
    Tw = (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) +
          np.arctan(T + RH) -
          np.arctan(RH - 1.676331) +
          0.00391838 * (RH ** (3/2)) * np.arctan(0.023101 * RH) -
          4.686035)
    return Tw

# 여름철 체감온도 계산 함수 정의
def calculate_feels_like_temperature(Ta, Tw, RH):
    return -0.2442 + 0.55399 * Tw + 0.45535 * Ta - 0.0022 * Tw**2 + 0.00278 * Tw * Ta + 3.0

# 겨울철 체감온도 계산 함수 정의
def calculate_winter_feels_like_temperature(Ta, V):
    return 13.12 + 0.6215 * Ta - 11.37 * (V ** 0.16) + 0.3965 * (V ** 0.16) * Ta

# 습구온도 계산
df['습구온도'] = calculate_wet_bulb_temperature(df['최고기온'], df['평균상대습도'])

# 여름철 체감온도 계산
df.loc[~df['월'].isin([1, 2, 3, 4, 10, 11, 12]), '체감온도'] = \
    calculate_feels_like_temperature(df['최고기온'], df['습구온도'], df['평균상대습도'])

# 겨울철 체감온도 계산
df.loc[df['월'].isin([1, 2, 3, 4, 10, 11, 12]), '체감온도'] = \
    calculate_winter_feels_like_temperature(df['최저기온'], df['평균풍속'])

df

df.isnull().sum()

import pandas as pd
import numpy as np
from datetime import timedelta

# 날짜 변환
df['일시'] = pd.to_datetime(df['일시'])

# 전체 기준 날짜
full_dates = pd.date_range(start='2014-10-05', end='2024-10-05')

# 기준 및 보간 대상 컬럼
group_cols = ['지점명', '시도', '시군구']
numeric_cols = [
    '평균기온', '최저기온', '최고기온', '평균풍속',
    '평균상대습도', '평균증기압', '평균지면온도',
    '습구온도', '체감온도'
]

### 1단계: 누락 날짜 행 추가 + 앞뒤 1일 보간
filled_rows = []

for name, group in df.groupby('지점명'):
    group = group.set_index('일시').sort_index()
    missing = full_dates.difference(group.index)
    meta_info = group[[col for col in group.columns if col not in numeric_cols]].iloc[0]

    for date in missing:
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)

        row = meta_info.copy()
        row['일시'] = date

        for col in numeric_cols:
            prev_val = group[col].get(prev_day, np.nan)
            next_val = group[col].get(next_day, np.nan)

            if pd.notna(prev_val) and pd.notna(next_val):
                row[col] = (prev_val + next_val) / 2
            elif pd.notna(prev_val):
                row[col] = prev_val
            elif pd.notna(next_val):
                row[col] = next_val
            else:
                row[col] = np.nan

        filled_rows.append(row)

# 결합 및 정렬
filled_df = pd.DataFrame(filled_rows)
final_df = pd.concat([df, filled_df], ignore_index=True)
final_df['일시'] = pd.to_datetime(final_df['일시'])
final_df = final_df.sort_values(by=['지점명', '일시']).reset_index(drop=True)

### 2단계: 남은 결측치 → 1일/2일 앞뒤 평균 보간
nan_rows = final_df[final_df[numeric_cols].isnull().any(axis=1)].copy()

for idx, row in nan_rows.iterrows():
    name = row['지점명']
    date = row['일시']

    before_1 = final_df[(final_df['지점명'] == name) & (final_df['일시'] == date - timedelta(days=1))]
    after_1  = final_df[(final_df['지점명'] == name) & (final_df['일시'] == date + timedelta(days=1))]
    before_2 = final_df[(final_df['지점명'] == name) & (final_df['일시'] == date - timedelta(days=2))]
    after_2  = final_df[(final_df['지점명'] == name) & (final_df['일시'] == date + timedelta(days=2))]

    for col in numeric_cols:
        if pd.isna(final_df.at[idx, col]):
            if not before_1.empty and not after_1.empty:
                final_df.at[idx, col] = (before_1[col].values[0] + after_1[col].values[0]) / 2
            elif not before_2.empty and not after_2.empty:
                final_df.at[idx, col] = (before_2[col].values[0] + after_2[col].values[0]) / 2

### 3단계: 연속 결측 블록 보간
def interpolate_block(final_df, name, col):
    target = final_df[final_df['지점명'] == name][['일시', col]].set_index('일시').sort_index()
    isna = target[col].isna()

    block_starts = (isna & ~isna.shift(1).fillna(False))
    block_ends = (isna & ~isna.shift(-1).fillna(False))
    block_ranges = list(zip(target.index[block_starts], target.index[block_ends]))

    for start, end in block_ranges:
        before = start - pd.Timedelta(days=1)
        after = end + pd.Timedelta(days=1)

        before_val = target[col].get(before, np.nan)
        after_val = target[col].get(after, np.nan)

        if pd.notna(before_val) and pd.notna(after_val):
            fill_val = (before_val + after_val) / 2
            final_df.loc[
                (final_df['지점명'] == name) &
                (final_df['일시'] >= start) &
                (final_df['일시'] <= end),
                col
            ] = fill_val

# 전체 적용
for col in numeric_cols:
    for name in final_df['지점명'].unique():
        interpolate_block(final_df, name, col)

final_df

final_df.isnull().sum()

df = final_df.copy()

폭염여부 칼럼

df['폭염여부'] = np.where(df['체감온도'] >= 33, 1, 0)

df

df[df['일강수량'].isnull()]

subset_df = df.iloc[:, 6:]

# 상관관계 행렬 계산
correlation_matrix = subset_df.corr()

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# 시도별 개수 계산
sido_counts = df['시도'].value_counts()

# 막대그래프 그리기
plt.figure(figsize=(12, 6))  # 그래프 크기 조정
sido_counts.plot(kind='bar', color='skyblue')
plt.title('시도별 행의 개수')
plt.xlabel('시도')
plt.ylabel('행의 개수')
plt.xticks(rotation=45)  # x축 레이블 회전
plt.grid(axis='y')  # y축에만 그리드 표시
plt.show()

# 시도별 고유 시군구 개수 계산
sido_sigungu_counts = df.groupby('시도')['시군구'].nunique()

# 막대그래프 그리기
plt.figure(figsize=(12, 6))  # 그래프 크기 조정
sido_sigungu_counts.plot(kind='bar', color='lightcoral')
plt.title('시도별 고유 시군구 개수')
plt.xlabel('시도')
plt.ylabel('고유 시군구 개수')
plt.xticks(rotation=45)  # x축 레이블 회전
plt.grid(axis='y')  # y축에만 그리드 표시
plt.show()

# 평균 폭염여부 수 계산
average_heatwave = df.groupby('지점명')['폭염여부'].mean().reset_index()

# 칼럼명 수정 (가독성을 위해)
average_heatwave.columns = ['지점명', '평균폭염여부']

# 시각화
plt.figure(figsize=(14, 7))
plt.bar(average_heatwave['지점명'], average_heatwave['평균폭염여부'], color='skyblue')
plt.title('지점명별 평균 폭염여부')
plt.xlabel('지점명')
plt.ylabel('평균 폭염여부')
plt.xticks(rotation=90)  # x축 레이블 회전
plt.grid(axis='y')  # y축 그리드 추가
plt.tight_layout()  # 레이아웃 조정
plt.show()

# 상위 30곳 선택
top_30_heatwave = average_heatwave.sort_values(by='평균폭염여부', ascending=False).head(30)

# 시각화
plt.figure(figsize=(14, 7))
plt.bar(top_30_heatwave['지점명'], top_30_heatwave['평균폭염여부'], color='skyblue')
plt.title('상위 30곳 지점명별 평균 폭염여부')
plt.xlabel('지점명')
plt.ylabel('평균 폭염여부')
plt.xticks(rotation=45)  # x축 레이블 회전
plt.grid(axis='y')  # y축 그리드 추가
plt.tight_layout()  # 레이아웃 조정
plt.show()

# 지점명별 폭염여부가 1인 행의 개수 계산
heatwave_count = df[df['폭염여부'] == 1].groupby('지점명')['폭염여부'].count().reset_index()

# 칼럼명 수정 (가독성을 위해)
heatwave_count.columns = ['지점명', '폭염여부_1_개수']

# 상위 30곳 선택 (폭염여부가 1인 행 개수 기준)
top_30_heatwave_count = heatwave_count.sort_values(by='폭염여부_1_개수', ascending=False).head(30)

# 시각화
plt.figure(figsize=(14, 7))
plt.bar(top_30_heatwave_count['지점명'], top_30_heatwave_count['폭염여부_1_개수'], color='salmon')
plt.title('상위 30곳 지점명별 폭염여부가 1인 행 개수')
plt.xlabel('지점명')
plt.ylabel('폭염여부가 1인 행 개수')
plt.xticks(rotation=45)  # x축 레이블 회전
plt.grid(axis='y')  # y축 그리드 추가
plt.tight_layout()  # 레이아웃 조정
plt.show()

평균폭염발생횟수가 많은 상위 10개 지역에 대한 시계열 분석 진행

average_heatwave.sort_values(by='평균폭염여부', ascending=False).head(10)

df.to_csv('폭염데이터_전처리완료_대학원.csv', encoding = 'utf-8-sig')

df
