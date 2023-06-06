import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 시간있으면 가능한 리스트들 다 넘파이로 바꾸기.

def main():
    st.title('대학생 행동 분석 앱')

    df = pd.read_csv('data/bot_detection_data2.csv')
    df1 = pd.read_csv('data/bot_detection_data.csv')
    menu = ['개요', '데이터 분석', '데이터 예측']
    choise = st.sidebar.selectbox('목록', menu)
    
    col_explain = {
        'Tweet': '트윗한 내용의 글자수', 
        'Retweet Count': '리트윗한 횟수', 
        'Mention Count': '멘션한 횟수', 
        'Follower Count': '팔로워 수', 
        'Verified': '사용자가 확인되었는지 여부', 
        'Bot Label': '봇인지 아닌지', 
        'Created At': '만든 시점', 
        'Hashtags': '해시태그 여부'
    }
    lb_list = ['Verified', 'Bot Label', 'Hashtags']


    if choise == menu[0]:
        st.image('data/img.jpg') # 수정
        if st.checkbox('데이터프레임 보기'):
            st.dataframe(df1)
        st.text('50000 rows × 11 columns')
        st.subheader('개요')
        st.text('트위터의 활동내역을 보고 AI인지 아닌지 예측합니다.')
        st.subheader('목차')
        st.text('''
        - 데이터의 값 설명
        - 데이터 값 확인
        - 데이터 시각화 및 분석
        - 여러 데이터 예측하기
        ''')
        
        st.subheader('출처')
        st.text('kaggle Twitter-Bot Detection Dataset')
        st.text('https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset?select=bot_detection_data.csv')

    elif choise == menu[1]:
        import platform
        platform.platform()
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['font.size'] = 15
        plt.rcParams['axes.unicode_minus'] = False
        calculator = {
            '최댓값' : 'df[i].max()',
            '최솟값' : 'df[i].min()',
            '평균' : 'round(df[i].mean(), 1)',
            '중앙값' : 'df[i].median()'
        }
        st.header('데이터 분석')
        st.subheader('데이터 설명')
        st.dataframe(df)
        for i in col_explain.keys():
            st.text(f'{i}: {col_explain[i]}')

        st.subheader('데이터 값')
        select_calcul = st.radio('보고싶은 값을 선택해주세요.', calculator.keys())
        for i in set(df.columns) - set(lb_list) - {'Created At'}:
            st.text(f'{i}의 {select_calcul}: {eval(calculator[select_calcul])}')

        st.subheader('상관관계')
        df_corr = df.corr()
        fig = plt.figure()
        plt.title('간략한 상관관계(%)')
        sns.heatmap(data=df.corr(numeric_only=True).loc[:, :] * 100, annot=True, vmin=-100, vmax=100, cmap='coolwarm', fmt='.1f', linewidths=1)
        st.pyplot(fig)

        for i in df.columns:
            df_corr.loc[abs(df_corr[i]) < 0.1, i] = np.NaN


    elif choise == menu[2]:
        st.header('데이터 예측')
        st.text('입력 받을 데이터를 정하면 봇인지 아닌지 예측합니다.')
        st.subheader('입력할 데이터')
        if st.checkbox('전부 선택하기'):
            pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', df.drop('Bot Label', axis=1).columns, default=df.drop('Bot Label', axis=1).columns.values)
        else:
            pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', set(df.columns) - set('Bot Label'))
        if pred_choise:
            new_data = []
            X_choise = []

            for i in pred_choise:
                if i.endswith('Count') or i == 'Tweet':
                    if i == 'Follower Count':
                        new_data.append(st.slider(f'{add_postposition(col_explain[i])} 입력해주세요.', min_value=0, max_value=10000 , value=int(df[i].mean())))
                    else:
                        new_data.append(st.number_input(f'{add_postposition(col_explain[i])} 입력해주세요.', min_value=0, max_value=int(df[i].max() * 1.5) , value=int(df[i].mean())))
                elif i == 'Created At':
                    new_data.append(date_int(str(st.date_input('날짜를 입력해주세요.', min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2099, 12, 31)))))
                else:
                    new_data_dict = {'Verified' : ['인증X', '인증O'], 'Hashtags' : ['해시태그X', '해시태그O']}
                    new_data.append(new_data_dict[i].index(st.select_slider(f'{add_postposition(col_explain[i])} 정해주세요.', new_data_dict[i])))
                X_choise.append(i)
            X = df[X_choise]
            y = df['Bot Label']

            if st.button('예측결과 보기'):
                st.subheader('결과')
                st.text('Bot Label')

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=600)
                regressor = LinearRegression()
                regressor.fit(X_train.values, y_train.values)
                y_pred = regressor.predict(X_test)
                accuracy = round(abs((y_test - y_pred).mean()), 2)
                new_data = np.array(new_data)
                new_data = new_data.reshape(1, len(new_data))
                new_data = pd.DataFrame(new_data)
                answer = regressor.predict(new_data.values)[0]
                ans, acc = [answer , accuracy]

                label_values = ['봇이 아닙니다.', '봇입니다.']
                val = label_values[1] if ans > 0.5 else label_values[0]
                ans_list = [round(1 - abs(ans - acc - label_values.index(val)) / (abs(ans - acc) + abs(ans - acc - 1)), 3) * 100, round(1 - abs(ans + acc - label_values.index(val)) / (abs(ans + acc) + abs(ans + acc - 1)), 3) * 100]
                min_per, max_per = min(ans_list), max(ans_list)
                over_under = [['이하', '이상'], ['이상', '이하']]
                if ans - acc > 1 or ans + acc < 0:
                    st.text(val)
                elif ans + acc > 1 and ans - acc < 1:
                    st.text(f'{val}가 {min_per}% {over_under[label_values.index(val)][0]}')
                elif ans - acc < 0 and ans + acc > 0:
                    st.text(f'{val}가 {max_per}% {over_under[label_values.index(val)][1]}')
                else:
                    st.text(f'{min_per} ~ {max_per}% {val}')
        else:
            st.text('선택된 값이 없습니다.')


def date_int(word):
    ans = int(word[5:7])
    ans +=  12 * int(word[2:4])
    return ans

def add_postposition(word):
    return word + ('를' if (ord(word[-1]) - 44032) % 28 == 0 else '을')


if __name__ == '__main__':
    main()