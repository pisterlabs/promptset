import os
import streamlit as st
import pandas as pd  # for DataFrames to store article sections and embedding
from fuzzywuzzy import fuzz
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks import StreamlitCallbackHandler

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 아파트 이름과 주소 유사도 비교
def split_address(addr):
    addrs = addr.split(",")  # 쉼표를 기준으로 주소를 나눈다.
    addrs = [addr.rsplit("(", 1)[0].strip() for addr in addrs]  # 괄호 안의 내용 제거
    bases_nums = [addr.rsplit(" ", 1) if " " in addr else (addr, "") for addr in addrs]
    return bases_nums

def get_address_similarity(addr1, addr2):
    bases_nums1 = split_address(addr1)
    bases_nums2 = split_address(addr2)
    
    max_similarity = 0
    
    for base1, num1 in bases_nums1:
        for base2, num2 in bases_nums2:
            num_similarity = fuzz.ratio(num1, num2)     # 번지수를 문자열로 비교
            # 앞부분은 토큰 비교 (OO동 있어도 됨), partial_ratio 쓰면 빈 칸 무시하기에는 좋으나 동이 들어있으면 낮게 나옴
            base_similarity = fuzz.token_set_ratio(base1, base2)
            addr_similarity = 0.3 * num_similarity + 0.7 * base_similarity
            max_similarity = max(max_similarity, addr_similarity)
    
    return max_similarity


def get_name_similarity(name1, name2):
    return fuzz.token_set_ratio(name1, name2)

def get_name_addr_similarity(data1, data2):
    name1, addr1 = data1
    name2, addr2 = data2
    
    name_similarity = get_name_similarity(name1, name2)
    address_similarity = get_address_similarity(addr1, addr2)

    if address_similarity == 100 or (name_similarity == 100 and address_similarity > 70):
        total_similarity = 100
    else:
        total_similarity = 0.5 * name_similarity + 0.5 * address_similarity
    return total_similarity, name_similarity, address_similarity

# 이름과 주소 유사도 비교 using OpenAI's Sentence Embedding
def get_oai_similarity(data1, data2):
    name1, addr1 = data1
    name2, addr2 = data2
    
    text1 = f"이름: {name1}, 주소: {addr1}"
    text2 = f"이름: {name2}, 주소: {addr2}"

    db = FAISS.from_texts([text1], OpenAIEmbeddings())
    similarity = db.similarity_search_with_relevance_scores(text2)
    return similarity[0][1] # similarity score

# 이름과 주소가 유사한 아파트 찾기
@st.cache_data
def get_matches(selEvc, selApt, THRESHOLD=75):
    def get_best_match(row):
        scores = selApt.apply(lambda row2: get_name_addr_similarity((row['충전소'], row['주소']), (row2['단지명'], row2['도로명주소'])), axis=1)
        scores = pd.DataFrame(scores.tolist(), columns=['매칭점수', '이름비교', '주소비교'])
        max_idx = scores['매칭점수'].idxmax()
        best_match = {
            '단지코드': selApt.iloc[max_idx]['단지코드'],
            '단지명': selApt.iloc[max_idx]['단지명'],
            '비교주소': selApt.iloc[max_idx]['도로명주소'],
            '매칭점수': scores.iloc[max_idx]['매칭점수'],
            '이름비교': scores.iloc[max_idx]['이름비교'],
            '주소비교': scores.iloc[max_idx]['주소비교'],
            'Confirm': True if scores.iloc[max_idx]['매칭점수'] >= THRESHOLD else False
        }
        # best_match['Confirm'] = True if best_match['매칭점수'] >= THRESHOLD else False
        # st.write(best_match)

        return best_match
    
    matches = selEvc.apply(get_best_match, axis=1)
    matches = pd.DataFrame(matches.tolist())
    selEvc = pd.concat([selEvc.reset_index(drop=True), matches], axis=1)

    compare_result = {
        '주소일치': len(selEvc[selEvc['매칭점수'] == 100.0]),
        '유사추정': len(selEvc[(selEvc['매칭점수'] < 100.0) & (selEvc['Confirm'] == True)]),
        '불일치': len(selEvc[selEvc['Confirm'] == False])
    }

    return selEvc, compare_result

@st.cache_data
def llm_match(df_unmatched: pd.DataFrame) -> pd.DataFrame:
    """
    [이름, 주소, 이름2, 주소2] 형태로 된 DF를 받아서 두 엔터티가 유사한지 확인하는 LLM을 실행합니다.
    :param df_unmatched: [이름, 주소, 이름2, 주소2] 형태로 된 DF
    :return: [이름, 주소, 이름2, 주소2, 유사도(Bool)] 형태로 된 DF 
    """

    # LLM을 이용하여 유사도가 애매한 경우에 대하여 재검토 (NA중에서 매칭점수가 특정 이상인 경우)

    # 모든 행에 대해서 T / F 표시
    PROMPT = """Every line in <INPUT> is composed two entities with name and address and separated by semi-colon. 
    "name1"; "address1"; "name2"; "address2"
    Compare two entities in single row and decide if they are similar or not.
    <OUTPUT> should be same as <INPUT> but with an additional column shows similar or not by saying "True" or "False".

    For example,
    <INPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20
    화성시 동탄지웰에스테이트; 경기도 화성시 동탄반석로 160; 동탄현대하이페리온; 경기도 화성시 동탄반석로 156

    <OUTPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132; True
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706; True
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20; True
    화성시 동탄지웰에스테이트; 경기도 화성시 동탄반석로 160; 동탄현대하이페리온; 경기도 화성시 동탄반석로 156; False

    Please check the similarity of the following entities.
    <INPUT>
    {input}
    """

    # True인 행만 남기기 - 잘 안됨
    PROMPT_1 = """Every line in <INPUT> is composed two entities with name and address and separated by semi-colon. 
    "name1"; "address1"; "name2"; "address2"
    Compare two entities in single row and decide if they are similar or not. If similar, append "True" in <OUTPUT> otherwise discard the row.

    For example,
    <INPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706
    화성시 동탄지웰에스테이트; 경기도 화성시 동탄반석로 160; 동탄현대하이페리온; 경기도 화성시 동탄반석로 156
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20

    <OUTPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132; True
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706; True
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20; True

    Please check the similarity of the following entities.
    <INPUT>
    {input}
    """

    # gpt3용 prompt
    PROMPT_gpt3 = """Every line in <INPUT> is composed two entities with name and address and separated by semi-colon. 
    "name1"; "address1"; "name2"; "address2"
    Compare two entities in single row and decide if they are similar or not.
    <OUTPUT> should be same as <INPUT> but with an additional column shows similar or not by saying "True" or "False".

    For example,
    <INPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20
    화성시 동탄지웰에스테이트; 경기도 화성시 동탄반석로 160; 동탄현대하이페리온; 경기도 화성시 동탄반석로 156

    <OUTPUT>
    신대연코오롱하늘채; 부산광역시 남구 홍곡로 320번길 132; 신대연코오롱하늘채아파트; 부산광역시 남구 홍곡로320번길 132; True
    화성동탄상록리슈빌아파트(공무원연금공단)(21년); 경기도 화성시 동탄순환대로29길 57; 화성동탄상록리슈빌아파트; 경기도 화성시 동탄순환대로 706; True
    동탄2 LH26단지(65BL); 경기도 화성시 송동 681-127; 동탄2 LH26단지(65BL)아파트; 경기도 화성시 동탄대로9길 20; True
    화성시 동탄지웰에스테이트; 경기도 화성시 동탄반석로 160; 동탄현대하이페리온; 경기도 화성시 동탄반석로 156; False

    Please compare the similarity of the following entities and write down <OUTPUT>.
    <INPUT>
    {input}
    """

    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False, collapse_completed_thoughts=True)
    gpt3_16 = ChatOpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0.0, streaming=True, callbacks=[st_callback])
    gpt3 = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.0, streaming=True, callbacks=[st_callback])
    gpt4 = ChatOpenAI(model_name = "gpt-4", temperature=0.0, streaming=True, callbacks=[st_callback])

    llm_chain = LLMChain(llm=gpt3, prompt=PromptTemplate.from_template(PROMPT_gpt3), verbose=True)

    str_buffer = StringIO()
    df_unmatched[['충전소', '주소', '단지명', '비교주소']].to_csv(str_buffer, sep=";", header=False, index=False)
    input = str_buffer.getvalue()

    output = llm_chain.run(input)   # return output string from LLM

    # convert output string to dataframe
    str_buffer = StringIO(output)
    df_new = pd.read_csv(str_buffer, sep=";", header=None, names=['충전소', '주소', '단지명', '비교주소', 'byLLM'], skiprows=1)

    # True/False 값을 문자열에서 불리언으로 변환
    df_new['byLLM'] = df_new['byLLM'].str.strip().map({'True': True, 'False': False})
    # '충전소', '주소', '단지명', '비교주소'를 기준으로 df_unmatched의 각 행에 대한 byLLM 값을 찾습니다.
    df_unmatched['byLLM'] = df_unmatched.set_index(['충전소', '주소', '단지명', '비교주소']).index.map(df_new.set_index(['충전소', '주소', '단지명', '비교주소'])['byLLM'])
    # byLLM 값이 True인 경우에만 Confirm 값을 True로 설정합니다.
    df_unmatched['Confirm'] = df_unmatched['byLLM'].apply(lambda x: True if x == True else False)
    df_unmatched = df_unmatched.sort_values(by=['byLLM', '매칭점수'], ascending=[False, False])

    return df_unmatched



@st.cache_data
def load_datafile():
    # load excel files from current directory
    
    with st.spinner('첫 실행 시 사례 데이터를 로딩합니다. 잠시만 기다려주세요...'):
        df = pd.read_csv("kr_evcharger_list.csv", header=2, usecols=range(15))
        df['주소'] = df['주소'].fillna('Unknown')
        # 모든 열에 rstrip() 적용
        evc = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)
        df = pd.read_csv("k-apt_info_20230818.csv", header=1)
        df['도로명주소'] = df['도로명주소'].fillna('Unknown')
        # 모든 열에 rstrip() 적용
        df = df[['시도', '시군구', '단지코드', '단지명', '도로명주소', '총주차대수', '세대수']]
        aptInfo = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)
        
        st.success('Data Loaded!')
    return evc, aptInfo

@st.cache_data
def load_data_parquet():
    with st.spinner('첫 실행 시 사례 데이터를 로딩합니다. 잠시만 기다려주세요...'):
        evc = pd.read_parquet("evc-list.parquet")
        aptInfo = pd.read_parquet("apt-list.parquet")
        # st.success('Data Loaded!')
    return evc, aptInfo


st.title("충전 데이터 분석")
"""
https://www.ev.or.kr/evmonitor 에서 받은 충전소 데이터에서 시설구분(소)가 '아파트'인 데이터를 가져옵니다.
충전소 이름으로 묶어서 충전기 수를 계산하고, K-APT에서 받은 단지 기본 정보를 이용하여 아파트 주소와 유사한 충전소를 찾습니다.
"""

# load parquet of (충전소 현황, 단지 기본 정보) from current directory
evc, aptInfo = load_data_parquet()

# Set up sidebar
THRESHOLD = st.sidebar.slider("(단지명 및 주소) 유사도 임계값", 60, 100, 75, 5)

# Study on the number of charging stations in each Apt
st.sidebar.header("아파트 충전기 설치 현황")

filtered_df = evc[evc['시설구분(소)'] == '아파트']

# If there are any rows satisfying the condition
if not filtered_df.empty:
    # Group by '충전소' and count the entries
    grouped = filtered_df.groupby('충전소').count()
    st.sidebar.write("'충전소'가 있는 아파트 총수: ", len(grouped))
    st.sidebar.write("k-apt 등록 아파트 총수: ", len(aptInfo))
    # st.write(grouped)
    
# 아파트별 충전기 데이터 새로 만들기
aptEvc = filtered_df.groupby('충전소').agg({
    '운영기관': 'first',
    '충전기ID': 'size',  # 충전기수 계산
    '충전기타입': 'first',
    '시설구분(대)': 'first',
    '시설구분(소)': 'first',
    '지역': 'first',
    '시군구': 'first',
    '주소': 'first',
}).reset_index()
# 열 이름 변경
aptEvc.rename(columns={'충전기ID': '충전기수'}, inplace=True)


# save files to parquet
# if st.button("Save Data"):
#     evc.to_parquet("evc-list.parquet")
#     aptInfo.to_parquet("apt-list.parquet")



# study on # of Apt in each region 
# 지역 선택한 것만 보이도록 selectbox 만들기

option1 = ['전체 선택'] + list(aptEvc['지역'].unique())
selected_province = st.sidebar.selectbox('시도를 선택하세요', option1, index=min(1, len(option1)-1))
if selected_province == '전체 선택':
    option2 = ['전체 선택']
    selected_province = None
else:
    option2 = ['전체 선택'] + list(aptEvc[aptEvc['지역'] == selected_province]['시군구'].unique())

selected_region = st.sidebar.selectbox('시군구를 선택하세요', option2, index=min(1, len(option2)-1))
if selected_region == '전체 선택':
    selected_region = None

st.sidebar.warning("전체선택을 하면 처리하는 데 시간이 꽤 걸립니다. 주의하세요.")

# selection이 바뀌면 llm_output을 초기화
if 'selected_province_prev' not in st.session_state:
    st.session_state.selected_province_prev = None

if st.session_state.selected_province_prev != selected_province:
    st.session_state.llm_output = None
    st.session_state.selected_province_prev = selected_province

if 'selected_region_prev' not in st.session_state:
    st.session_state.selected_region_prev = None

if st.session_state.selected_region_prev != selected_region:
    st.session_state.llm_output = None
    st.session_state.selected_region_prev = selected_region


# 선택된 시군구 관련 데이터 표시하기 위한 sidebar
sidebar_selected = st.sidebar.container()

view_raw = st.expander("시도/시군구별 데이터 현황 (충전소 & 단지 기본 정보)", expanded=False)

with view_raw:
    # 두 칼럼으로 충전소 데이터와 아파트 데이터 보여주기
    colEvc, colApt = st.columns(2)

    with colEvc:
        if selected_province and selected_region:
            selected = aptEvc[(aptEvc['지역'] == selected_province) & (aptEvc['시군구'] == selected_region)]
        elif selected_province:
            selected = aptEvc[aptEvc['지역'] == selected_province]
        else:
            selected = aptEvc
            
        selEvc = selected[['충전소', '주소', '충전기수', '지역', '시군구']]
        
        st.write("충전소 현황", len(selected))
        st.write(selected)
        with sidebar_selected:
            st.write("충전소수(ev.or.kr)", len(selected))
        # count_addr = selected[selected['주소'].apply(lambda x: pd.isnull(x) or not isinstance(x, str))].shape[0]
        # st.write(count_addr, "개의 주소가 없습니다.")

    with colApt:
        if selected_province and selected_region:
            selected = aptInfo[(aptInfo['시도'] == selected_province) & (aptInfo['시군구'] == selected_region)]
        elif selected_province:
            selected = aptInfo[aptInfo['시도'] == selected_province]
        else:
            selected = aptInfo
        
        selApt = selected[['단지명', '단지코드', '도로명주소', '총주차대수', '세대수', '시도', '시군구']]
        # selected = selApt.sort_values(by='단지명')

        st.write("아파트 현황", len(selected))
        st.write(selected)
        with sidebar_selected:
            st.write("아파트수(k-apt)", len(selected))

        # count_addr = selected[selected['도로명주소'].apply(lambda x: pd.isnull(x))].shape[0]
        # st.write(count_addr, "개의 주소가 없습니다.")



view_context = st.expander("아파트 이름 & 주소 유사도 검색 결과", expanded=False)

# selEvc, compare_result = get_matches(selEvc)
# 각 그룹의 결과를 저장할 빈 리스트를 생성합니다.
result_list = []
compare_result = {'주소일치': 0, '유사추정': 0, '불일치': 0}

# 각 그룹에 대해 get_matches 함수를 반복적으로 호출합니다. "전체 선택"이 아니면 region, district가 하나여서 반복문이 한 번만 실행됩니다.
for (region, district), group in selEvc.groupby(['지역', '시군구']):
    for (r2, d2), g2 in selApt.groupby(['시도', '시군구']):
        if (region, district) == (r2, d2):
            selEvc_group, compare_result_group = get_matches(group, g2, THRESHOLD)
            result_list.append(selEvc_group)
            st.info(f"{region} {district}의 데이터를 처리했습니다.")
            for key in compare_result.keys():
                compare_result[key] += compare_result_group[key]

# 각 그룹의 결과를 연결하여 최종 결과 데이터프레임을 만듭니다.
if result_list:
    result_df = pd.concat(result_list, ignore_index=True)
    selEvc = result_df

with sidebar_selected:
    st.write("주소 일치 하는 경우가", compare_result['주소일치'], "개,\n비슷하게 추정한 경우가", compare_result['유사추정'], "개,\n불일치하는 경우가", compare_result['불일치'], "개 입니다.")

with view_context:
    st.write("주소 일치", compare_result['주소일치'], selEvc[selEvc['매칭점수'] == 100])
    st.write("이름 및 주소 유사", compare_result['유사추정'], selEvc[(selEvc['매칭점수'] < 100) & selEvc['Confirm']])
    st.write("이름 및 주소 불일치", compare_result['불일치'], selEvc[selEvc['Confirm'] == False])


# LLM을 이용하여 유사도가 애매한 경우에 대하여 재검토 (Confirm이 아니며 매칭점수가 특정 이상인 경우)
st.markdown("## 유사도 추가 검토가 필요한 항목")
df_unmatched = selEvc[(selEvc['매칭점수'] > 60) & (selEvc['Confirm'] == False)].copy()

st.session_state.llm_on = st.toggle("LLM으로 유사도 검토 결과 체크하기")
if st.session_state.llm_on:
    df_unmatched = llm_match(df_unmatched)

# 사용자에게 수정한 결과를 반영할 수 있도록 테이블을 보여줍니다.
df_confirmed = st.data_editor(
    df_unmatched,
    column_order= ['Confirm', '충전소', '주소', '단지명', '비교주소', '단지코드', 'byLLM', '매칭점수', '이름비교', '주소비교'],
    column_config={
        "Confirm": st.column_config.CheckboxColumn(
            "Confirm",
            help="이름 및 주소가 유사한 경우에만 체크하세요.",
        )
    },
    use_container_width=True,
    disabled=['충전소', '주소', 'byLLM', '매칭점수', '이름비교', '주소비교']
)

n_confirm = sum(df_confirmed['Confirm'])
if st.button(f"➕ {n_confirm}개의 수정한 검토 결과 반영하기"):
    selEvc.update(df_confirmed)


# 단지코드별로 groupby해서 전체 주차장수와 충전기수, 비율을 구하기
df = selEvc[selEvc['Confirm'] == True]
df = df.groupby('단지코드').agg({'충전기수': 'sum'}).reset_index()
df = df.merge(selApt, on='단지코드', how='left')
df['충전기설치율'] = df['충전기수'] / df['총주차대수']

st.markdown("""## 단지별 충전기 설치 현황
K-APT의 단지 정보와 일치(또는 유사)한 경우 단지코드별로 합산하여 계산했습니다.""")

st.dataframe(
    df[['단지코드', '단지명', '총주차대수', '충전기수', '충전기설치율']].sort_values(by='충전기설치율', ascending=False),
    column_config={
        "충전기설치율": st.column_config.ProgressColumn(
            "충전기설치율",
            min_value=0,
            max_value=0.2,
            format="%.2f",
        ),
    },
    hide_index=True,
)

df_remained = selEvc[selEvc['Confirm'] == False]
st.markdown("""## k-apt와 매칭되지 않는 충전소
K-APT의 단지 정보와 유사하지 않아 개별적으로 표시하였습니다.""")
st.dataframe(
    df_remained[['충전소', '주소', '충전기수']],
    hide_index=True,
)

# st.header("충전소 주소와 아파트 주소가 완전히 일치하지 않는 경우")
# df = selEvc[(selEvc['매칭점수'] < 100) & (selEvc['매칭점수'] > 0)]
# st.table(df[['매칭점수', '이름비교', '충전소', '단지명', '주소비교', '주소', '비교주소']].sort_values(by='매칭점수', ascending=False))
# st.write("충전소 주소와 아파트 주소가 완전히 일치하지 않는 경우: ", len(df))


# df = selEvc[(selEvc['매칭점수'] < 90) & (selEvc['주소비교'] > 50)].head(5)

# OpenAI Embedding을 이용한 유사도 비교
# for i, row in df.iterrows():
#     oai_similarity = get_oai_similarity((row['충전소'], row['주소']), (row['단지명'], row['비교주소']))
#     st.write(row, oai_similarity)


#for row in df.iterrows():

# '매칭점수'가 100점 미만인 경우에 한해서만 실행




# st.table(df[['매칭점수', '이름비교', '충전소', '단지명', '주소비교', '주소', '비교주소']].sort_values(by='매칭점수', ascending=False))

# st.table(selEvc[(selEvc['매칭점수'] < 100) & (selEvc['매칭점수'] > 0)])

#            st.write(row['충전소'], row['주소'], row2['단지명'], row2['도로명주소'], fuzz.ratio(row['주소'], row2['도로명주소']))





