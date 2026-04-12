import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 1. 페이지 설정
st.set_page_config(page_title="AI S&OP Control Tower", layout="wide")
st.title("🤖 AI 전문 상담가 기반 생산계획(APP) 관제 시스템")

# --- [신규] API 키 로테이션 설정 ---
# Streamlit Secrets에 GEMINI_KEYS = ["key1", "key2", "key3"] 형태로 저장하세요.
def get_gemini_response(prompt, context_data):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys:
            return "⚠️ API 키가 설정되지 않았습니다. Secrets를 확인하세요."
        
        # 키 순환 (랜덤하게 하나 선택)
        selected_key = random.choice(keys)
        genai.configure(api_key=selected_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 시스템 프롬프트 설정 (전문가 페르소나 부여) [cite: 1, 11]
        system_instruction = f"""
        당신은 홍익대학교 스마트제조 전문가 Chunghun Ha 박사의 AI 조수입니다. 
        사용자가 수립한 총괄생산계획(APP) 데이터를 바탕으로 전문적인 조언을 제공하세요. [cite: 3, 46]
        
        [현재 공장 데이터 현황]
        {context_data}
        
        위 데이터를 바탕으로 사용자의 질문에 답하세요. 비용 절감 방안이나 인력 운영의 적절성을 중점적으로 분석하세요.
        """
        
        response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 응답 중 오류 발생: {str(e)}"

# --- 사이드바: 운영 변수 (기존 기능 유지) ---
with st.sidebar:
    st.header("🎮 시스템 제어")
    opt_mode = st.radio("최적화 알고리즘", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.header("💰 단위 비용 (천원)")
    costs_input = {
        "reg": st.number_input("정규 임금", value=640),
        "ot": st.number_input("초과 근무", value=6),
        "h": st.number_input("고용 비용", value=300),
        "l": st.number_input("해고 비용", value=500),
        "inv": st.number_input("재고 유지", value=2),
        "back": st.number_input("부재고 비용", value=5),
        "mat": st.number_input("원자재비", value=10),
        "sub": st.number_input("외주비", value=30)
    }
    
    demand_raw = st.text_input("수요 예측 (6개월)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("초기 인원", value=80)
    i_init = st.number_input("초기 재고", value=1000)

# --- 최적화 실행 로직 ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(
        costs_input["reg"]*m.W[t] + costs_input["ot"]*m.O[t] + costs_input["h"]*m.H[t] + costs_input["l"]*m.L[t] + 
        costs_input["inv"]*m.I[t] + costs_input["back"]*m.S[t] + costs_input["mat"]*m.P[t] + costs_input["sub"]*m.C[t] 
        for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w_init); m.c.add(m.I[0] == i_init); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.c.add(m.P[t] <= 40*m.W[t] + 0.25*m.O[t])
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])
        m.c.add(m.O[t] <= 10*m.W[t])
    m.c.add(m.I[len(D)] >= 500); m.c.add(m.S[len(D)] == 0)
    SolverFactory('glpk').solve(m)
    return m

# --- 메인 화면 구성 ---
tab1, tab2 = st.tabs(["📊 데이터 분석 대시보드", "💬 AI 전문가 대화"])

with tab1:
    if st.button("🚀 최적화 실행"):
        model = solve_app(demand, domain_type)
        st.session_state['opt_model'] = model # 대화방 공유용
        
        # [시각화 생략 - 기존 코드와 동일하게 배치 가능]
        st.success(f"최적 운영 비용: {model.cost():,.0f} 천원")
        # 시각화 차트들... (생략)

with tab2:
    st.subheader("💬 AI 생산관리 컨설턴트")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 대화 이력 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("계획에 대해 궁금한 점을 물어보세요 (예: 이 계획의 리스크는 뭐야?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 최적화 데이터 요약해서 AI에게 전달
        opt_res = ""
        if 'opt_model' in st.session_state:
            m = st.session_state['opt_model']
            opt_res = f"총비용: {m.cost()}, 평균재고: {sum(m.I[t]() for t in range(1,7))/6}"

        with st.chat_message("assistant"):
            response = get_gemini_response(prompt, opt_res)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
