import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

st.set_page_config(page_title="Horticultural S&OP Master", layout="wide")
st.title("🛡️ 스마트제조 AI 생산전략 관제탑 (S&OP Control Tower)")

# --- [수정] AI 설정: gemini-2.5-flash-lite 반영 ---
def get_ai_consultant(prompt, context):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys: return "⚠️ API 키를 설정해주세요."
        genai.configure(api_key=random.choice(keys))
        # 사용자 요청 모델 반영
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        system_msg = f"당신은 생산관리 전문가입니다. 아래 데이터를 분석하여 경영적 통찰을 제공하세요.\n{context}"
        response = model.generate_content(system_msg + "\n\n질문: " + prompt)
        return response.text
    except Exception as e: return f"❌ AI 오류: {str(e)}"

# --- 사이드바: 공정 제약 조건 추가 ---
with st.sidebar:
    st.header("⚙️ 공정 및 운영 제약")
    opt_mode = st.radio("최적화 알고리즘", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals
    
    st.markdown("---")
    st.subheader("⏱️ 공정 효율 설정")
    std_time = st.slider("제품당 표준 작업 시간(Hr)", 1.0, 10.0, 4.0) # 
    working_days = st.slider("월간 작업 일수", 15, 30, 20) # 

    st.markdown("---")
    st.subheader("💰 비용 변수")
    # (비용 변수 입력창들 생략 - 이전과 동일하게 유지)
    c_reg, c_ot, c_h, c_l, c_inv, c_back, c_mat, c_sub = 640, 6, 300, 500, 2, 5, 10, 30

    demand_raw = st.text_input("수요 예측", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]

# --- 최적화 로직 (공정 효율 변수 반영) ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1); TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
                                c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] for t in T), sense=minimize)
    m.c = ConstraintList()
    m.c.add(m.W[0] == 80); m.c.add(m.I[0] == 1000)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        # 작업 효율 반영: (1/표준시간) * 8시간 * 작업일수 * 인원
        capacity_reg = (1/std_time) * 8 * working_days * m.W[t]
        m.c.add(m.P[t] <= capacity_reg + (1/std_time)*m.O[t])
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])
        m.c.add(m.O[t] <= 10*m.W[t])
    m.c.add(m.I[len(D)] >= 500); m.c.add(m.S[len(D)] == 0)
    SolverFactory('glpk').solve(m)
    return m

# --- 탭 구성 강화 ---
tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크 및 효율 분석", "💬 AI 전략 상담"])

with tab1:
    if st.button("🚀 최적 생산 계획 수립"):
        model = solve_app(demand, domain_type)
        st.session_state['res'] = model
        # (기존 복합 차트 및 KPI 출력)

with tab2:
    st.subheader("⚠️ 운영 리스크 및 자원 활용도")
    if 'res' in st.session_state:
        m = st.session_state['res']
        # 자원 활용도 계산: 정규시간 대비 얼마나 썼나?
        utilization = [(m.P[t]() * std_time) / (working_days * 8 * m.W[t]()) * 100 for t in range(1,7)]
        fig_util = px.line(x=list(range(1,7)), y=utilization, title="월별 생산 설비 가동률 (%)", markers=True)
        st.plotly_chart(fig_util, use_container_width=True)
        st.warning("💡 가동률이 100%를 초과하는 구간은 초과 근무가 필수적인 위험 구간입니다.")

with tab3:
    # (AI 챗봇 인터페이스 - 이전과 동일하게 유지)
