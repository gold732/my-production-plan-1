import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 페이지 설정
st.set_page_config(page_title="Horticultural S&OP Control Tower", layout="wide")
st.title("🚀 AI 기반 생산계획(APP) 최적화 관제 시스템")

# --- AI 설정 및 API 로테이션 ---
def get_ai_consultant(prompt, context):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys: return "⚠️ Secrets에 API 키를 설정해주세요."
        
        # API 설정 및 모델 호출 (에러 방지를 위해 모델명 재확인)
        genai.configure(api_key=random.choice(keys))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_msg = f"""당신은 스마트제조 생산관리 전문가입니다. 
        사용자가 수립한 총괄생산계획(APP) 최적화 데이터를 바탕으로 통찰력을 제공하세요.
        [최적화 결과 요약]: {context}
        사용자의 질문에 대해 비용 효율성과 운영 안정성 측면에서 답변하세요."""
        
        response = model.generate_content(system_msg + "\n\n질문: " + prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 서비스 연결 오류: {str(e)}"

# --- 사이드바: 모든 운영 변수 복구 ---
with st.sidebar:
    st.header("🎮 시스템 제어")
    opt_mode = st.radio("최적화 알고리즘", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.header("💰 단위 비용 설정 (천원)")
    c_reg = st.number_input("정규 임금 (월)", value=640) 
    c_ot  = st.number_input("초과 근무 (시간)", value=6) 
    c_h   = st.number_input("신규 고용 비용", value=300) 
    c_l   = st.number_input("해고 비용", value=500) 
    c_inv = st.number_input("재고 유지 비용", value=2) 
    c_back= st.number_input("부재고(Backlog) 비용", value=5) 
    c_mat = st.number_input("원자재비 (개당)", value=10) 
    c_sub = st.number_input("외주 하청 비용", value=30) 

    st.markdown("---")
    st.header("📈 수요 및 초기값")
    demand_raw = st.text_input("6개월 수요 예측", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("초기 인원", value=80) 
    i_init = st.number_input("초기 재고", value=1000) 
    i_final = st.number_input("목표 기말 재고", value=500)

# --- 최적화 로직 ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
                                c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w_init); m.c.add(m.I[0] == i_init); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.c.add(m.P[t] <= 40*m.W[t] + 0.25*m.O[t])
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])
        m.c.add(m.O[t] <= 10*m.W[t])
    m.c.add(m.I[len(D)] >= i_final); m.c.add(m.S[len(D)] == 0)
    SolverFactory('glpk').solve(m)
    return m

# --- 메인 화면 탭 구성 ---
tab1, tab2 = st.tabs(["📊 통합 분석 대시보드", "💬 AI 전략 컨설턴트"])

with tab1:
    if st.button("🚀 최적화 실행 및 결과 업데이트"):
        model = solve_app(demand, domain_type)
        st.session_state['res'] = model
        
        # 1. KPI 메트릭
        kpis = st.columns(4)
        kpis[0].metric("총 운영 비용", f"{model.cost():,.0f}k")
        kpis[1].metric("평균 재고", f"{sum(model.I[t]() for t in range(1,7))/6:,.0f}ea")
        kpis[2].metric("총 인력 변동", f"{sum(model.H[t]() + model.L[t]() for t in range(1,7)):.0f}명")
        kpis[3].metric("외주 처리량", f"{sum(model.C[t]() for t in range(1,7)):,.0f}ea")

        # 2. 메인 복합 차트
        st.subheader("📈 공급망 흐름 진단 (수요/생산/재고)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,7)), y=[model.P[t]() for t in range(1,7)], name="자체 생산", marker_color='royalblue'))
        fig.add_trace(go.Bar(x=list(range(1,7)), y=[model.C[t]() for t in range(1,7)], name="외주 하청", marker_color='lightslategray'))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="예상 수요", line=dict(color='crimson', width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=[model.I[t]() for t in range(1,7)], name="재고 수준", yaxis="y2", line=dict(color='orange', width=3)))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # 3. 비용 파이 차트 & 인력 선 그래프
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("💰 비용 세부 구성")
            cost_data = {
                "노무비": sum(c_reg*model.W[t]() for t in range(1,7)),
                "재고비": sum(c_inv*model.I[t]() for t
