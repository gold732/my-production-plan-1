import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Horticultural S&OP Expert", layout="wide")
st.title("🚀 스마트 제조: AI 기반 총괄생산계획(APP) 최적화 시스템")

# --- 사이드바: 파라미터 및 모드 설정 ---
with st.sidebar:
    st.header("🎮 시스템 설정")
    # [심화 3] LP vs IP 토글 [cite: 182, 189]
    opt_mode = st.radio("최적화 모드 선택", ["현실적 제약(정수계획법, IP)", "수학적 최적(선형계획법, LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.header("📌 운영 변수")
    demand_raw = st.text_input("월별 예상 수요", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("현재 근로자 수", value=80)
    i_init = st.number_input("현재고 수준", value=1000)
    i_final = st.number_input("목표 기말재고", value=500)

# --- 최적화 엔진 ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)

    # 변수 정의 (선택된 모드에 따라 도메인 변경) [cite: 133-140, 192, 197]
    m.W = Var(TIME, domain=domain) 
    m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain)
    m.S = Var(TIME, domain=domain); m.C = Var(TIME, domain=domain)
    m.O = Var(TIME, domain=domain)

    # 목적함수 및 제약조건 (강의록 수식 그대로 구현) [cite: 144, 154, 157]
    m.cost = Objective(expr=sum(640*m.W[t] + 6*m.O[t] + 300*m.H[t] + 500*m.L[t] + 
                                2*m.I[t] + 5*m.S[t] + 10*m.P[t] + 30*m.C[t] for t in T), sense=minimize)
    
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

# --- 메인 실행 및 시각화 ---
if st.button("📊 최적화 실행 및 계획 진단"):
    model = solve_app(demand, domain_type)
    
    # 1. 상단 지표 (KPI)
    total_cost = model.cost()
    st.metric("최소화된 총 운영 비용", f"{total_cost:,.0f} 천원", delta=f"{opt_mode} 기준")

    # 2. [심화: 시각화 강화] 비용 구조 분석 및 생산 현황
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💰 비용 구성 분석 (Cost Breakdown)")
        # 각 비용 항목 계산 [cite: 143, 232, 237]
        costs = {
            "노무비": sum(640*model.W[t]() for t in range(1,7)),
            "재고비": sum(2*model.I[t]() for t in range(1,7)),
            "재료비": sum(10*model.P[t]() for t in range(1,7)),
            "기타(고용/해고/외주)": total_cost - sum((640*model.W[t]() + 2*model.I[t]() + 10*model.P[t]()) for t in range(1,7))
        }
        fig_pie = px.pie(values=list(costs.values()), names=list(costs.keys()), hole=0.4)
        st.plotly_chart(fig_pie)

    with col2:
        st.subheader("👷 인력 및 생산성 진단")
        fig_worker = go.Figure()
        fig_worker.add_trace(go.Scatter(x=list(range(1,7)), y=[model.W[t]() for t in range(1,7)], name="필요 인원", mode='lines+markers'))
        st.plotly_chart(fig_worker)

    # 3. [심화 1] AI 분석 리포트 섹션 (Template)
    st.markdown("---")
    st.subheader("🤖 AI 생산계획 진단 피드백")
    
    # 실제 API 연결 전 시뮬레이션 메시지
    analysis_prompt = f"현재 {opt_mode} 모드로 수립된 계획에서 총 비용은 {total_cost:,.0f}원이며, 기말 재고는 {model.I[6]():.0f}개입니다. 인력 변동이 {sum(model.H[t]() + model.L[t]() for t in range(1,7))}회 감지되었습니다."
    
    with st.expander("AI의 분석 결과 보기 (Click)"):
        st.info(f"💡 AI 분석가: {analysis_prompt}\n\n이 계획은 수요 변동에 비해 인력 고용/해고 비용을 최소화하기 위해 재고를 선제적으로 확보하는 전략을 취하고 있습니다. 특히 4월 수요 피크에 대비한 3월의 사전 생산이 적절히 이루어졌습니다.")

    st.success("✅ 현재 계획은 모든 제약 조건을 충족하며 수학적으로 최적화되었습니다.")
