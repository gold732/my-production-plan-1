import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="Smart S&OP Expert", layout="wide")
st.title("🚀 AI 기반 원예장비 총괄생산계획(APP) 최적화 시스템")

# --- 사이드바: 모든 운영 변수 복구 ---
with st.sidebar:
    st.header("🎮 시스템 제어")
    opt_mode = st.radio("최적화 알고리즘", ["현실적 제약(정수계획법, IP)", "수학적 최적(선형계획법, LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.header("💰 단위 비용 설정 (천원)")
    c_reg = st.number_input("정규 임금 (월/인)", value=640) # [cite: 144]
    c_ot  = st.number_input("초과 근무 임금 (시간당)", value=6) # [cite: 126]
    c_h   = st.number_input("신규 고용 비용 (인당)", value=300) # [cite: 127]
    c_l   = st.number_input("해고 비용 (인당)", value=500) # [cite: 127]
    c_inv = st.number_input("재고 유지 비용 (개당/월)", value=2) # [cite: 124]
    c_back= st.number_input("부재고(Backlog) 비용", value=5) # [cite: 124]
    c_mat = st.number_input("원자재비 (개당)", value=10) # [cite: 123]
    c_sub = st.number_input("외주 하청 비용 (개당)", value=30) # [cite: 144]

    st.markdown("---")
    st.header("📈 수요 및 초기값")
    demand_raw = st.text_input("6개월 수요 예측", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("초기 인원", value=80) # [cite: 171]
    i_init = st.number_input("초기 재고", value=1000) # [cite: 172]
    i_final = st.number_input("기말 목표 재고", value=500) # [cite: 173]

# --- 최적화 엔진 ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)

    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    # 목적함수: 사용자가 입력한 비용 변수 반영 [cite: 141-144]
    m.cost = Objective(expr=sum(
        c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
        c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] 
        for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w_init); m.c.add(m.I[0] == i_init); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t]) # [cite: 149]
        m.c.add(m.P[t] <= 40*m.W[t] + 0.25*m.O[t])     # [cite: 154]
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) # [cite: 157]
        m.c.add(m.O[t] <= 10*m.W[t])                   # [cite: 160]
    m.c.add(m.I[len(D)] >= i_final); m.c.add(m.S[len(D)] == 0)

    SolverFactory('glpk').solve(m)
    return m

# --- 결과 출력 ---
if st.button("📊 생산 계획 최적화 및 AI 진단 실행"):
    model = solve_app(demand, domain_type)
    
    # KPI 지표
    cols = st.columns(4)
    cols[0].metric("총 운영 비용", f"{model.cost():,.0f}k")
    cols[1].metric("평균 재고량", f"{sum(model.I[t]() for t in range(1,7))/6:,.0f}ea")
    cols[2].metric("총 고용/해고", f"{sum(model.H[t]() + model.L[t]() for t in range(1,7)):.0f}명")
    cols[3].metric("외주 물량", f"{sum(model.C[t]() for t in range(1,7)):,.0f}ea")

    # 시각화 1: 수요 vs 생산 vs 재고 (복합 차트)
    st.subheader("📈 생산 및 재고 흐름 진단")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1,7)), y=[model.P[t]() for t in range(1,7)], name="자체생산"))
    fig.add_trace(go.Bar(x=list(range(1,7)), y=[model.C[t]() for t in range(1,7)], name="외주처리"))
    fig.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="예상수요", line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=list(range(1,7)), y=[model.I[t]() for t in range(1,7)], name="재고수준", yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

    # 시각화 2: 비용 구조 (Pie)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("💰 비용 세부 항목")
        cost_df = pd.DataFrame({
            "항목": ["노무비", "재고비", "재료비", "고용/해고", "외주비"],
            "비용": [sum(c_reg*model.W[t]() for t in range(1,7)), 
                   sum(c_inv*model.I[t]() for t in range(1,7)),
                   sum(c_mat*model.P[t]() for t in range(1,7)),
                   sum(c_h*model.H[t]() + c_l*model.L[t]() for t in range(1,7)),
                   sum(c_sub*model.C[t]() for t in range(1,7))]
        })
        st.plotly_chart(px.pie(cost_df, values='비용', names='항목', hole=0.3))

    with col_b:
        # [추가 대시보드] 인력 계획 적절성
        st.subheader("👷 인력 운영 안정성")
        st.line_chart(pd.DataFrame({"작업자 수": [model.W[t]() for t in range(1,7)]}))

    # --- [심화] AI 분석 리포트 (통찰력 강화) ---
    st.markdown("---")
    st.subheader("🤖 AI S&OP 전문가의 분석 리포트")
    
    # 진단 로직 (간이 AI)
    hiring_total = sum(model.H[t]() for t in range(1,7))
    if hiring_total > (w_init * 0.2):
        ai_msg = "⚠️ 인력 변동성이 매우 높습니다. 고용 비용보다 재고 유지비가 저렴하다면 생산 평준화를 고려하십시오."
    elif sum(model.C[t]() for t in range(1,7)) > 0:
        ai_msg = "💡 자체 생산 능력이 부족하여 외주를 활용 중입니다. 장기적으로 설비 확충 검토가 필요합니다."
    else:
        ai_msg = "✅ 현재 계획은 안정적인 인력 운영과 최적의 재고 수준을 유지하고 있습니다."

    st.info(f"**진단 결과:** {ai_msg}")
    with st.expander("📝 상세 분석 내용 보기"):
        st.write(f"""
        1. **모드**: 현재 **{opt_mode}**를 적용하여 { '소수점 없는 실제 인원' if "IP" in opt_mode else '수학적 한계치' }를 계산했습니다.
        2. **비용 효율**: 총 비용 중 {max(cost_df['비용'])/model.cost()*100:,.1f}%가 {cost_df.loc[cost_df['비용'].idxmax(), '항목']}에 집중되어 있습니다.
        3. **재고 전략**: {i_final}개의 목표 기말 재고를 맞추기 위해 { '외주' if sum(model.C[t]() for t in range(1,7))>0 else '잔업' } 전략을 적절히 혼합했습니다.
        """)
