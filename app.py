import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="AI S&OP Control Tower", layout="wide")
st.title("🛡️ 스마트제조 AI 생산전략 관제탑 (S&OP Master)")

# 2. AI 컨설턴트 로직 (기존 유지)
def get_ai_consultant(prompt, context_summary):
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys: return "⚠️ Secrets에 'GEMINI_KEYS'를 설정해주세요."
    available_keys = list(keys)
    random.shuffle(available_keys)
    for key in available_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            system_instruction = f"당신은 생산관리 전문가입니다. 다음 데이터를 분석하세요: {context_summary}"
            response = model.generate_content(system_instruction + "\n\n질문: " + prompt)
            return response.text
        except Exception: continue 
    return "❌ AI 연결 오류"

# 3. 사이드바: 파라미터 제어
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("💰 비용 구조 설정 (천원)")
    v_c_mat = st.number_input("기본 재료비 (개당)", value=10, help="자체 생산 및 외주 시 공통으로 발생하는 원가")
    v_c_sub_premium = st.number_input("외주 가공 프리미엄 (개당)", value=30, help="외주 시 재료비 외에 추가로 지불하는 가공 수수료")
    
    st.info(f"💡 외주 총 단가: {v_c_mat + v_c_sub_premium} (재료비 + 프리미엄)")

    v_c_reg = st.number_input("정규 임금 (인/월)", value=640)
    v_c_ot  = st.number_input("초과 근무 수당 (Hr)", value=6)
    v_c_h   = st.number_input("신규 고용 비용 (인)", value=300)
    v_c_l   = st.number_input("해고 비용 (인)", value=500)
    v_c_inv = st.number_input("재고 유지비 (개/월)", value=2)
    v_c_back= st.number_input("부재고 비용 (개/월)", value=5)

    st.markdown("---")
    st.subheader("📈 수요 및 초기값")
    demand_raw = st.text_input("수요 예측 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    v_w_init = st.number_input("현재 근로자 수", value=80)
    v_i_init = st.number_input("현재고 수준", value=1000)
    v_i_final = st.number_input("기말 목표 재고", value=500)

# 4. 최적화 엔진
def solve_production_plan(D, domain, reg, ot, h, l, inv, back, mat, premium, w0, i0, ifinal):
    m = ConcreteModel()
    T = range(1, len(D) + 1); TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    # [핵심 수정] 외주 비용 = (재료비 + 프리미엄) * 외주량
    total_sub_cost = mat + premium
    
    m.cost = Objective(expr=sum(reg*m.W[t] + ot*m.O[t] + h*m.H[t] + l*m.L[t] + 
                                inv*m.I[t] + back*m.S[t] + mat*m.P[t] + total_sub_cost*m.C[t] for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w0); m.c.add(m.I[0] == i0); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.c.add(m.P[t] <= 40*m.W[t] + 0.25*m.O[t]) 
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.c.add(m.O[t] <= 10 * m.W[t])
    m.c.add(m.I[len(D)] >= ifinal); m.c.add(m.S[len(D)] == 0)

    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 메인 UI (기존 유지)
if 'success' not in st.session_state: st.session_state['success'] = False
tab1, tab2 = st.tabs(["📊 대시보드", "💬 AI 상담"])

with tab1:
    if st.button("🚀 최적 생산계획 수립"):
        model, sol = solve_production_plan(demand, domain_type, v_c_reg, v_c_ot, v_c_h, v_c_l, v_c_inv, v_c_back, v_c_mat, v_c_sub_premium, v_w_init, v_i_init, v_i_final)
        if sol.solver.termination_condition == TerminationCondition.optimal:
            st.session_state['res'] = model
            st.session_state['success'] = True
            st.success("최적화 성공!")

    if st.session_state.get('success'):
        m = st.session_state['res']
        st.metric("총 운영 비용 (재료비+프리미엄 합산)", f"{m.cost():,.0f}k")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.P[t]() for t in range(1,len(demand)+1)], name="자체생산"))
        fig.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.C[t]() for t in range(1,len(demand)+1)], name="외주하청(프리미엄 적용)"))
        st.plotly_chart(fig, use_container_width=True)
