import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 1. 페이지 설정
st.set_page_config(page_title="AI S&OP Control Tower", layout="wide")
st.title("🛡️ 스마트제조 AI 생산전략 관제탑 (S&OP Master)")

# 2. AI 컨설턴트 (Gemini 2.5-Flash-Lite)
def get_ai_consultant(prompt, context_summary):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys: return "⚠️ Secrets에 API 키를 설정해주세요."
        genai.configure(api_key=random.choice(keys))
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        system_instruction = f"당신은 생산관리 전문가입니다. 데이터 기반으로 조언하세요.\n[데이터]: {context_summary}"
        response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
        return response.text
    except Exception as e: return f"❌ AI 오류: {str(e)}"

# 3. 사이드바 제어판 [cite: 121-129, 171-176]
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals
    
    st.markdown("---")
    st.subheader("⏱️ 공정 효율")
    std_time = st.slider("제품당 작업 시간(Hr)", 1.0, 10.0, 4.0)
    working_days = st.slider("월간 가동 일수", 0, 30, 20) # 0으로 설정 시 에러 방지 필요
    ot_limit = st.slider("인당 초과근무 제한(Hr)", 0, 30, 10)

    st.markdown("---")
    st.subheader("💰 비용 변수")
    # 변수들을 딕셔너리로 관리하여 코드 간소화
    c = {
        "reg": st.number_input("정규 임금", value=640), "ot": st.number_input("초과 근무", value=6),
        "h": st.number_input("고용비", value=300), "l": st.number_input("해고비", value=500),
        "inv": st.number_input("재고비", value=2), "back": st.number_input("부재고비", value=5),
        "mat": st.number_input("재료비", value=10), "sub": st.number_input("외주비", value=30)
    }
    demand_raw = st.text_input("수요 예측", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init, i_init, i_final = 80, 1000, 500

# 4. 최적화 엔진 [cite: 130-160]
def solve_production_plan(D, domain):
    m = ConcreteModel()
    T, TIME = range(1, len(D) + 1), range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(c["reg"]*m.W[t] + c["ot"]*m.O[t] + c["h"]*m.H[t] + c["l"]*m.L[t] + 
                                c["inv"]*m.I[t] + c["back"]*m.S[t] + c["mat"]*m.P[t] + c["sub"]*m.C[t] for t in T), sense=minimize)
    
    m.cons = ConstraintList()
    m.cons.add(m.W[0] == w_init); m.cons.add(m.I[0] == i_init); m.cons.add(m.S[0] == 0)
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        cap_reg = (1/std_time) * 8 * working_days * m.W[t]
        m.cons.add(m.P[t] <= cap_reg + (1/std_time)*m.O[t]) 
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.cons.add(m.O[t] <= ot_limit * m.W[t])
    m.cons.add(m.I[len(D)] >= i_final); m.cons.add(m.S[len(D)] == 0)

    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 메인 UI
tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담"])

if st.button("🚀 최적 생산계획 수립 실행"):
    with st.spinner('계산 중...'):
        model, sol = solve_production_plan(demand, domain_type)
        if sol.solver.termination_condition == TerminationCondition.optimal:
            st.session_state['res'] = model
            st.session_state['success'] = True
        else:
            st.session_state['success'] = False
            st.error("❌ 최적해를 찾을 수 없습니다. 제약 조건이나 수요를 조정해주세요.")

# 데이터가 성공적으로 도출된 경우에만 대시보드 출력
if st.session_state.get('success'):
    m = st.session_state['res']
    
    # [에러 방지용] 안전한 가동률 리스트 계산
    utils = []
    for t in range(1, 7):
        denom = 8 * working_days * m.W[t]()
        utils.append((m.P[t]() * std_time / denom * 100) if denom > 0 else 0)

    with tab1:
        kpis = st.columns(4)
        kpis[0].metric("총 운영 비용", f"{m.cost():,.0f}k")
        kpis[1].metric("최종 재고", f"{m.I[6]():,.0f}ea")
        kpis[2].metric("인력 변동", f"{sum(m.H[t]() + m.L[t]() for t in range(1,7)):.0f}명")
        kpis[3].metric("평균 가동률", f"{sum(utils)/6:.1f}%") # 안전하게 계산된 값 사용

        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(x=list(range(1,7)), y=[m.P[t]() for t in range(1,7)], name="생산", marker_color='royalblue'))
        fig_main.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="수요", line=dict(color='crimson', dash='dash')))
        fig_main.add_trace(go.Scatter(x=list(range(1,7)), y=[m.I[t]() for t in range(1,7)], name="재고", yaxis="y2", line=dict(color='orange')))
        fig_main.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack')
        st.plotly_chart(fig_main, use_container_width=True)

    with tab2:
        st.subheader("⚠️ 가동률 및 리스크 추이")
        fig_risk = px.area(x=list(range(1,7)), y=utils, title="생산 설비 가동률 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red")
        st.plotly_chart(fig_risk, use_container_width=True)

    with tab3:
        # AI 상담방 로직 (기존과 동일)
        st.subheader("💬 AI 전문가 상담")
        if 'messages' not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("질문하세요."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            ctx = f"총비용: {m.cost():,.0f}, 가동률: {sum(utils)/6:.1f}%"
            with st.chat_message("assistant"):
                ai_res = get_ai_consultant(prompt, ctx)
                st.markdown(ai_res)
                st.session_state.messages.append({"role": "assistant", "content": ai_res})
