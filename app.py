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

# 2. AI 컨설턴트 로직 (Gemini 2.5-Flash-Lite + 로테이션)
def get_ai_consultant(prompt, context_summary):
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys: return "⚠️ Secrets 설정 확인 필요"
    available_keys = list(keys)
    random.shuffle(available_keys)
    for key in available_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            sys_msg = f"원예장비 제조 APP 전문가로서 다음 데이터를 분석하세요: {context_summary}"
            response = model.generate_content(sys_msg + "\n\n질문: " + prompt)
            return response.text
        except Exception: continue 
    return "❌ AI 연결 실패"

# 3. 사이드바 제어판 (외주 On/Off 삭제, 총 15개 파라미터 유지)
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("⏱️ 공정 효율 및 제약")
    std_time = st.slider("제품당 표준 작업 시간 (Hr)", 1.0, 10.0, 4.0)
    working_days = st.slider("월간 가동 일수", 1, 30, 20)
    ot_limit = st.slider("인당 월간 초과근무 제한 (Hr)", 0, 30, 10)

    st.markdown("---")
    st.subheader("💰 운영 비용 설정 (천원)")
    v_c_reg = st.number_input("정규 임금 (인/월)", value=640)
    v_c_ot  = st.number_input("초과 근무 수당 (Hr)", value=6)
    v_c_h   = st.number_input("신규 고용 비용 (인)", value=300)
    v_c_l   = st.number_input("해고 비용 (인)", value=500)
    v_c_inv = st.number_input("재고 유지비 (개/월)", value=2)
    v_c_back= st.number_input("부재고 비용 (개/월)", value=5)
    v_c_mat = st.number_input("재료비 (개당)", value=10)
    v_c_sub = st.number_input("외주 하청 비용 (개당)", value=30)

    st.markdown("---")
    st.subheader("📈 초기값 및 수요")
    demand_raw = st.text_input("6개월 수요 예측", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    v_w_init, v_i_init, v_i_final = 80, 1000, 500

# 4. 최적화 엔진
def solve_plan(D, domain, reg, ot, h, l, inv, back, mat, sub, stime, wdays, ot_lim, w0, i0, ifinal):
    m = ConcreteModel()
    T = range(1, len(D) + 1); TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(reg*m.W[t] + ot*m.O[t] + h*m.H[t] + l*m.L[t] + 
                                inv*m.I[t] + back*m.S[t] + mat*m.P[t] + sub*m.C[t] for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w0); m.c.add(m.I[0] == i0); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        m.c.add(m.P[t] <= (1/stime) * 8 * wdays * m.W[t] + (1/stime)*m.O[t]) 
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.c.add(m.O[t] <= ot_lim * m.W[t])
    m.c.add(m.I[len(D)] >= ifinal); m.c.add(m.S[len(D)] == 0)
    
    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 세션 관리 (데이터 보존을 위해 초기화)
if 'success' not in st.session_state: st.session_state['success'] = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'res_data' not in st.session_state: st.session_state['res_data'] = {}

# 6. 메인 화면 구성
tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    if st.button("🚀 최적 생산계획 수립 실행"):
        with st.spinner('최적화 연산 중...'):
            try:
                model, sol = solve_plan(demand, domain_type, v_c_reg, v_c_ot, v_c_h, v_c_l, v_c_inv, v_c_back, v_c_mat, v_c_sub, std_time, working_days, ot_limit, v_w_init, v_i_init, v_i_final)
                if sol.solver.termination_condition == TerminationCondition.optimal:
                    # 결과를 세션에 딕셔너리로 저장하여 증발 방지
                    st.session_state['res_data'] = {
                        'cost': model.cost(),
                        'P': [model.P[t]() for t in range(1, 7)],
                        'C': [model.C[t]() for t in range(1, 7)],
                        'I': [model.I[t]() for t in range(1, 7)],
                        'W': [model.W[t]() for t in range(1, 7)],
                        'H_L': sum(model.H[t]() + model.L[t]() for t in range(1, 7)),
                        'utils': [((model.P[t]() * std_time) / (8 * working_days * model.W[t]()) * 100) if (8 * working_days * model.W[t]()) > 0 else 0 for t in range(1, 7)]
                    }
                    st.session_state['success'] = True
                    st.toast("최적화 완료!")
                else: st.error("최적해를 찾을 수 없습니다.")
            except Exception as e: st.error(f"오류: {str(e)}")

    if st.session_state['success']:
        d = st.session_state['res_data']
        k = st.columns(4)
        k[0].metric("총 운영 비용", f"{d['cost']:,.0f}k")
        k[1].metric("평균 가동률", f"{sum(d['utils'])/6:.1f}%")
        k[2].metric("인력 변동 수", f"{d['H_L']:.0f}명")
        k[3].metric("기말 재고", f"{d['I'][5]:,.0f}ea")

        # 메인 그래프 대시보드
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,7)), y=d['P'], name="자체 생산", marker_color='royalblue'))
        fig.add_trace(go.Bar(x=list(range(1,7)), y=d['C'], name="외주 부품 하청", marker_color='lightslategray'))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="예상 수요", line=dict(color='crimson', dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=d['I'], name="재고 수준", yaxis="y2", line=dict(color='orange')))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack', hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💰 비용 세부 구성")
            costs = {"노무비": sum(v_c_reg*w for w in d['W']), "재고비": sum(v_c_inv*i for i in d['I']), "재료비": sum(v_c_mat*p for p in d['P']), "기타": d['cost'] - (sum(v_c_reg*w for w in d['W']) + sum(v_c_inv*i for i in d['I']) + sum(v_c_mat*p for p in d['P']))}
            st.plotly_chart(px.pie(names=list(costs.keys()), values=list(costs.values()), hole=0.4))
        with c2:
            st.subheader("👷 월별 인력 운영")
            st.line_chart(pd.DataFrame({"인원": d['W']}))

with tab2:
    if st.session_state['success']:
        d = st.session_state['res_data']
        st.subheader("⚠️ 운영 리스크 및 가동률 진단")
        fig_risk = px.area(x=list(range(1,7)), y=d['utils'], title="월별 공장 가동률 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red")
        st.plotly_chart(fig_risk, use_container_width=True)
        if max(d['utils']) > 100: st.error(f"과부하 감지: 최대 {max(d['utils']):.1f}%")

with tab3:
    st.subheader("💬 AI 전략 상담방")
    if st.button("🧹 초기화"): st.session_state.messages = []; st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("질문하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        ctx = f"비용:{st.session_state['res_data'].get('cost', 0):,.0f}, 가동률:{st.session_state['res_data'].get('utils', [])}"
        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, ctx)
            st.markdown(ai_res); st.session_state.messages.append({"role": "assistant", "content": ai_res})
