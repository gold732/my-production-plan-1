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

# 2. AI 컨설턴트 로직 (Gemini 2.5-Flash-Lite + API 로테이션)
def get_ai_consultant(prompt, context_summary):
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys: return "⚠️ Secrets에 'GEMINI_KEYS'를 설정해주세요."
    
    available_keys = list(keys)
    random.shuffle(available_keys)
    
    for key in available_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            system_instruction = f"""
            당신은 스마트제조 및 생산관리 전문 컨설턴트입니다. 
            아래 제공되는 최적화 결과 데이터를 분석하여 경영적 통찰을 제공하세요.
            특히 가동률(Utilization)이 100%를 넘는 달은 생산 과부하 리스크가 크므로 이를 강력히 경고해야 합니다.
            
            [현재 최적화 결과 데이터]
            {context_summary}
            """
            response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
            return response.text
        except Exception:
            continue 
    return "❌ AI 연결 오류가 발생했습니다."

# 3. 사이드바: 모든 제어 파라미터 및 외주 전략 토글
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    # [기능 추가] 외주 사용 여부 토글 (강의록 기반 공급망 전략)
    st.markdown("---")
    st.subheader("🏭 공급망 전략")
    enable_outsourcing = st.toggle("외주(Outsourcing) 허용", value=True, help="해제 시 하청 없이 자체 생산으로만 물량을 해결합니다.")

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
    demand_raw = st.text_input("6개월 수요 예측 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    v_w_init = st.number_input("현재 근로자 수", value=80)
    v_i_init = st.number_input("현재고 수준", value=1000)
    v_i_final = st.number_input("기말 목표 재고", value=500)

# 4. 최적화 엔진 (외주 허용 인자 use_sub 반영)
def solve_production_plan(D, domain, reg, ot, h, l, inv, back, mat, sub, stime, wdays, ot_lim, w0, i0, ifinal, use_sub):
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
        cap_reg = (1/stime) * 8 * wdays * m.W[t]
        m.c.add(m.P[t] <= cap_reg + (1/stime)*m.O[t]) 
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.c.add(m.O[t] <= ot_lim * m.W[t])
        # [기능 반영] 외주 미허용 시 하청 수량을 0으로 제약
        if not use_sub:
            m.c.add(m.C[t] == 0)

    m.c.add(m.I[len(D)] >= ifinal); m.c.add(m.S[len(D)] == 0)
    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 세션 상태 관리
if 'messages' not in st.session_state: st.session_state.messages = []
if 'success' not in st.session_state: st.session_state['success'] = False
if 'utils' not in st.session_state: st.session_state['utils'] = []

tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    if st.button("🚀 최적 생산계획 수립 실행"):
        st.session_state['success'] = False
        with st.spinner('데이터 분석 및 최적화 중...'):
            try:
                # 외주 허용 여부 토글 값 전달
                model, sol = solve_production_plan(demand, domain_type, v_c_reg, v_c_ot, v_c_h, v_c_l, v_c_inv, v_c_back, v_c_mat, v_c_sub, std_time, working_days, ot_limit, v_w_init, v_i_init, v_i_final, enable_outsourcing)
                if sol.solver.termination_condition == TerminationCondition.optimal:
                    st.session_state['res'] = model
                    st.session_state['success'] = True
                    # 가동률 정밀 계산 및 전역 저장
                    temp_utils = []
                    for t in range(1, len(demand) + 1):
                        denom = 8 * working_days * model.W[t]()
                        temp_utils.append((model.P[t]() * std_time / denom * 100) if denom > 0 else 0)
                    st.session_state['utils'] = temp_utils
                    st.toast("✅ 최적화 성공!")
                else:
                    st.error("❌ 최적해 없음: 외주를 허용하거나 가동 시간을 조정하세요.")
            except Exception as e:
                st.error(f"⚠️ 오류: {str(e)}")

    if st.session_state.get('success'):
        m, utils = st.session_state['res'], st.session_state['utils']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 운영 비용", f"{m.cost():,.0f}k")
        k2.metric("평균 가동률", f"{sum(utils)/len(utils):.1f}%")
        k3.metric("인력 변동 수", f"{sum(m.H[t]() + m.L[t]() for t in range(1,7)):.0f}명")
        k4.metric("기말 재고량", f"{m.I[6]():,.0f}ea")

        st.subheader("📈 월별 생산/수요/재고 흐름")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,7)), y=[m.P[t]() for t in range(1,7)], name="자체 생산", marker_color='royalblue'))
        fig.add_trace(go.Bar(x=list(range(1,7)), y=[m.C[t]() for t in range(1,7)], name="외주 하청", marker_color='lightslategray'))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="예상 수요", line=dict(color='crimson', dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=[m.I[t]() for t in range(1,7)], name="재고 수준", yaxis="y2", line=dict(color='orange')))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack', hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if st.session_state.get('success'):
        utils = st.session_state['utils']
        st.subheader("⚠️ 운영 리스크 분석 (가동률)")
        fig_risk = px.area(x=list(range(1,7)), y=utils, title="생산 가동률 추이 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red")
        st.plotly_chart(fig_risk, use_container_width=True)
        if max(utils) > 100: st.error(f"🚨 가동 한계 초과 구간 감지: 최대 {max(utils):.1f}%")
        else: st.success("✅ 생산 능력이 수요를 안정적으로 소화 중입니다.")

with tab3:
    st.subheader("💬 AI 전략 상담방")
    if st.button("🧹 대화 내용 초기화"):
        st.session_state.messages = []; st.rerun()
    st.markdown("---")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("가동률 리스크나 최적화 결과에 대해 질문하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        if st.session_state['success']:
            m = st.session_state['res']
            u_str = ", ".join([f"{i+1}월:{val:.1f}%" for i, val in enumerate(st.session_state['utils'])])
            # AI에게 전달할 상세 문맥 데이터 구성
            ctx = f"총비용:{m.cost():,.0f}, 외주허용:{enable_outsourcing}, 상세가동률:[{u_str}], 기말재고:{m.I[6]()}"
        else:
            ctx = "최적화 미수행"

        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, ctx)
            st.markdown(ai_res)
            st.session_state.messages.append({"role": "assistant", "content": ai_res})
