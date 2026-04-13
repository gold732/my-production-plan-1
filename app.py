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

# 2. AI 컨설턴트 로직
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
            특히 가동률(Utilization)이 100%를 넘는 달은 생산 과부하 리스크가 크므로 이를 강력히 경고하고 대안을 제시해야 합니다.
            
            [현재 최적화 결과 데이터]
            {context_summary}
            """
            response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
            return response.text
        except Exception as e:
            continue 
    return "❌ AI 연결 오류가 발생했습니다."

# 3. 사이드바: 제어 파라미터
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("🏭 외주 전략")
    enable_outsourcing = st.toggle("외주(Outsourcing) 허용", value=True)

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

# 4. 최적화 엔진
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
        if not use_sub: m.c.add(m.C[t] == 0)

    m.c.add(m.I[len(D)] >= ifinal); m.c.add(m.S[len(D)] == 0)
    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 세션 상태 및 탭 구성
if 'messages' not in st.session_state: st.session_state.messages = []
if 'success' not in st.session_state: st.session_state['success'] = False
if 'utils' not in st.session_state: st.session_state['utils'] = []

tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    if st.button("🚀 최적 생산계획 수립 실행"):
        st.session_state['success'] = False
        with st.spinner('계산 중...'):
            try:
                model, sol = solve_production_plan(demand, domain_type, v_c_reg, v_c_ot, v_c_h, v_c_l, v_c_inv, v_c_back, v_c_mat, v_c_sub, std_time, working_days, ot_limit, v_w_init, v_i_init, v_i_final, enable_outsourcing)
                if sol.solver.termination_condition == TerminationCondition.optimal:
                    st.session_state['res'] = model
                    st.session_state['success'] = True
                    temp_utils = []
                    for t in range(1, len(demand) + 1):
                        reg_cap_hrs = 8 * working_days * model.W[t]()
                        # 가동률 = (생산에 투입된 총 시간 / 정규 근무 가능 시간) * 100
                        temp_utils.append((model.P[t]() * std_time / reg_cap_hrs * 100) if reg_cap_hrs > 0 else 0)
                    st.session_state['utils'] = temp_utils
                    st.toast("✅ 최적화 완료!")
                else: st.error("❌ 해를 찾을 수 없습니다.")
            except Exception as e: st.error(f"⚠️ 오류: {str(e)}")

    if st.session_state.get('success'):
        m = st.session_state['res']
        utils = st.session_state['utils']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 운영 비용", f"{m.cost():,.0f}k")
        k2.metric("평균 가동률", f"{sum(utils)/len(utils):.1f}%")
        k3.metric("인력 변동", f"{sum(m.H[t]() + m.L[t]() for t in range(1,len(demand)+1)):.0f}명")
        k4.metric("기말 재고", f"{m.I[len(demand)]():,.0f}ea")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.P[t]() for t in range(1,len(demand)+1)], name="자체 생산"))
        fig.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.C[t]() for t in range(1,len(demand)+1)], name="외주 하청"))
        fig.add_trace(go.Scatter(x=list(range(1,len(demand)+1)), y=demand, name="수요 예측", line=dict(color='red', dash='dash')))
        fig.update_layout(barmode='stack', title="월별 생산 및 수요 현황")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if st.session_state.get('success'):
        utils = st.session_state['utils']
        st.subheader("⚠️ 운영 리스크 분석 (가동률)")
        
        # [수정] 가동률이 100%를 넘어도 잘리지 않도록 Y축 범위 자동 설정
        y_max = max(max(utils) * 1.2, 120) # 데이터 최대값의 120% 또는 최소 120%까지 표시
        
        fig_risk = px.line(x=list(range(1,len(demand)+1)), y=utils, title="생산 가동률 추이 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="정규 능력 한계 (100%)")
        
        # Y축 범위 명시적 설정
        fig_risk.update_yaxes(range=[0, y_max], title="가동률 (%)")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        if max(utils) > 100:
            st.warning(f"🚨 초과 근무 발생: 특정 달의 가동률이 100%를 상회합니다 (최대 {max(utils):.1f}%).")
        else:
            st.success("✅ 모든 구간이 정규 생산 능력 범위 내에 있습니다.")

with tab3:
    st.subheader("💬 AI 전략 상담방")
    if st.button("🧹 대화 초기화"): st.session_state.messages = []; st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("가동률 리스크에 대해 분석해줘."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        if st.session_state['success']:
            m = st.session_state['res']
            u_str = ", ".join([f"{i+1}월:{v:.1f}%" for i, v in enumerate(st.session_state['utils'])])
            ctx = f"총비용:{m.cost():,.0f}, 외주허용:{enable_outsourcing}, 월별 가동률:[{u_str}]"
        else: ctx = "데이터 없음"

        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, ctx)
            st.markdown(ai_res)
            st.session_state.messages.append({"role": "assistant", "content": ai_res})
