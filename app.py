import time

import numpy as np
import pandas as pd
import streamlit as st

from charts import forecast_chart
from download_model import MODEL_DIR
from forecast import PAST_COLUMNS, load_model, predict_selected, read_history, read_results
from generate_data import CONTEXT, HORIZON, ORIGINAL
from prepare_real import DAILY, SOURCE


st.set_page_config(page_title="パン屋の販売予測", layout="wide")
st.markdown("""<style>
.block-container{max-width:1350px;padding-top:2.2rem;padding-bottom:2rem}
h1,h2,h3{letter-spacing:-.035em}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:#d7e2df}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def model():
    return load_model("cpu")


def day_label(dates, index):
    if index == -1:
        return "セールなし"
    day = dates[index]
    return f"{day.month}/{day.day}（{'月火水木金土日'[day.weekday()]}）"


def run_comparison(history, columns, sale, replay):
    settings = {"A": ((), None)}
    if columns:
        settings["B"] = (columns, None)
    if sale is not None:
        settings["C"] = (columns, sale)
    values = {}
    start = time.perf_counter()
    if replay:
        bundle = read_results()
        for key in settings:
            values[key] = bundle["C"][str(sale)] if key == "C" else bundle[key]
    else:
        ready = model()
        for key, (past, promo) in settings.items():
            values[key], _ = predict_selected(ready, history, past, promo)
    return {"values": values, "columns": columns, "sale": sale, "replay": replay,
            "elapsed": time.perf_counter() - start}


st.title("パン屋の販売予測")
mode = st.radio("使用するデータ", ["セール日を試す（模擬データ）", "店舗の実績と比べる（実データ）"],
                horizontal=True, label_visibility="collapsed")
real = mode.startswith("店舗")
dataset = "french" if real else "demo"
target = "トラディショナル・バゲット" if real else "食パン"
if real:
    if not DAILY.exists():
        st.info("実店舗の販売記録を用意するには、プロジェクトの仮想環境で次を実行してください。")
        st.code(r".\.venv\Scripts\python.exe prepare_real.py")
        st.stop()
    frame = pd.read_csv(DAILY, parse_dates=["date"])
    if len(frame) != CONTEXT + HORIZON or not frame.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        st.error("日次データの日付が連続していません。prepare_real.py で再作成してください。")
        st.stop()
    history = frame.iloc[:CONTEXT].copy()
    evaluation = frame.iloc[CONTEXT:].loaf.reset_index(drop=True)
    names = {"croissant": "クロワッサン", "chocolate": "パン・オ・ショコラ", "tickets": "会計件数"}
else:
    history = read_history()
    names = {"roll": "ロールパン", "baguette": "バゲット", "visitors": "来店者数"}

model_history = history
if real:
    # Use the same history-only interpolation for the model and its input traces.
    model_history = history.copy()
    numeric = ["loaf", *names]
    model_history[numeric] = history[numeric].interpolate(limit_direction="both")
dates = pd.date_range(history.date.iloc[-1] + pd.Timedelta(days=1), periods=HORIZON)
result_key = f"result_{dataset}"
result = st.session_state.get(result_key)
controls, plot = st.columns([1, 2.55], gap="large")
with controls:
    st.subheader("推論に使うデータ")
    st.checkbox(f"{target}の販売履歴（必須）", value=True, disabled=True, key=f"target_{dataset}")
    st.caption("追加する過去の情報")
    columns = tuple(col for col, name in names.items()
                    if st.checkbox(name, value=result is None or col in result["columns"], key=f"{dataset}_{col}"))
    sale = None
    if not real:
        include_sale = st.checkbox("セール実績と今後の予定", value=result is None or result["sale"] is not None,
                                   key="demo_has_sale")
        old_sale = result["sale"] if result and result["sale"] is not None else ORIGINAL
        choice = st.selectbox("今後のセール日", list(range(-1, HORIZON)), index=old_sale + 1,
                              format_func=lambda i: day_label(dates, i), disabled=not include_sale, key="sale_choice")
        sale = choice if include_sale else None
    replay = False
    has_model = (MODEL_DIR / "model.safetensors").is_file()
    if not real and (columns == PAST_COLUMNS or (not columns and sale is None)):
        with st.expander("実行方法", expanded=not has_model):
            replay = st.checkbox("保存結果を再生する", value=not has_model)
    if not has_model and not replay:
        st.caption("ライブ推論には download_model.py でのモデル準備が必要です。")
    submitted = st.button("保存結果を表示" if replay else "推論", type="primary", width="stretch",
                          disabled=not has_model and not replay)
    if submitted:
        try:
            with st.spinner("保存結果を読み込み中…" if replay else "TimesFM-3 が選択した入力で推論中…"):
                fresh = run_comparison(model_history, columns, sale, replay)
            if (result and "C" in result["values"] and "C" in fresh["values"]
                    and result["columns"] == columns and result["sale"] != sale):
                fresh["previous"] = (result["sale"], result["values"]["C"])
            result = fresh
            st.session_state[result_key] = fresh
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            st.error(f"計算を完了できませんでした。{exc}")
    if result:
        pending = (columns, sale, replay) != (result["columns"], result["sale"], result["replay"])
        if pending:
            st.info("入力を変更しました。右の図は前回の結果です。")
        st.caption(("保存結果を表示" if result["replay"] else "CPU推論完了")
                   + f"　{'・'.join(result['values'])}　{result['elapsed']:.2f} 秒")
    if real:
        st.caption("会計件数は異なるレシート番号の数です。元の記録にはセール予定が含まれていません。")

with plot:
    st.subheader(f"{target}の28日間予測")
    st.caption(f"予測期間　{dates[0]:%Y/%m/%d}〜{dates[-1]:%m/%d}　｜　入力履歴 {len(history)} 日")
    values = result["values"] if result else {}
    applied_sale = result["sale"] if result else None
    previous = result.get("previous") if result else None
    band = None
    if values:
        option_cols = st.columns([1, 1])
        with option_cols[0]:
            if st.checkbox("最新条件の予測帯（P10–P90）", value=True, key=f"band_{dataset}"):
                band = list(values)[-1]
        if real:
            with option_cols[1]:
                show_actual = st.checkbox("店舗の実績を重ねる", value=True)
        elif previous:
            with option_cols[1]:
                if not st.checkbox("前回のセール予定も重ねる", value=True):
                    previous = None
    actual = evaluation if real and values and show_actual else None
    st.plotly_chart(forecast_chart(history, dates, values, target, model_history=model_history, names=names,
                                   columns=result["columns"] if result else columns,
                                   sale_day=applied_sale if result else sale,
                                   actual=actual, previous=previous, band_key=band),
                    width="stretch", theme=None, config={"displayModeBar": False})
    if result:
        past = "・".join(names[k] for k in result["columns"])
        descriptions = {"A": f"{target}の販売履歴", "B": f"A ＋ {past}"}
        parent = "B" if past else "A"
        if applied_sale is not None:
            descriptions["C"] = f"{parent} ＋ セール実績・予定（{day_label(dates, applied_sale)}）"
        if real:
            metric = f"平均絶対誤差（{evaluation.count()}日、個/日）"
        else:
            focus = applied_sale if applied_sale is not None and applied_sale >= 0 else 0
            metric = f"{day_label(dates, focus)} の予測（個）"
        rows = []
        for key, output in values.items():
            score = np.nanmean(np.abs(evaluation - output["p50"])) if real else output["p50"][focus]
            rows.append({"条件": key, "推論に使用した入力": descriptions[key], metric: f"{score:.1f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if previous:
            old_day, old = previous
            differences = [f"{day_label(dates, i)} {values['C']['p50'][i] - old['p50'][i]:+.1f} 個"
                           for i in dict.fromkeys((old_day, applied_sale)) if i >= 0]
            st.write("前回のCからの変化　" + "　／　".join(differences))
        if real:
            st.caption("店舗全体の取引記録がない9/19は欠測として、グラフの実績線と誤差集計から除外しています。")
    else:
        st.caption("Aは販売履歴のみ、Bは選んだ過去情報を追加。" +
                   ("" if real else "Cはさらにセール実績と予定を追加します。"))

with st.expander("入力履歴とデータの出典" if real else "入力履歴を見る"):
    if real:
        st.markdown(f"商品別の数量は返品を含む日次純販売数です。入力区間で全店の記録がない日は、推論時に過去区間内で線形補間します。 [Matthieu Gimbert · French bakery daily sales]({SOURCE})")
    else:
        st.write("発表用に生成した販売履歴です。セール日を変えたときの予測の動きを比較できます。")
    header = {"loaf": target, **names, "sale": "セール（1＝開催）"}
    show = history.set_index("date").rename(columns=header)
    show.index = show.index.strftime("%Y/%m/%d")
    show.index.name = "日付"
    st.dataframe(show.iloc[::-1], width="stretch", height=250)
