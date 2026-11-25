import plotly.graph_objects as go
import pandas as pd


COLORS = {"A": "#6352A1", "B": "#087877", "C": "#BC481A"}


def _input_rows(fig, recent, model_recent, dates, names, columns, sale_day, previous_sale, calculated):
    # All tracks share the main x axis. Size the y domains in pixels so each series
    # stays readable when switching between the three- and four-track datasets.
    rows = [*names, *(["sale"] if "sale" in recent else [])]
    main_px, gap_px, row_px, row_gap_px = 202, 21, 38, 12
    total_px = main_px + gap_px + len(rows) * (row_px + row_gap_px) - row_gap_px
    main_bottom = 1 - main_px / total_px
    units = {"visitors": "人", "tickets": "件"}
    for number, column in enumerate(rows):
        axis_num = number + 2
        top = main_bottom - (gap_px + number * (row_px + row_gap_px)) / total_px
        bottom = max(0, top - row_px / total_px)
        is_sale = column == "sale"
        used = sale_day is not None if is_sale else column in columns
        if not used:
            status = "未使用" if calculated else "未選択"
        elif not calculated:
            status = "選択中"
        elif is_sale:
            status = "Cで使用"
        else:
            status = "B・Cで使用" if sale_day is not None else "Bで使用"
        name = "セール日" if is_sale else names[column]
        unit = units.get(column, "個")
        sublabel = status if is_sale else f"{unit} · {status}"
        color = (COLORS["C"] if is_sale else COLORS["B"]) if used else "#9AA6A8"
        fig.add_annotation(xref="paper", x=0, xanchor="right", xshift=-8,
                           yref="paper", y=(top + bottom) / 2, yanchor="middle", showarrow=False,
                           align="right", text=f"<b>{name}</b><br><span style='font-size:10px'>{sublabel}</span>",
                           font=dict(size=12, color=color if used else "#768486"))
        axis = dict(domain=[bottom, top], anchor="x", side="right", fixedrange=True,
                    nticks=2, showgrid=False, zeroline=False, tickfont=dict(size=10, color="#657674"))
        yref = f"y{axis_num}"
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="paper", y0=bottom, y1=bottom,
                      line=dict(width=1, color="#E3EAE7"), layer="below")
        if is_sale:
            axis.update(range=[0, 1], showticklabels=False)
            events = recent.loc[recent.sale.eq(1), "date"]
            fig.add_scatter(x=events, y=[.5] * len(events), yaxis=yref, mode="markers",
                            marker=dict(symbol="diamond-open", size=8, color=color), showlegend=False,
                            name="過去のセール", hovertemplate="セール実績<extra></extra>")
            if previous_sale is not None and previous_sale >= 0:
                fig.add_scatter(x=[dates[previous_sale]], y=[.5], yaxis=yref, mode="markers",
                                marker=dict(symbol="circle-open", size=11, color="#775F54"), showlegend=False,
                                name="前回の予定", hovertemplate="前回のセール予定<extra></extra>")
            if sale_day is not None and sale_day >= 0:
                day = dates[sale_day]
                fig.add_scatter(x=[day], y=[.5], yaxis=yref, mode="markers", showlegend=False,
                                marker=dict(symbol="diamond", size=11, color=color), name="セール予定",
                                hovertemplate="今回のセール予定<extra></extra>")
                late = sale_day >= len(dates) - 5
                fig.add_annotation(x=day, y=.5, yref=yref, xshift=-10 if late else 10,
                                   xanchor="right" if late else "left", showarrow=False,
                                   text=f"{day.month}/{day.day} 予定", font=dict(size=11, color=color))
            elif sale_day == -1:
                fig.add_annotation(x=dates[len(dates)//2], y=.5, yref=yref, showarrow=False,
                                   text="セール予定なし", font=dict(size=11, color=color))
        else:
            data = model_recent[column] if used else recent[column]
            fig.add_scatter(x=recent.date, y=data, yaxis=yref, mode="lines", connectgaps=False,
                            line=dict(width=1.8, color=color), showlegend=False, name=name,
                            hovertemplate=f"{name}　%{{y:.0f}} {unit}<extra></extra>")
            missing = recent[column].isna() & data.notna()
            if used and missing.any():
                fig.add_scatter(x=recent.date[missing], y=data[missing], yaxis=yref, mode="markers",
                                marker=dict(size=6, color="white", line=dict(color=color, width=1.4)),
                                showlegend=False, name=f"{name}（補間）",
                                hovertemplate=f"{name}（補間）　%{{y:.1f}} {unit}<extra></extra>")
            if number == 0:
                fig.add_annotation(x=dates[len(dates)//2], y=(top+bottom)/2, yref="paper", showarrow=False,
                                   text="予測期間の販売数・客数は入力しません" if "visitors" in names
                                   else "予測期間の販売数・会計件数は入力しません",
                                   font=dict(size=11, color="#617472"))
        fig.update_layout({f"yaxis{axis_num}": axis})
    fig.update_layout(height=total_px + 92)
    return main_bottom


def forecast_chart(history, dates, values, target, *, model_history, names, columns, sale_day=None,
                   previous=None, actual=None, band_key=None):
    fig = go.Figure()
    recent = history.tail(28)

    def line(x, y, name, color, dash="solid", width=2.6):
        fig.add_scatter(x=x, y=y, name=name, mode="lines", connectgaps=False,
                        line=dict(color=color, width=width, dash=dash),
                        hovertemplate="%{y:.1f} 個<extra>%{fullData.name}</extra>")

    if band_key:
        band = values[band_key]
        fig.add_scatter(x=list(dates) + list(dates[::-1]), y=band["p90"] + band["p10"][::-1],
                        mode="lines", line=dict(width=0), fill="toself", hoverinfo="skip",
                        name=f"{band_key} 予測帯（P10–P90）", fillcolor="rgba(94,110,122,.14)")
    line(recent.date, recent.loaf, f"{target}の履歴", "#354750")
    if previous:
        line(dates, previous[1]["p50"], "C 前回の予定", "#9A7567", "dot")
    for key, forecast in values.items():
        name = {"A": "A 販売履歴のみ", "B": "B 過去情報を追加", "C": "C セール予定を追加"}[key]
        line(dates, forecast["p50"], name, COLORS[key], {"A":"dash", "B":"solid", "C":"solid"}[key])
    if actual is not None:
        fig.add_scatter(x=dates, y=actual, name="店舗の実績", mode="lines+markers", connectgaps=False,
                        line=dict(color="#212A31", width=1.8, dash="dot"), marker=dict(size=4),
                        hovertemplate="%{y:.0f} 個<extra>店舗の実績</extra>")
    split = dates[0] - pd.Timedelta(hours=12)
    main_bottom = _input_rows(fig, recent, model_history.tail(len(recent)), dates, names, columns,
                              sale_day, previous[0] if previous else None, bool(values))
    fig.add_shape(type="line", xref="x", x0=split, x1=split, yref="paper", y0=0, y1=1,
                  line=dict(color="#788D88", width=1, dash="dot"))
    fig.add_annotation(x=split, y=1, yshift=3, yref="paper", text="予測開始", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=11))
    if sale_day is not None and sale_day >= 0:
        day = dates[sale_day]
        fig.add_shape(type="rect", xref="x", x0=day - pd.Timedelta(hours=12), x1=day + pd.Timedelta(hours=12),
                      yref="paper", y0=0, y1=1, fillcolor="#E58848", opacity=.15, line_width=0, layer="below")
    if not values:
        fig.add_annotation(x=dates[len(dates)//2], y=(1+main_bottom)/2, yref="paper", text="「推論」を押すと予測を表示",
                           showarrow=False, font=dict(color="#607675", size=14))
    fig.update_layout(margin=dict(l=112, r=29, t=62, b=30), hovermode="x unified", hoversubplots="axis",
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#30434A", size=13),
                      legend=dict(orientation="h", x=0, y=1, yanchor="top", yref="container",
                                  font=dict(size=12)), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(range=[recent.date.iloc[0], dates[-1] + pd.Timedelta(hours=10)],
                     tickformat="%m/%d", hoverformat="%m/%d", dtick=604800000, tick0=dates[0],
                     anchor="free", position=0, showgrid=False, showline=True, linecolor="#CBD4D1", fixedrange=True)
    fig.layout.yaxis.update(domain=[main_bottom, 1], title=dict(text="販売個数", standoff=7), rangemode="tozero",
                            gridcolor="#E8ECEA", fixedrange=True, automargin=True)
    return fig
