from flask import Flask, jsonify, render_template_string
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback

app = Flask(__name__)

def fetch_data():
    """Fetch latest 90 days of data from FRED"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    try:
        # Fetch all data
        tnx = web.DataReader('DGS10', 'fred', start_date, end_date)
        sp500 = web.DataReader('SP500', 'fred', start_date, end_date)
        nasdaq = web.DataReader('NASDAQCOM', 'fred', start_date, end_date)
        dji = web.DataReader('DJIA', 'fred', start_date, end_date)

        # Merge on date index to handle different trading calendars
        df = pd.DataFrame({'tnx': tnx['DGS10']})
        df = df.join(sp500['SP500'].rename('sp500'), how='outer')
        df = df.join(nasdaq['NASDAQCOM'].rename('nasdaq'), how='outer')
        df = df.join(dji['DJIA'].rename('dji'), how='outer')
        df = df.sort_index()
        df.index.name = 'date'
        df = df.reset_index()

        # Calculate daily changes
        df['tnx_change'] = df['tnx'].diff() * 100  # basis points
        df['sp500_change'] = df['sp500'].pct_change() * 100
        df['nasdaq_change'] = df['nasdaq'].pct_change() * 100
        df['dji_change'] = df['dji'].pct_change() * 100

        # Normalize to cumulative % change
        first_valid = df.dropna(subset=['tnx', 'sp500', 'nasdaq']).iloc[0]
        df['tnx_pct'] = (df['tnx'] - first_valid['tnx']) / first_valid['tnx'] * 100
        df['sp500_pct'] = (df['sp500'] - first_valid['sp500']) / first_valid['sp500'] * 100
        df['nasdaq_pct'] = (df['nasdaq'] - first_valid['nasdaq']) / first_valid['nasdaq'] * 100
        df['dji_pct'] = (df['dji'] - first_valid['dji']) / first_valid['dji'] * 100

        # Convert to list of dicts for JSON
        records = []
        for _, row in df.iterrows():
            records.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'tnx': round(float(row['tnx']), 3) if pd.notna(row['tnx']) else None,
                'tnx_change': round(float(row['tnx_change']), 1) if pd.notna(row['tnx_change']) else None,
                'tnx_pct': round(float(row['tnx_pct']), 2) if pd.notna(row['tnx_pct']) else None,
                'sp500': round(float(row['sp500']), 2) if pd.notna(row['sp500']) else None,
                'sp500_change': round(float(row['sp500_change']), 2) if pd.notna(row['sp500_change']) else None,
                'sp500_pct': round(float(row['sp500_pct']), 2) if pd.notna(row['sp500_pct']) else None,
                'nasdaq': round(float(row['nasdaq']), 2) if pd.notna(row['nasdaq']) else None,
                'nasdaq_change': round(float(row['nasdaq_change']), 2) if pd.notna(row['nasdaq_change']) else None,
                'nasdaq_pct': round(float(row['nasdaq_pct']), 2) if pd.notna(row['nasdaq_pct']) else None,
                'dji': round(float(row['dji']), 2) if pd.notna(row['dji']) else None,
                'dji_change': round(float(row['dji_change']), 2) if pd.notna(row['dji_change']) else None,
                'dji_pct': round(float(row['dji_pct']), 2) if pd.notna(row['dji_pct']) else None,
            })

        # Calculate summary statistics
        valid = df.dropna(subset=['tnx_change', 'sp500_change', 'nasdaq_change'])
        corr_df = valid[['tnx_change', 'sp500_change', 'nasdaq_change', 'dji_change']].dropna()
        corr = corr_df.corr()

        # Count opposite direction days
        opposite_days = sum(1 for _, r in valid.iterrows()
                           if (r['tnx_change'] > 0 and r['sp500_change'] < 0) or
                              (r['tnx_change'] < 0 and r['sp500_change'] > 0))

        summary = {
            'total_days': len(valid),
            'tnx_start': round(float(df['tnx'].dropna().iloc[0]), 3),
            'tnx_end': round(float(df['tnx'].dropna().iloc[-1]), 3),
            'tnx_high': round(float(df['tnx'].dropna().max()), 3),
            'tnx_low': round(float(df['tnx'].dropna().min()), 3),
            'tnx_change': round(float(df['tnx'].dropna().iloc[-1] - df['tnx'].dropna().iloc[0]), 3),
            'sp500_start': round(float(df['sp500'].dropna().iloc[0]), 2),
            'sp500_end': round(float(df['sp500'].dropna().iloc[-1]), 2),
            'sp500_change_pct': round(float((df['sp500'].dropna().iloc[-1] / df['sp500'].dropna().iloc[0] - 1) * 100), 2),
            'nasdaq_start': round(float(df['nasdaq'].dropna().iloc[0]), 2),
            'nasdaq_end': round(float(df['nasdaq'].dropna().iloc[-1]), 2),
            'nasdaq_change_pct': round(float((df['nasdaq'].dropna().iloc[-1] / df['nasdaq'].dropna().iloc[0] - 1) * 100), 2),
            'dji_start': round(float(df['dji'].dropna().iloc[0]), 2),
            'dji_end': round(float(df['dji'].dropna().iloc[-1]), 2),
            'dji_change_pct': round(float((df['dji'].dropna().iloc[-1] / df['dji'].dropna().iloc[0] - 1) * 100), 2),
            'correlation': {
                'tnx_sp500': round(float(corr.loc['tnx_change', 'sp500_change']), 4),
                'tnx_nasdaq': round(float(corr.loc['tnx_change', 'nasdaq_change']), 4),
                'tnx_dji': round(float(corr.loc['tnx_change', 'dji_change']), 4),
                'sp500_nasdaq': round(float(corr.loc['sp500_change', 'nasdaq_change']), 4),
            },
            'opposite_days': int(opposite_days),
            'opposite_pct': round(opposite_days / len(valid) * 100, 1) if len(valid) > 0 else 0,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': f"{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}"
        }

        return {'data': records, 'summary': summary}

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


@app.route('/api/data')
def api_data():
    result = fetch_data()
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TACO曲线</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e27;--bg2:#111638;--card:rgba(17,22,56,0.85);
  --glass:rgba(255,255,255,0.04);--glass-border:rgba(255,255,255,0.08);
  --primary:#6c63ff;--primary-glow:rgba(108,99,255,0.3);
  --accent:#ff6b6b;--accent-glow:rgba(255,107,107,0.3);
  --success:#51cf66;--success-glow:rgba(81,207,102,0.3);
  --warning:#fcc419;--warning-glow:rgba(252,196,25,0.3);
  --info:#339af0;--info-glow:rgba(51,154,240,0.3);
  --orange:#ff922b;--orange-glow:rgba(255,146,43,0.3);
  --text:#e8eaf6;--text2:#9fa8da;--muted:#5c6bc0;
  --radius:16px;
}
html{scroll-behavior:smooth}
body{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;
}
/* Animated background */
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 600px 600px at 20% 20%, rgba(108,99,255,0.08) 0%, transparent 70%),
    radial-gradient(ellipse 500px 500px at 80% 80%, rgba(255,107,107,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 400px 400px at 50% 50%, rgba(51,154,240,0.05) 0%, transparent 70%);
  animation: bgShift 20s ease-in-out infinite alternate;
}
@keyframes bgShift{
  0%{opacity:1;transform:scale(1)}
  100%{opacity:.7;transform:scale(1.1)}
}

/* Navbar */
.navbar{
  position:sticky;top:0;z-index:50;
  background:rgba(10,14,39,0.8);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--glass-border);
  padding:16px 0;
}
.navbar-inner{
  max-width:1200px;margin:0 auto;padding:0 24px;
  display:flex;align-items:center;justify-content:space-between;
}
.navbar-brand{
  display:flex;align-items:center;gap:10px;
  font-weight:800;font-size:1.4rem;color:#fff;text-decoration:none;letter-spacing:-0.5px;
}
.navbar-brand .logo-icon{
  width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,var(--primary),var(--accent));
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
  box-shadow:0 4px 15px var(--primary-glow);
}
.navbar-meta{font-size:.8rem;color:var(--text2);display:flex;align-items:center;gap:6px}
.navbar-meta .live-dot{
  width:8px;height:8px;border-radius:50%;background:var(--success);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}

/* Container */
.container{max-width:1200px;margin:0 auto;padding:24px;position:relative;z-index:1}

/* Section Title */
.section-title{
  font-size:1rem;font-weight:700;color:var(--text);margin-bottom:20px;
  display:flex;align-items:center;gap:10px;
  padding:10px 16px;border-radius:10px;
  background:var(--glass);border:1px solid var(--glass-border);
  backdrop-filter:blur(10px);
}
.section-title .icon-wrap{
  width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;
  font-size:.85rem;
}
.section-title .icon-wrap.purple{background:var(--primary-glow);color:var(--primary)}
.section-title .icon-wrap.blue{background:var(--info-glow);color:var(--info)}
.section-title .icon-wrap.orange{background:var(--orange-glow);color:var(--orange)}

/* Glass Card */
.glass-card{
  background:var(--card);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border:1px solid var(--glass-border);border-radius:var(--radius);
  transition:all .3s ease;overflow:hidden;
}
.glass-card:hover{border-color:rgba(255,255,255,0.12);transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,0.3)}
.glass-card-header{
  padding:16px 24px;border-bottom:1px solid var(--glass-border);
  font-weight:600;font-size:.9rem;color:var(--text2);
  display:flex;align-items:center;gap:8px;
}
.glass-card-header i{font-size:1rem}
.glass-card-body{padding:20px 24px}

/* Stat Cards */
.stat-card{
  text-align:center;padding:24px 16px;position:relative;overflow:hidden;
}
.stat-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  border-radius:var(--radius) var(--radius) 0 0;
}
.stat-card.tnx::before{background:linear-gradient(90deg,var(--accent),#ff8a80)}
.stat-card.sp::before{background:linear-gradient(90deg,var(--info),#74c0fc)}
.stat-card.nq::before{background:linear-gradient(90deg,var(--orange),#ffd43b)}
.stat-card.dji::before{background:linear-gradient(90deg,var(--success),#8ce99a)}
.stat-icon{
  width:40px;height:40px;border-radius:12px;margin:0 auto 12px;
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
}
.stat-card.tnx .stat-icon{background:var(--accent-glow);color:var(--accent)}
.stat-card.sp .stat-icon{background:var(--info-glow);color:var(--info)}
.stat-card.nq .stat-icon{background:var(--orange-glow);color:var(--orange)}
.stat-card.dji .stat-icon{background:var(--success-glow);color:var(--success)}
.stat-label{font-size:.78rem;color:var(--text2);font-weight:500;letter-spacing:.3px}
.stat-value{font-size:1.9rem;font-weight:800;line-height:1.3;margin-top:6px;letter-spacing:-0.5px}
.stat-card.tnx .stat-value{color:var(--accent)}
.stat-card.sp .stat-value{color:var(--info)}
.stat-card.nq .stat-value{color:var(--orange)}
.stat-card.dji .stat-value{color:var(--success)}
.stat-change{
  font-size:.85rem;font-weight:600;margin-top:8px;
  display:inline-flex;align-items:center;gap:4px;
  padding:3px 10px;border-radius:20px;
}
.stat-change.up{color:var(--success);background:rgba(81,207,102,0.1)}
.stat-change.down{color:var(--accent);background:rgba(255,107,107,0.1)}
.stat-change.flat{color:var(--text2);background:var(--glass)}

/* Chart */
.chart-container{position:relative;height:420px;padding:8px 0}
.chart-container canvas{border-radius:8px}

/* Correlation Badges */
.corr-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}
.corr-item{
  text-align:center;padding:20px 12px;border-radius:14px;
  background:var(--glass);border:1px solid var(--glass-border);
  transition:all .3s;
}
.corr-item:hover{border-color:rgba(255,255,255,0.15);background:rgba(255,255,255,0.06)}
.corr-item .corr-label{font-size:.78rem;color:var(--text2);margin-bottom:8px;font-weight:500}
.corr-item .corr-value{
  font-size:1.5rem;font-weight:800;letter-spacing:-0.5px;
  padding:6px 16px;border-radius:10px;display:inline-block;
}
.corr-value.neg{color:var(--accent);background:rgba(255,107,107,0.1)}
.corr-value.pos{color:var(--success);background:rgba(81,207,102,0.1)}
.corr-item .corr-desc{font-size:.72rem;color:var(--muted);margin-top:6px}

/* Insight & Advice Boxes */
.insight-box{
  padding:18px 22px;margin-bottom:14px;border-radius:12px;
  background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(51,154,240,0.06));
  border:1px solid rgba(108,99,255,0.15);
  transition:all .3s;
}
.insight-box:hover{border-color:rgba(108,99,255,0.3);background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(51,154,240,0.08))}
.insight-box strong{color:var(--primary);display:flex;align-items:center;gap:8px;font-size:.92rem}
.insight-box p{color:var(--text2);font-size:.88rem;line-height:1.7}

.advice-box{
  padding:18px 22px;margin-bottom:14px;border-radius:12px;
  background:linear-gradient(135deg,rgba(252,196,25,0.06),rgba(255,146,43,0.04));
  border:1px solid rgba(252,196,25,0.12);
  transition:all .3s;
}
.advice-box:hover{border-color:rgba(252,196,25,0.25)}
.advice-box strong{color:var(--warning);display:flex;align-items:center;gap:8px;font-size:.92rem}
.advice-box p{color:var(--text2);font-size:.88rem;line-height:1.7}
.advice-box.disclaimer{
  background:rgba(255,107,107,0.06);border-color:rgba(255,107,107,0.12);
}
.advice-box.disclaimer strong{color:var(--accent)}
.advice-box.disclaimer p{font-size:.82rem;color:var(--muted)}

/* Refresh Button */
.refresh-btn{
  position:fixed;bottom:28px;right:28px;width:52px;height:52px;border-radius:16px;
  background:linear-gradient(135deg,var(--primary),#8b83ff);color:#fff;border:none;
  font-size:1.2rem;cursor:pointer;z-index:100;
  box-shadow:0 8px 24px var(--primary-glow);
  transition:all .3s;display:flex;align-items:center;justify-content:center;
}
.refresh-btn:hover{transform:translateY(-3px) scale(1.05);box-shadow:0 12px 32px rgba(108,99,255,0.5)}
.refresh-btn.spinning i{animation:spin 1s linear infinite}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

/* Footer */
footer{
  text-align:center;padding:32px 24px;color:var(--muted);font-size:.8rem;
  position:relative;z-index:1;
}
footer a{color:var(--primary);text-decoration:none}
footer a:hover{text-decoration:underline}

/* Grid */
.row{display:flex;gap:16px;flex-wrap:wrap}
.row>.col-6{flex:0 0 calc(50% - 8px);max-width:calc(50% - 8px)}
.row>.col-md-3{flex:0 0 calc(25% - 12px);max-width:calc(25% - 12px)}
.mb-4{margin-bottom:24px}
.mb-3{margin-bottom:12px}
.p-4{padding:20px 24px}
.mt-1{margin-top:4px}
.mt-2{margin-top:8px}

/* Loading skeleton */
.skeleton{background:linear-gradient(90deg,var(--glass) 25%,rgba(255,255,255,0.08) 50%,var(--glass) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:6px;height:1em;display:inline-block;min-width:60px}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* Responsive */
@media(max-width:768px){
  .row>.col-md-3{flex:0 0 calc(50% - 8px);max-width:calc(50% - 8px)}
  .stat-value{font-size:1.5rem!important}
  .chart-container{height:300px}
  .corr-grid{grid-template-columns:1fr}
  .container{padding:16px}
  .navbar-inner{padding:0 16px}
}
@media(max-width:480px){
  .row>.col-6,.row>.col-md-3{flex:0 0 100%;max-width:100%}
}
</style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
  <div class="navbar-inner">
    <a class="navbar-brand" href="#">
      <div class="logo-icon"><i class="bi bi-graph-up-arrow"></i></div>
      TACO曲线
    </a>
    <div class="navbar-meta">
      <span class="live-dot"></span>
      <span id="updateTime">加载中...</span>
    </div>
  </div>
</nav>

<div class="container" style="padding-top:28px">

  <!-- ===== MODULE 1: DATA TRACKING ===== -->
  <div class="mb-4">
    <div class="section-title">
      <span class="icon-wrap purple"><i class="bi bi-bar-chart-line"></i></span>
      实时数据追踪
    </div>

    <!-- Summary Cards -->
    <div class="row mb-4">
      <div class="col-6 col-md-3">
        <div class="glass-card stat-card tnx">
          <div class="stat-icon"><i class="bi bi-bank"></i></div>
          <div class="stat-label">10年期美债利率</div>
          <div class="stat-value" id="tnxValue">--</div>
          <div class="stat-change" id="tnxChange">--</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="glass-card stat-card sp">
          <div class="stat-icon"><i class="bi bi-graph-up"></i></div>
          <div class="stat-label">标普500</div>
          <div class="stat-value" id="sp500Value">--</div>
          <div class="stat-change" id="sp500Change">--</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="glass-card stat-card nq">
          <div class="stat-icon"><i class="bi bi-cpu"></i></div>
          <div class="stat-label">纳斯达克</div>
          <div class="stat-value" id="nasdaqValue">--</div>
          <div class="stat-change" id="nasdaqChange">--</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="glass-card stat-card dji">
          <div class="stat-icon"><i class="bi bi-building"></i></div>
          <div class="stat-label">道琼斯</div>
          <div class="stat-value" id="djiValue">--</div>
          <div class="stat-change" id="djiChange">--</div>
        </div>
      </div>
    </div>

    <!-- Chart 1: Correlation Line Chart -->
    <div class="glass-card mb-4">
      <div class="glass-card-header">
        <i class="bi bi-activity" style="color:var(--primary)"></i>
        美债利率 vs 美股走势关联性对比（累计变动%）
      </div>
      <div class="glass-card-body">
        <div class="chart-container">
          <canvas id="correlationChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Chart 2: Bar Chart -->
    <div class="glass-card mb-4">
      <div class="glass-card-header">
        <i class="bi bi-bar-chart" style="color:var(--accent)"></i>
        美债利率 vs 标普500 每日涨跌对比
      </div>
      <div class="glass-card-body">
        <div class="chart-container">
          <canvas id="barChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== MODULE 2: CORRELATION ANALYSIS ===== -->
  <div class="mb-4">
    <div class="section-title">
      <span class="icon-wrap blue"><i class="bi bi-diagram-3"></i></span>
      关联规律分析
    </div>
    <div class="glass-card">
      <div class="glass-card-body p-4">
        <div class="corr-grid">
          <div class="corr-item">
            <div class="corr-label">美债利率 vs 标普500</div>
            <div class="corr-value neg" id="corrSP">--</div>
            <div class="corr-desc">皮尔逊相关系数</div>
          </div>
          <div class="corr-item">
            <div class="corr-label">美债利率 vs 纳斯达克</div>
            <div class="corr-value neg" id="corrNQ">--</div>
            <div class="corr-desc">皮尔逊相关系数</div>
          </div>
          <div class="corr-item">
            <div class="corr-label">反向变动天数占比</div>
            <div class="corr-value neg" id="oppositePct">--</div>
            <div class="corr-desc">利率与标普方向相反</div>
          </div>
        </div>
        <div id="insightsContainer">
          <div class="insight-box">
            <strong><i class="bi bi-lightbulb"></i> 核心发现</strong>
            <p class="mb-0 mt-2" id="insight1"><span class="skeleton"></span> <span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
          <div class="insight-box">
            <strong><i class="bi bi-arrow-left-right"></i> 跷跷板效应</strong>
            <p class="mb-0 mt-2" id="insight2"><span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
          <div class="insight-box">
            <strong><i class="bi bi-speedometer2"></i> 弹性差异</strong>
            <p class="mb-0 mt-2" id="insight3"><span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== MODULE 3: INVESTMENT ADVICE ===== -->
  <div class="mb-4">
    <div class="section-title">
      <span class="icon-wrap orange"><i class="bi bi-shield-check"></i></span>
      投资建议
    </div>
    <div class="glass-card">
      <div class="glass-card-body p-4">
        <div class="advice-box disclaimer">
          <strong><i class="bi bi-exclamation-triangle"></i> 免责声明</strong>
          <p class="mb-0 mt-1">以下建议基于历史数据统计分析，仅供参考，不构成任何投资建议。投资有风险，决策需谨慎。</p>
        </div>
        <div id="adviceContainer">
          <div class="advice-box">
            <strong><i class="bi bi-1-circle"></i> 利率上行阶段策略</strong>
            <p class="mb-0 mt-2" id="advice1"><span class="skeleton"></span> <span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
          <div class="advice-box">
            <strong><i class="bi bi-2-circle"></i> 利率下行/拐点阶段策略</strong>
            <p class="mb-0 mt-2" id="advice2"><span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
          <div class="advice-box">
            <strong><i class="bi bi-3-circle"></i> 配置建议</strong>
            <p class="mb-0 mt-2" id="advice3"><span class="skeleton"></span> <span class="skeleton"></span> <span class="skeleton"></span></p>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<footer>
  数据来源：<a href="https://fred.stlouisfed.org" target="_blank">FRED</a> (Federal Reserve Economic Data) &nbsp;|&nbsp; 每次访问实时获取最新数据
</footer>

<button class="refresh-btn" id="refreshBtn" title="刷新数据" onclick="loadData()">
  <i class="bi bi-arrow-clockwise"></i>
</button>

<script>
let correlationChart = null;
let barChart = null;

function fmt(val, decimals=2) {
  if (val === null || val === undefined) return '--';
  return Number(val).toFixed(decimals);
}

function fmtChange(val, unit='') {
  if (val === null || val === undefined) return '--';
  const n = Number(val);
  const prefix = n > 0 ? '+' : '';
  return prefix + n.toFixed(2) + unit;
}

function changeClass(val) {
  return val > 0 ? 'up' : val < 0 ? 'down' : 'flat';
}

function loadData() {
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('spinning');

  fetch('/api/data')
    .then(r => r.json())
    .then(result => {
      if (result.error) {
        alert('数据加载失败: ' + result.error);
        return;
      }
      renderSummary(result.summary);
      renderCharts(result.data);
      renderAnalysis(result.summary);
      renderAdvice(result.summary);
      document.getElementById('updateTime').textContent =
        '更新于 ' + result.summary.last_update + ' | ' + result.summary.date_range;
    })
    .catch(err => {
      alert('请求失败: ' + err.message);
    })
    .finally(() => {
      btn.classList.remove('spinning');
    });
}

function renderSummary(s) {
  document.getElementById('tnxValue').textContent = s.tnx_end + '%';
  document.getElementById('tnxChange').textContent = fmtChange(s.tnx_change, ' bp');
  document.getElementById('tnxChange').className = 'stat-change ' + changeClass(s.tnx_change);

  document.getElementById('sp500Value').textContent = fmt(s.sp500_end, 0);
  document.getElementById('sp500Change').textContent = fmtChange(s.sp500_change_pct, '%');
  document.getElementById('sp500Change').className = 'stat-change ' + changeClass(s.sp500_change_pct);

  document.getElementById('nasdaqValue').textContent = fmt(s.nasdaq_end, 0);
  document.getElementById('nasdaqChange').textContent = fmtChange(s.nasdaq_change_pct, '%');
  document.getElementById('nasdaqChange').className = 'stat-change ' + changeClass(s.nasdaq_change_pct);

  document.getElementById('djiValue').textContent = fmt(s.dji_end, 0);
  document.getElementById('djiChange').textContent = fmtChange(s.dji_change_pct, '%');
  document.getElementById('djiChange').className = 'stat-change ' + changeClass(s.dji_change_pct);
}

function renderCharts(data) {
  const dates = data.map(d => d.date.substring(5));
  const validData = data.filter(d => d.tnx_pct !== null && d.sp500_pct !== null && d.nasdaq_pct !== null);
  const validDates = validData.map(d => d.date.substring(5));
  const barData = data.filter(d => d.tnx_change !== null);

  // ---- Correlation Line Chart ----
  const ctx1 = document.getElementById('correlationChart').getContext('2d');
  if (correlationChart) correlationChart.destroy();

  correlationChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: validDates,
      datasets: [
        {
          label: '10年期美债利率',
          data: validData.map(d => d.tnx_pct),
          borderColor: '#D32F2F',
          backgroundColor: 'rgba(211,47,47,0.08)',
          fill: true,
          tension: 0.3,
          borderWidth: 2.5,
          pointRadius: 2,
          pointHoverRadius: 5,
          yAxisID: 'y'
        },
        {
          label: '标普500',
          data: validData.map(d => d.sp500_pct),
          borderColor: '#1565C0',
          backgroundColor: 'transparent',
          tension: 0.3,
          borderWidth: 2.2,
          pointRadius: 2,
          pointHoverRadius: 5,
          yAxisID: 'y'
        },
        {
          label: '纳斯达克',
          data: validData.map(d => d.nasdaq_pct),
          borderColor: '#FF8F00',
          backgroundColor: 'transparent',
          tension: 0.3,
          borderWidth: 2.2,
          pointRadius: 2,
          pointHoverRadius: 5,
          borderDash: [5, 3],
          yAxisID: 'y'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: '#9fa8da', font: { size: 12 }, usePointStyle: true, pointStyle: 'circle' } },
        tooltip: {
          backgroundColor: 'rgba(17,22,56,0.95)',
          titleColor: '#e8eaf6', bodyColor: '#9fa8da',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          cornerRadius: 10, padding: 12,
          callbacks: {
            label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + '%'
          }
        }
      },
      scales: {
        x: {
          ticks: { maxRotation: 45, font: { size: 10 }, color: '#5c6bc0' },
          grid: { display: false }
        },
        y: {
          title: { display: true, text: '累计变动 (%)', font: { weight: 'bold' }, color: '#9fa8da' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5c6bc0' }
        }
      }
    }
  });

  // ---- Bar Chart ----
  const ctx2 = document.getElementById('barChart').getContext('2d');
  if (barChart) barChart.destroy();

  barChart = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: barData.map(d => d.date.substring(5)),
      datasets: [
        {
          label: '美债利率变动 (bp)',
          data: barData.map(d => d.tnx_change),
          backgroundColor: barData.map(d => d.tnx_change >= 0 ? 'rgba(211,47,47,0.8)' : 'rgba(211,47,47,0.3)'),
          borderColor: barData.map(d => d.tnx_change >= 0 ? '#D32F2F' : '#EF9A9A'),
          borderWidth: 1,
          yAxisID: 'yBp',
          order: 2
        },
        {
          label: '标普500变动 (%)',
          data: barData.map(d => d.sp500_change),
          backgroundColor: barData.map(d => d.sp500_change >= 0 ? 'rgba(21,101,192,0.8)' : 'rgba(21,101,192,0.3)'),
          borderColor: barData.map(d => d.sp500_change >= 0 ? '#1565C0' : '#90CAF9'),
          borderWidth: 1,
          yAxisID: 'yPct',
          order: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: '#9fa8da', font: { size: 12 }, usePointStyle: true, pointStyle: 'circle' } },
        tooltip: {
          backgroundColor: 'rgba(17,22,56,0.95)',
          titleColor: '#e8eaf6', bodyColor: '#9fa8da',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          cornerRadius: 10, padding: 12,
          callbacks: {
            label: ctx => {
              const unit = ctx.datasetIndex === 0 ? 'bp' : '%';
              return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + unit;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { maxRotation: 55, font: { size: 9 }, color: '#5c6bc0' },
          grid: { display: false },
          stacked: false
        },
        yBp: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: '美债利率 (bp)', color: '#ff6b6b', font: { weight: 'bold' } },
          ticks: { color: '#ff6b6b' },
          grid: { color: 'rgba(255,255,255,0.04)' }
        },
        yPct: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: '标普500 (%)', color: '#339af0', font: { weight: 'bold' } },
          ticks: { color: '#339af0' },
          grid: { display: false }
        }
      }
    }
  });
}

function renderAnalysis(s) {
  const corrSP = s.correlation.tnx_sp500;
  const corrNQ = s.correlation.tnx_nasdaq;

  document.getElementById('corrSP').textContent = 'r = ' + corrSP.toFixed(2);
  document.getElementById('corrNQ').textContent = 'r = ' + corrNQ.toFixed(2);
  document.getElementById('oppositePct').textContent = s.opposite_pct + '%';

  const strength = Math.abs(corrSP) > 0.7 ? '强' : Math.abs(corrSP) > 0.4 ? '中等' : '较弱';
  const direction = corrSP < 0 ? '负' : '正';

  document.getElementById('insight1').innerHTML =
    `在过去 <strong>${s.total_days}</strong> 个交易日中，10年期美债利率与标普500的相关系数为 <strong>r = ${corrSP.toFixed(2)}</strong>，` +
    `与纳斯达克的相关系数为 <strong>r = ${corrNQ.toFixed(2)}</strong>，呈<strong>${direction}相关</strong>关系，` +
    `相关强度为<strong>${strength}</strong>。`;

  document.getElementById('insight2').innerHTML =
    `在${s.total_days}个交易日中，美债利率与标普500呈<strong>反向变动</strong>的天数达 <strong>${s.opposite_days}天</strong>（占${s.opposite_pct}%），` +
    `即利率上涨时股市往往下跌，利率下跌时股市往往上涨，"跷跷板效应"非常显著。`;

  const nqStronger = s.nasdaq_change_pct > s.sp500_change_pct;
  document.getElementById('insight3').innerHTML =
    `本期纳斯达克累计变动 <strong>${fmtChange(s.nasdaq_change_pct, '%')}</strong>，` +
    `标普500累计变动 <strong>${fmtChange(s.sp500_change_pct, '%')}</strong>。` +
    (nqStronger ? '纳斯达克弹性更大，对利率变动更敏感。' : '标普500表现更稳健。') +
    ` 这反映了成长股（科技股）对利率环境变化的高敏感性。`;
}

function renderAdvice(s) {
  const corr = s.correlation.tnx_sp500;
  const tnxTrend = s.tnx_change > 0 ? 'up' : s.tnx_change < 0 ? 'down' : 'flat';

  let advice1Text, advice2Text, advice3Text;

  if (tnxTrend === 'up') {
    advice1Text = `当前美债利率处于<strong>上行趋势</strong>（期间累计上升 ${fmtChange(s.tnx_change, 'bp')}），` +
      `根据历史负相关规律，股市可能继续承压。建议：<br>` +
      `<strong>1)</strong> 适度降低股票仓位，增加现金或短债配置比例；<br>` +
      `<strong>2)</strong> 规避高估值成长股，关注利率敏感度低的防御性板块（公用事业、必需消费品）；<br>` +
      `<strong>3)</strong> 可考虑配置浮动利率债券，对冲利率上行风险。`;
  } else {
    advice1Text = `当前美债利率处于<strong>下行/平稳趋势</strong>（期间累计变动 ${fmtChange(s.tnx_change, 'bp')}），` +
      `利率环境对股市相对友好。建议：<br>` +
      `<strong>1)</strong> 可维持或适度增加股票仓位；<br>` +
      `<strong>2)</strong> 关注受益于利率下行的板块（科技、房地产、公用事业）；<br>` +
      `<strong>3)</strong> 可考虑配置长端国债，锁定较高票息。`;
  }

  advice2Text = `当观察到美债利率<strong>连续3天以上回落</strong>且跌幅超过10bp时，往往是股市反弹的先行信号。` +
    `此时可逐步加仓，优先考虑超跌的成长股和科技股。反之，当利率<strong>快速飙升</strong>（单日>8bp）时，` +
    `应警惕短期回调风险，建议设置止损或使用期权对冲。`;

  advice3Text = `基于当前利率与股市的相关性（r=${corr.toFixed(2)}），建议采用<strong>"股债平衡"策略</strong>：<br>` +
    `<strong>1)</strong> 核心仓位（60%）：宽基指数ETF（如SPY、QQQ），跟随市场整体走势；<br>` +
    `<strong>2)</strong> 卫星仓位（20%）：根据利率趋势灵活调整，利率上行时增配短债ETF（SHV），利率下行时增配长债ETF（TLT）；<br>` +
    `<strong>3)</strong> 现金储备（20%）：等待利率拐点出现时进行再平衡操作。`;

  document.getElementById('advice1').innerHTML = advice1Text;
  document.getElementById('advice2').innerHTML = advice2Text;
  document.getElementById('advice3').innerHTML = advice3Text;
}

// Initial load
document.addEventListener('DOMContentLoaded', loadData);
</script>

</body>
</html>
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
