#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aux_fetch.py —— M3 GG 变量级 / M5 PINE / M7 EDGAR 数据抓取尝试"""
import io, os, json, urllib.request, urllib.parse, csv

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "02_datasets", "raw")
UA = {'User-Agent': 'paper-data/1.0'}

def try_fetch(url, saveto, tag):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        io.open(saveto, 'wb').write(data)
        print(f"[{tag}] OK {len(data)} bytes → {saveto}")
        return True
    except Exception as e:
        print(f"[{tag}] FAIL: {e}")
        return False

# M3: GG 变量级（能源强度 = ENVRESPROD 主题下的 TPES/GDP 或类似）
# 先探 DSD_GG 的 VAR 维度
print("=== M3: GG variable-level energy intensity ===")
try:
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.ENV.EPI,DSD_GG@DF_GREEN_GROWTH,"
           "1.0/ALL?startPeriod=2000&endPeriod=2020&format=csvfilewithlabels"
           "&dimensionAtObservation=AllDimensions")
    ok = try_fetch(url, os.path.join(OUT, "oecd_gg_varlevel_2000_2020.csv"), "M3-GG-var")
except Exception as e:
    print(f"  GG var-level fail: {e}")

# M5: PINE (Policy Instruments for the Environment)
print("=== M5: PINE ===")
for dfid in ("DSD_PINE@DF_PINE", "DSD_PINE@DF_PINE_FULL", "OECD.ENV.EPI,DSD_PINE@DF_PINE"):
    base = f"https://sdmx.oecd.org/public/rest/data/{dfid}/all?format=csvfilewithlabels"
    if try_fetch(base, os.path.join(OUT, "oecd_pine.csv"), f"M5-{dfid}"):
        break

# M6: OECD National Accounts GDP (real GDP growth, annual)
print("=== M6b: OECD NA GDP growth ===")
try:
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,"
           "1.0/A..Q...?startPeriod=2000&endPeriod=2020&format=csvfilewithlabels")
    ok = try_fetch(url, os.path.join(OUT, "oecd_qna_gdp_2000_2020.csv"), "M6b-NA")
except Exception as e:
    print(f"  NA fail: {e}")

# M7: EDGAR CO2 (all countries, v8 or latest)
print("=== M7: EDGAR ===")
for url, fn in [
    ("https://edgar.jrc.ec.europa.eu/report_2023?emission=CO2", "M7-edgar-co2-page"),
    ("https://ccd.supportportal.jrc.ec.europa.eu_EDGAR/dataset/81p1hh/v8-GHG_booklet_2023.zip", "M7-edgar-v8"),
]:
    tag = fn.split("-", 1)[-1]
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        io.open(os.path.join(OUT, fn), 'wb').write(data)
        print(f"[{tag}] OK {len(data)} bytes")
    except Exception as e:
        print(f"[{tag}] FAIL: {e}")
