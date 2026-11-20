"""경북 주변 기상청 ASOS 24개 지점의 일자료(1973-01-01 ~ 어제)를 지점별 파일로 미리 저장

  python data/prepare_stations.py            # 24개 지점 모두 (처음 한 번 30~60분, 이미 받은 지점은 최근 며칠만)
  python data/prepare_stations.py 277 279    # 일부 지점만 (지점번호)

저장 위치: data/asos/asos_<지점번호>.csv  (지점당 약 1MB)
이 폴더를 GitHub에 올리면 웹앱이 처음 보는 저수지도 기상청 접속 없이 몇 초 만에 기상 자료를 만든다.
인증키: .streamlit/secrets.toml 의 KMA_APIHUB_KEY (또는 DATA_GO_KR_KEY)
중간에 끊겨도 다시 실행하면 받은 지점은 건너뛰고 이어서 받는다.
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import kma  # noqa: E402

if not kma.get_key():
    sys.exit("기상청 인증키가 없습니다. .streamlit/secrets.toml 에 KMA_APIHUB_KEY 를 넣으세요.")

stns = [int(a) for a in sys.argv[1:]] or list(kma.ASOS_STATIONS)
lines = []
t_all = time.time()
for i, stn in enumerate(stns, 1):
    name = kma.ASOS_STATIONS.get(stn, ("?",))[0]
    t0 = time.time()
    print(f"[{i}/{len(stns)}] {name}({stn}) 받는 중…", flush=True)
    try:
        info = kma.update_station(stn, start="1973-01-01")
        msg = f"  ✅ {name}({stn}) {info['first']} ~ {info['last']} · {info['days']:,}일 · {time.time() - t0:.0f}초"
    except Exception as e:
        msg = f"  ❌ {name}({stn}) 실패 — {e}"
        traceback.print_exc()
    print(msg, flush=True)
    lines.append(msg)
size = sum(p.stat().st_size for p in kma.STN_DIR.glob("asos_*.csv")) / 1e6
summary = f"\n완료: {len(stns)}개 지점 · 전체 {time.time() - t_all:.0f}초 · data/asos 폴더 {size:.1f}MB"
print(summary)
(ROOT / "stations_result.txt").write_text("\n".join(lines) + summary, encoding="utf-8")
print("결과 요약: stations_result.txt — GitHub에는 data/asos 폴더를 올리세요.")
