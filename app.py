import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from flask import Flask, request, render_template_string
from shapely.ops import unary_union, linemerge
from folium.plugins import MarkerCluster

# =========================================================
# 기본 설정
# =========================================================
GRAPH_FILE = "okcheon_walk_network.graphml"
LIGHT_FILE = "충청북도_옥천군_보안등정보.csv"
CCTV_FILE = "CCTV정보_충북옥천군.csv"
ACCIDENT_FILE = "옥천군_보행자교통사고다발지역_2021-2024.csv"

VWORLD_KEY = "인증키"  # 기존에 사용하던 VWorld 인증키 입력
BUFFER_DISTANCE = 30
ACCIDENT_DISTANCE = 100

app = Flask(__name__)

# =========================================================
# 데이터 불러오기
# =========================================================
print("보행자 도로망과 시설물 데이터를 불러옵니다...")
G = ox.io.load_graphml(GRAPH_FILE)

light_df = pd.read_csv(LIGHT_FILE, encoding="utf-8-sig")
for col in ["위도", "경도", "설치개수"]:
    light_df[col] = pd.to_numeric(light_df[col], errors="coerce")
light_df["설치개수"] = light_df["설치개수"].fillna(1)
light_df = light_df.dropna(subset=["위도", "경도"])
lights = gpd.GeoDataFrame(
    light_df,
    geometry=gpd.points_from_xy(light_df["경도"], light_df["위도"]),
    crs="EPSG:4326"
).to_crs(5179)

cctv_df = pd.read_csv(CCTV_FILE, encoding="cp949")
for col in ["WGS84위도", "WGS84경도", "카메라대수"]:
    cctv_df[col] = pd.to_numeric(cctv_df[col], errors="coerce")
cctv_df["카메라대수"] = cctv_df["카메라대수"].fillna(1)
cctv_df = cctv_df.dropna(subset=["WGS84위도", "WGS84경도"])
cctvs = gpd.GeoDataFrame(
    cctv_df,
    geometry=gpd.points_from_xy(
        cctv_df["WGS84경도"], cctv_df["WGS84위도"]
    ),
    crs="EPSG:4326"
).to_crs(5179)

accident_df = pd.read_csv(ACCIDENT_FILE, encoding="utf-8-sig")
for col in ["lo_crd", "la_crd"]:
    accident_df[col] = pd.to_numeric(accident_df[col], errors="coerce")
accident_df = accident_df.dropna(subset=["lo_crd", "la_crd"])
accident_df = accident_df.drop_duplicates(subset=["lo_crd", "la_crd"])
accidents = gpd.GeoDataFrame(
    accident_df,
    geometry=gpd.points_from_xy(
        accident_df["lo_crd"], accident_df["la_crd"]
    ),
    crs="EPSG:4326"
).to_crs(5179)
accident_union = unary_union(accidents.geometry)

# 지도의 최초 중심을 도로망 노드의 중심으로 설정
node_lats = [float(data["y"]) for _, data in G.nodes(data=True)]
node_lons = [float(data["x"]) for _, data in G.nodes(data=True)]
MAP_CENTER = [36.306, 127.57]  # 옥천읍 중심부 근사 좌표

# =========================================================
# 도로별 안전 시설 고려 가중치 계산
# 서버 실행 때 한 번만 계산
# =========================================================
print("도로 구간별 시설물 분포를 분석합니다...")
edges_proj = ox.graph_to_gdfs(
    G, nodes=False, fill_edge_geometry=True
).to_crs(5179)

light_index = lights.sindex
cctv_index = cctvs.sindex
accident_areas = unary_union(
    accidents.geometry.buffer(ACCIDENT_DISTANCE)
)

# 각 도로 구간의 시설물 밀도와 사고 다발지역 노출률을 미리 계산합니다.
edge_features = {}
for (u, v, key), row in edges_proj.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    length = max(float(row["length"]), 1.0)
    corridor = geom.buffer(BUFFER_DISTANCE)
    light_ids = light_index.query(corridor, predicate="intersects")
    cctv_ids = cctv_index.query(corridor, predicate="intersects")
    light_density = len(set(light_ids)) / (length / 100.0)
    cctv_density = len(set(cctv_ids)) / (length / 100.0)
    exposed = geom.intersection(accident_areas)
    exposed_length = float(exposed.length) if not exposed.is_empty else 0.0
    exposure_ratio = min(max(exposed_length / length, 0.0), 1.0)

    # 보안등 밀도를 어두움의 반대 지표로 사용합니다.
    # CCTV는 보조 지표로만 반영하고, 사고 노출은 별도 감점합니다.
    darkness_index = 1.0 / (1.0 + light_density)
    cctv_credit = min(cctv_density, 1.0) * 0.10
    edge_features[(u, v, key)] = {
        "light_density_100m": light_density,
        "cctv_density_100m": cctv_density,
        "darkness_index": darkness_index,
        "cctv_credit": cctv_credit,
        "accident_exposure": exposure_ratio,
    }

print(f"도로 구간 분석 완료: {len(edge_features):,}개 구간")
G_safe = G.copy()  # 기존 자동 검증 기능 호환용\nprint("분석 데이터 준비 완료!")


# =========================================================
# 공통 함수
# =========================================================
def analyze_facilities(area):
    selected_lights = lights[lights.geometry.within(area)].copy()
    selected_cctvs = cctvs[cctvs.geometry.within(area)].copy()
    return selected_lights, selected_cctvs


def analyze_route_safety(line):
    """경로 전체를 같은 기준으로 평가합니다."""
    accident_areas = unary_union(accidents.geometry.buffer(ACCIDENT_DISTANCE))
    exposed = line.intersection(accident_areas)
    exposed_length = float(exposed.length) if not exposed.is_empty else 0.0
    nearest_accident = (
        float(line.distance(accidents.geometry.union_all()))
        if len(accidents) else None
    )
    return {
        "accident_exposed_m": exposed_length,
        "accident_exposed_pct": (
            exposed_length / float(line.length) * 100
            if line.length else 0.0
        ),
        "nearest_accident_m": nearest_accident
    }


def as_route_line(geometry):
    """도로 구간을 가능한 하나의 연속 경로선으로 정리합니다."""
    if geometry is None or geometry.is_empty:
        return geometry
    try:
        merged = linemerge(geometry)
        return merged
    except Exception:
        return geometry


def analyze_route_lighting(line):
    """30m 경로 회랑 안의 보안등 설치 지점 수와 경로상 최대 간격을 계산합니다.

    밀도 분자는 카드의 '보안등 위치'와 동일한 설치 지점 수를 사용합니다.
    최대 공백은 각 보안등을 경로에 투영한 위치 사이의 간격을 근사합니다.
    """
    line = as_route_line(line)
    route_length = float(line.length) if line is not None else 0.0
    if route_length <= 0:
        return {"light_sites": 0, "light_density_100m": 0.0,
                "max_light_gap_m": 0.0, "max_gap_segment": None}

    nearby = lights[lights.geometry.within(line.buffer(BUFFER_DISTANCE))].copy()
    site_count = int(len(nearby))
    density = site_count / (route_length / 100.0)

    # MultiLineString도 처리할 수 있도록 구성요소별 누적 길이를 사용합니다.
    components = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    components = [g for g in components if not g.is_empty and g.length > 0]
    offsets = []
    cumulative = 0.0
    for component in components:
        offsets.append(cumulative)
        cumulative += float(component.length)

    positions = []
    for point in nearby.geometry:
        candidates = []
        for component, offset in zip(components, offsets):
            if component.distance(point) <= BUFFER_DISTANCE:
                candidates.append((component.distance(point), offset + component.project(point)))
        if candidates:
            positions.append(min(candidates, key=lambda x: x[0])[1])

    # 같은 시설물의 중복 투영만 완화하되, 밀도용 설치 지점 수는 줄이지 않습니다.
    positions.sort()
    unique_positions = []
    for pos in positions:
        if not unique_positions or pos - unique_positions[-1] > 2.0:
            unique_positions.append(pos)

    gaps = []
    previous = 0.0
    for pos in unique_positions:
        gaps.append((max(0.0, pos - previous), previous, pos))
        previous = pos
    gaps.append((max(0.0, route_length - previous), previous, route_length))
    max_gap, gap_start, gap_end = max(gaps, key=lambda x: x[0]) if gaps else (route_length, 0.0, route_length)

    # 최대 공백 구간을 지도에 표시할 수 있도록 선분을 만듭니다.
    gap_segment = None
    if components and len(components) == 1:
        try:
            gap_segment = components[0].interpolate(gap_start), components[0].interpolate(gap_end)
        except Exception:
            gap_segment = None

    return {
        "light_sites": site_count,
        "light_density_100m": density,
        "max_light_gap_m": max_gap,
        "max_gap_segment": gap_segment,
    }


def build_weighted_graph(preference):
    """안전 우선 정도(0~100)에 따라 구간별 비용을 구성한 그래프를 반환합니다."""
    weighted = G.copy()
    alpha = max(0.0, min(100.0, float(preference))) / 100.0 * 5.0
    for u, v, key, data in weighted.edges(keys=True, data=True):
        feature = edge_features.get((u, v, key))
        length = max(float(data.get("length", 1.0)), 1.0)
        if feature is None:
            data["route_cost"] = length
            continue
        darkness = feature["darkness_index"]
        accident = feature["accident_exposure"]
        cctv_credit = feature["cctv_credit"]
        # 낮은 조명 밀도와 사고 다발지역 노출은 비용을 높이고,
        # CCTV는 작은 보조 가점으로만 반영합니다.
        factor = 1.0 + alpha * darkness + 2.0 * accident - alpha * cctv_credit
        data["route_cost"] = length * max(0.5, factor)
    return weighted


def judge_alternative(metrics_a, metrics_b, distance_a, distance_b):
    """
    초기 운영 기준(추후 실제 데이터 검증으로 조정):
    - 경로 B 우회율 15% 이하
    - 보안등/CCTV 지점 수 감소는 각각 10% 이내
    - 사고 다발지역 100m 이내 노출 비율 악화는 5%p 이내
    - 위 조건 안에서 안전 관련 지표 중 하나 이상 개선되어야 대안 후보
    """
    detour_pct = (
        (distance_b - distance_a) / distance_a * 100
        if distance_a else 0.0
    )

    lights_ok = metrics_b["light_sites"] >= metrics_a["light_sites"] * 0.90
    cctv_ok = metrics_b["cctv_sites"] >= metrics_a["cctv_sites"] * 0.90
    accident_ok = (
        metrics_b["accident_exposed_pct"]
        <= metrics_a["accident_exposed_pct"] + 5.0
    )

    improvements = []
    if metrics_b["light_sites"] > metrics_a["light_sites"]:
        improvements.append("보안등 위치 수 증가")
    if metrics_b["cctv_sites"] > metrics_a["cctv_sites"]:
        improvements.append("CCTV 설치 지점 수 증가")
    if (
        metrics_b["accident_exposed_m"]
        < metrics_a["accident_exposed_m"] - 1.0
    ):
        improvements.append("사고 다발지역 100m 이내 경로 구간 감소")

    conditions = {
        "우회 거리 15% 이내": detour_pct <= 15.0,
        "보안등 감소 폭 허용 범위": lights_ok,
        "CCTV 감소 폭 허용 범위": cctv_ok,
        "사고 다발지역 노출 악화 허용 범위": accident_ok,
        "안전 관련 지표 중 하나 이상 개선": bool(improvements)
    }
    recommended = all(conditions.values())

    if recommended:
        message = "경로 B는 설정한 비교 기준을 충족하는 대안 경로입니다."
        status = "대안 경로 후보"
    else:
        message = (
            "현재 조건에서는 경로 B가 충분한 개선 효과를 보이지 않아 "
            "최단 거리 경로 A를 기본 경로로 안내합니다."
        )
        status = "최단 거리 경로 A 권장"

    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "recommended": recommended,
        "status": status,
        "message": message,
        "detour_pct": detour_pct,
        "improvements": improvements,
        "failed": failed,
        "conditions": conditions
    }


def add_facility_layer(map_object, data, layer_name, color, popup_column):
    """시설물 종류별 색을 유지하고, 클러스터는 반투명 원형으로 표시합니다."""
    group = folium.FeatureGroup(name=layer_name, show=True)

    # 보안등 클러스터는 개수에 따라 초록/노랑 계열로 구분하되,
    # 두 색 모두 반투명하게 하여 첫 번째 화면의 부드러운 느낌을 유지합니다.
    if "보안등" in layer_name:
        cluster_icon = """function(cluster){
            var count = cluster.getChildCount();
            var fill = count >= 10 ? 'rgba(250,204,21,.52)' : 'rgba(132,204,22,.48)';
            var border = count >= 10 ? '#ca8a04' : '#65a30d';
            return new L.DivIcon({
                html: '<div style="background:'+fill+';border:2px solid '+border+';border-radius:50%;width:42px;height:42px;line-height:38px;text-align:center;color:#263238;font-size:14px;font-weight:600;">'+count+'</div>',
                className:'', iconSize:new L.Point(42,42)
            });
        }"""
    else:
        # CCTV는 보안등과 구분되는 청록색 반투명 원으로 표시합니다.
        cluster_icon = """function(cluster){
            var count = cluster.getChildCount();
            return new L.DivIcon({
                html: '<div style="background:rgba(15,157,154,.48);border:2px solid rgba(15,118,110,.75);border-radius:50%;width:40px;height:40px;line-height:36px;text-align:center;color:#123b3a;font-size:14px;font-weight:600;">'+count+'</div>',
                className:'', iconSize:new L.Point(40,40)
            });
        }"""

    cluster = MarkerCluster(
        name=layer_name,
        icon_create_function=cluster_icon,
        options={"maxClusterRadius": 45}
    ).add_to(group)

    for _, row in data.to_crs(4326).iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5 if "보안등" in layer_name else 5.5,
            color=color,
            weight=1.2,
            opacity=0.65,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            popup=f"{layer_name}<br>{popup_column}: {row[popup_column]}"
        ).add_to(cluster)

    group.add_to(map_object)


def create_result_map(start_xy, end_xy, snapped_start, snapped_end,
                      edges_a, edges_b, buffer_a, buffer_b,
                      lights_a, cctvs_a, lights_b, cctvs_b,
                      selected_route="A", selected_gap=None):
    center = [(start_xy[0] + end_xy[0]) / 2, (start_xy[1] + end_xy[1]) / 2]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    folium.GeoJson(edges_a.to_crs(4326).to_json(), name="A · 최단 거리",
        style_function=lambda f: {"color":"#2864d7","weight":5,"opacity":0.95}).add_to(m)
    folium.GeoJson(edges_b.to_crs(4326).to_json(), name="B · 시설물 분포 고려",
        style_function=lambda f: {"color":"#8b45b5","weight":5,"opacity":0.95,"dashArray":"8,6"}).add_to(m)

    # 두 경로가 겹치지 않는 구간을 굵게 덧그려 차이가 드러나도록 합니다.
    try:
        geom_a = unary_union(edges_a.to_crs(5179).geometry)
        geom_b = unary_union(edges_b.to_crs(5179).geometry)
        only_a = geom_a.difference(geom_b.buffer(2))
        only_b = geom_b.difference(geom_a.buffer(2))
        if not only_a.is_empty:
            a_diff = gpd.GeoSeries([only_a], crs="EPSG:5179").to_crs(4326)
            folium.GeoJson(a_diff.to_json(), name="A만 지나는 구간",
                style_function=lambda f: {"color":"#1746a2","weight":8,"opacity":0.95},
                tooltip="A 경로만 지나는 구간").add_to(m)
        if not only_b.is_empty:
            b_diff = gpd.GeoSeries([only_b], crs="EPSG:5179").to_crs(4326)
            folium.GeoJson(b_diff.to_json(), name="B만 지나는 구간",
                style_function=lambda f: {"color":"#a21caf","weight":8,"opacity":0.95},
                tooltip="B 경로만 지나는 구간").add_to(m)
    except Exception as map_diff_error:
        print("경로 차이 구간 표시 생략:", map_diff_error)

    chosen_lights = lights_b if selected_route == "B" else lights_a
    chosen_cctvs = cctvs_b if selected_route == "B" else cctvs_a
    add_facility_layer(m, chosen_lights, "보안등 위치 (선택 경로 기준)", "#eab308", "설치개수")
    add_facility_layer(m, chosen_cctvs, "CCTV 설치 지점 (선택 경로 기준)", "#155e75", "카메라대수")

    if selected_gap and selected_gap[0] is not None:
        gap_start, gap_end = selected_gap
        gap_points = gpd.GeoSeries([gap_start, gap_end], crs="EPSG:5179").to_crs(4326)
        gap_coords = [(gap_points.iloc[0].y, gap_points.iloc[0].x),
                      (gap_points.iloc[1].y, gap_points.iloc[1].x)]
        folium.PolyLine(
            gap_coords, color="#4b5563", weight=9, opacity=0.9,
            tooltip="가장 긴 보안등 공백 구간(근사)"
        ).add_to(m)
        folium.PolyLine(
            gap_coords, color="#f9fafb", weight=3, opacity=1,
            tooltip="가장 긴 보안등 공백 구간(근사)"
        ).add_to(m)

    accident_group = folium.FeatureGroup(name="공식 보행자 사고 다발지역", show=True)
    for _, row in accidents.to_crs(4326).iterrows():
        folium.RegularPolygonMarker(location=[row.geometry.y,row.geometry.x],
            number_of_sides=3, rotation=0, radius=9, color="#c2410c",
            fill=True, fill_color="#f97316", fill_opacity=0.9,
            popup=f"{row['spot_nm']}<br>발생 연도: {row['search_year']}<br>사고 건수: {row['occrrnc_cnt']}").add_to(accident_group)
    accident_group.add_to(m)

    for clicked, snapped, label, color in [
        (start_xy,snapped_start,"출발 선택 위치 → 도로 연결점","#16803c"),
        (end_xy,snapped_end,"목적지 선택 위치 → 도로 연결점","#374151")]:
        folium.PolyLine([clicked,snapped],color=color,weight=2,opacity=.9,
            dash_array="4,6",tooltip=label).add_to(m)

    folium.Marker(snapped_start,tooltip="출발지(도로 연결점)",
        icon=folium.Icon(color="green",icon="play",prefix="fa")).add_to(m)
    folium.Marker(snapped_end,tooltip="목적지(도로 연결점)",
        icon=folium.Icon(color="black",icon="flag-checkered",prefix="fa")).add_to(m)

    legend = """<details open style="position:absolute;right:12px;bottom:22px;z-index:9999;background:white;border:1px solid #d7dce5;border-radius:10px;padding:9px 12px;box-shadow:0 2px 8px rgba(0,0,0,.18);font:13px/1.8 Arial,sans-serif;max-width:245px">
    <summary style="cursor:pointer;font-weight:700">지도 범례</summary>
    <div><span style="color:#2864d7">━</span> A 최단 거리 경로</div>
    <div><span style="color:#8b45b5">┅</span> B 시설물 분포 고려 경로</div>
    <div><span style="color:#1746a2">━</span> A만 지나는 구간</div><div><span style="color:#a21caf">━</span> B만 지나는 구간</div>
    <div style="display:flex;align-items:center;gap:7px">
      <span style="display:inline-flex;gap:3px;align-items:center">
        <i style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(132,204,22,.58);border:1.5px solid #65a30d"></i>
        <i style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgba(250,204,21,.62);border:1.5px solid #ca8a04"></i>
        <i style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#eab308"></i>
      </span>
      보안등 (선택 경로 기준)
    </div>
    <div style="display:flex;align-items:center;gap:7px">
      <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:rgba(15,157,154,.48);border:2px solid rgba(15,118,110,.75)"></span>
      CCTV (선택 경로 기준)
    </div>
    <div><span style="color:#4b5563">━</span> 가장 긴 보안등 공백(근사)</div>
    <div><span style="color:#f97316">▲</span> 사고 다발지역</div>
    <div>🟢 출발지　⚑ 목적지</div></details>"""
    from folium import Element
    m.get_root().html.add_child(Element(legend))
    m.fit_bounds([start_xy,end_xy,snapped_start,snapped_end],padding=(30,30))
    return m.get_root().render()


# =========================================================
# 웹페이지
# 지도에서 출발지/목적지 선택
# =========================================================
PAGE = r"""
<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>옥천군 밤길 경로 탐색</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#202733;font-family:Arial,'Malgun Gothic',sans-serif}
.wrap{max-width:1320px;margin:auto;padding:22px 16px 42px}h1{font-size:26px;margin:0 0 7px}.sub,.hint,.note{color:#667085;line-height:1.6}
.panel{background:#fff;border:1px solid #e3e7ee;border-radius:14px;padding:20px;margin:16px 0}#selectMap{height:520px;border:1px solid #d9dee8;border-radius:11px;z-index:1}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin:13px 0}button{border:0;border-radius:9px;padding:13px 17px;font-size:15px;font-weight:700;cursor:pointer;background:#e9edf5;color:#263247}
button.active,button.submit{background:#2457c5;color:white}button:disabled{opacity:.45;cursor:not-allowed}.status{padding:12px;background:#f4f6fa;border-radius:9px;margin:10px 0}
.selected{display:grid;grid-template-columns:1fr 1fr;gap:10px}.selected div{padding:12px;background:#f7f8fb;border-radius:9px}.selected small{display:block;color:#667085;margin-bottom:5px}
.result-layout{display:grid;grid-template-columns:minmax(300px,.78fr) minmax(0,1.45fr);gap:18px;align-items:start}.summary-card{border:1px solid #e1e6ef;border-top:4px solid #2864d7;border-radius:12px;padding:18px}
.distance{font-size:30px;font-weight:800;margin:5px 0 14px}.time-card{background:#f4f6fa;border-radius:10px;padding:13px 15px;margin-bottom:14px}.time-card small,.metric small{display:block;color:#667085;margin-bottom:4px}.time-card b{font-size:24px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:9px}.metric{background:#f7f8fb;border-radius:9px;padding:11px}.route-reason{margin-top:13px;padding:13px;background:#f0f4ff;border-left:4px solid #4263eb;border-radius:8px;line-height:1.65;font-weight:600}.map-result{width:100%;height:650px;border:1px solid #d9dee8;border-radius:11px}
.compare{margin-top:16px}.compare summary{cursor:pointer;font-weight:700;padding:4px}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{padding:11px;border-bottom:1px solid #e8ebf1;text-align:center}th{background:#f7f8fb}th:first-child,td:first-child{text-align:left}
.error{background:#fff0ef;color:#b42318;padding:13px;border-radius:9px}.note{font-size:13px}.preference{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0}.preference input{flex:1;min-width:150px}.loading{display:none;color:#2457c5;font-weight:700;margin-top:10px}
@media(max-width:850px){.result-layout{grid-template-columns:1fr}.map-result{height:520px}}@media(max-width:600px){.wrap{padding:14px 10px}.panel{padding:15px}#selectMap{height:430px}.map-result{height:470px}.selected{grid-template-columns:1fr}.distance{font-size:27px}h1{font-size:22px}}
</style></head><body><div class="wrap">
<h1>옥천군 밤길 경로 탐색</h1><p class="sub">보안등·CCTV 분포와 공식 보행자 사고 다발지역 정보를 참고해 경로를 비교합니다. 실제 안전을 보장하는 서비스는 아닙니다.</p>
<div class="panel"><h2>① 출발지와 목적지 선택</h2><p class="hint">지도를 클릭해 출발지와 목적지를 선택하세요. 도로에서 떨어진 곳을 선택하면 가장 가까운 보행 도로에 연결합니다.</p>
<div class="tools"><button type="button" id="startMode" class="active">출발지 선택</button><button type="button" id="endMode">목적지 선택</button><button type="button" id="swap">출발·도착 바꾸기</button><button type="button" id="reset">초기화</button></div>
<div id="selectMap"></div><div class="status" id="status">출발지를 선택해 주세요.</div>
<div class="selected"><div><small>출발지</small><strong id="startText">선택되지 않음</strong></div><div><small>목적지</small><strong id="endText">선택되지 않음</strong></div></div>
<div class="preference"><label for="preference"><b>거리 우선 ↔ 시설물 고려</b></label><input id="preference" type="range" min="0" max="100" value="55"><span id="preferenceValue">55%</span><small class="note">0%는 거리 우선, 100%는 시설물 분포 고려 비중이 가장 큽니다. 값을 바꾼 뒤 경로 탐색하기를 다시 누르세요.</small></div>
<form method="post" id="routeForm" style="margin-top:14px"><input type="hidden" name="start_lat" id="start_lat"><input type="hidden" name="start_lon" id="start_lon"><input type="hidden" name="end_lat" id="end_lat"><input type="hidden" name="end_lon" id="end_lon"><input type="hidden" name="preference" id="preferenceInput" value="55">
<button type="submit" class="submit" id="searchButton" disabled>경로 탐색하기</button><div class="loading" id="loading">경로를 분석하고 있습니다. 잠시만 기다려 주세요.</div></form>
{% if error %}<div class="error" style="margin-top:12px">{{error}}</div>{% endif %}</div>
{% if result %}<div class="panel" id="results"><h2>② 경로 분석 결과</h2><div class="result-layout">
<div class="summary-card"><div>{{result.route_name}}</div><div class="distance">약 {{result.route_distance_round}}m</div>
<div class="time-card"><small>평균 예상 도보 시간</small><b>약 {{result.route_time_round}}</b><small>평균 보행 속도 시속 4km 기준</small></div>
<p>{{result.route_message}}</p><div class="metrics">
<div class="metric"><small>보안등 위치</small><b>{{result.route_lights}}곳</b><small>{{result.route_light_density}}곳 / 100m</small></div>
<div class="metric"><small>CCTV 설치 지점</small><b>{{result.route_cctvs}}곳</b><small>{{result.route_cctv_total}}대</small></div>
<div class="metric"><small>가장 긴 보안등 공백(근사)</small><b>{{result.route_max_gap}}m</b></div>
<div class="metric"><small>사고 다발지역 노출</small><b>{{result.route_accident_pct}}%</b><small>{{result.accident_warning}}</small></div>
<div class="metric"><small>가장 가까운 사고 다발지역</small><b>{{result.route_nearest_accident}}</b><small>경로 중심선 기준 직선거리</small></div></div>
<div class="route-reason">{{result.route_reason}}</div></div>
<div><iframe class="map-result" title="경로 비교 지도" srcdoc="{{map_html|e}}"></iframe></div></div>
<details class="compare"><summary>두 경로의 상세 비교 보기</summary><table><tr><th>비교 항목</th><th style="color:#2864d7">A · 최단 거리</th><th style="color:#8b45b5">B · 시설물 고려</th></tr>
<tr><td>이동 거리</td><td>{{result.da_round}}m</td><td>{{result.db_round}}m</td></tr><tr><td>예상 도보 시간</td><td>{{result.ta_round}}</td><td>{{result.tb_round}}</td></tr>
<tr><td>보안등 위치</td><td>{{result.la}}곳</td><td>{{result.lb}}곳</td></tr><tr><td>보안등 밀도</td><td>{{result.light_density_a}}곳/100m</td><td>{{result.light_density_b}}곳/100m</td></tr>
<tr><td>최대 보안등 공백(근사)</td><td>{{result.gap_a}}m</td><td>{{result.gap_b}}m</td></tr><tr><td>CCTV 설치 지점 / 카메라</td><td>{{result.ca}}곳 / {{result.cat}}대</td><td>{{result.cb}}곳 / {{result.cbt}}대</td></tr>
<tr><td>사고 다발지역 100m 이내 경로</td><td>{{result.acc_ap}}%</td><td>{{result.acc_bp}}%</td></tr>
<tr><td>가장 가까운 사고 다발지역</td><td>{{result.near_a}}m</td><td>{{result.near_b}}m</td></tr>
<tr><td>우회율</td><td>—</td><td>{{result.pct}}%</td></tr></table>
<p class="note">지도에 표시되는 보안등과 CCTV는 선택된 경로 기준입니다. 두 경로를 동시에 비교하는 표의 시설물 수와 구분해 확인하세요.</p></details>
<p class="note">보안등 밀도는 경로 중심선 양쪽 30m 안에 포함된 설치 지점 수 ÷ 실제 경로 길이 × 100으로 계산합니다. 최대 공백은 해당 보안등 위치를 경로에 투영해 계산한 근사치입니다. 사고 다발지역 노출은 해당 구역과의 공간적 중첩이며 사고 위험을 직접 측정하지 않습니다.</p></div>{% endif %}
<div class="panel note"><b>데이터와 분석 기준</b><br>보안등: 충청북도 옥천군 보안등정보 파일 / CCTV: 옥천군 CCTV정보 파일 / 사고: 옥천군 보행자 교통사고 다발지역(2021~2024) 파일. 시설물은 경로 중심선 양쪽 30m, 사고 다발지역 노출은 공식 사고 다발지역 100m 이내 경로 길이로 계산합니다. 원본 파일에 별도 기준일이 확인되지 않아 기준일은 원본 파일 제공·갱신일 확인이 필요합니다. CCTV는 조명 대체 지표가 아니며, 실제 야간 밝기나 보행 안전을 보장하지 않습니다.</div>
</div>
<script>
const map=L.map('selectMap').setView([36.306,127.57],14);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(map);
let mode='start',startMarker=null,endMarker=null,startPoint=null,endPoint=null;const $=id=>document.getElementById(id),searchButton=$('searchButton'),statusBox=$('status');
function setMode(m){mode=m;$('startMode').classList.toggle('active',m==='start');$('endMode').classList.toggle('active',m==='end');statusBox.textContent=m==='start'?'출발지를 지도에서 선택하세요.':'목적지를 지도에서 선택하세요.'}
$('startMode').addEventListener('click',()=>setMode('start'));$('endMode').addEventListener('click',()=>setMode('end'));
function clearOldResults(){const r=$('results');if(r)r.remove()}
function updateSubmit(){searchButton.disabled=!(startPoint&&endPoint)}
map.on('click',e=>{const p={lat:e.latlng.lat,lon:e.latlng.lng};if(mode==='start'){startPoint=p;if(startMarker)map.removeLayer(startMarker);startMarker=L.marker([p.lat,p.lon]).addTo(map);$('start_lat').value=p.lat;$('start_lon').value=p.lon;$('startText').textContent=p.lat.toFixed(5)+', '+p.lon.toFixed(5);statusBox.textContent='출발지를 선택했습니다. 목적지를 선택하세요.'}else{endPoint=p;if(endMarker)map.removeLayer(endMarker);endMarker=L.marker([p.lat,p.lon]).addTo(map);$('end_lat').value=p.lat;$('end_lon').value=p.lon;$('endText').textContent=p.lat.toFixed(5)+', '+p.lon.toFixed(5);statusBox.textContent='두 위치가 선택되었습니다.'}clearOldResults();updateSubmit()});
$('swap').addEventListener('click',()=>{if(!startPoint||!endPoint)return;[startPoint,endPoint]=[endPoint,startPoint];$('start_lat').value=startPoint.lat;$('start_lon').value=startPoint.lon;$('end_lat').value=endPoint.lat;$('end_lon').value=endPoint.lon;$('startText').textContent=startPoint.lat.toFixed(5)+', '+startPoint.lon.toFixed(5);$('endText').textContent=endPoint.lat.toFixed(5)+', '+endPoint.lon.toFixed(5);if(startMarker)map.removeLayer(startMarker);if(endMarker)map.removeLayer(endMarker);startMarker=L.marker([startPoint.lat,startPoint.lon]).addTo(map);endMarker=L.marker([endPoint.lat,endPoint.lon]).addTo(map);clearOldResults()});
$('reset').addEventListener('click',()=>{if(startMarker)map.removeLayer(startMarker);if(endMarker)map.removeLayer(endMarker);startMarker=endMarker=null;startPoint=endPoint=null;['start_lat','start_lon','end_lat','end_lon'].forEach(id=>$(id).value='');$('startText').textContent='선택되지 않음';$('endText').textContent='선택되지 않음';clearOldResults();setMode('start');updateSubmit()});
if($('start_lat').value && $('start_lon').value){startPoint={lat:parseFloat($('start_lat').value),lon:parseFloat($('start_lon').value)};startMarker=L.marker([startPoint.lat,startPoint.lon]).addTo(map);$('startText').textContent=startPoint.lat.toFixed(5)+', '+startPoint.lon.toFixed(5)}
if($('end_lat').value && $('end_lon').value){endPoint={lat:parseFloat($('end_lat').value),lon:parseFloat($('end_lon').value)};endMarker=L.marker([endPoint.lat,endPoint.lon]).addTo(map);$('endText').textContent=endPoint.lat.toFixed(5)+', '+endPoint.lon.toFixed(5)}
updateSubmit();
$('preference').addEventListener('input',e=>{$('preferenceValue').textContent=e.target.value+'%';$('preferenceInput').value=e.target.value});
$('routeForm').addEventListener('submit',()=>{$('loading').style.display='block';searchButton.disabled=true;searchButton.textContent='분석 중...'});
</script></body></html>



"""
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    map_html = None
    error = None
    selected_start = None
    selected_end = None

    if request.method == "POST":
        try:
            start_lat = float(request.form["start_lat"])
            start_lon = float(request.form["start_lon"])
            end_lat = float(request.form["end_lat"])
            end_lon = float(request.form["end_lon"])
            selected_start = [start_lat, start_lon]
            selected_end = [end_lat, end_lon]

            # 클릭 좌표를 가장 가까운 보행 도로망 노드에 연결
            start_node = ox.distance.nearest_nodes(
                G, X=start_lon, Y=start_lat
            )
            end_node = ox.distance.nearest_nodes(
                G, X=end_lon, Y=end_lat
            )
            from math import hypot
            start_snap_m = hypot((float(G.nodes[start_node]["x"])-start_lon)*90000,
                                 (float(G.nodes[start_node]["y"])-start_lat)*111000)
            end_snap_m = hypot((float(G.nodes[end_node]["x"])-end_lon)*90000,
                               (float(G.nodes[end_node]["y"])-end_lat)*111000)
            if start_snap_m > 500 or end_snap_m > 500:
                raise ValueError("옥천군 보행 도로망에서 너무 먼 위치입니다. 옥천읍 주변 도로를 선택해 주세요.")
            # 출발·목적지 클릭 좌표가 50m 이내면 경로 탐색 전에 안내합니다.
            from math import hypot
            click_distance_m = hypot((end_lon - start_lon) * 90000,
                                     (end_lat - start_lat) * 111000)
            if click_distance_m < 50 or start_node == end_node:
                raise ValueError(
                    "출발지와 목적지가 너무 가깝습니다(약 50m 이내). 지도에서 조금 더 떨어진 지점을 선택해 주세요."
                )

            route_a = ox.routing.shortest_path(
                G, start_node, end_node, weight="length"
            )
            preference = float(request.form.get("preference", 55))
            G_weighted = build_weighted_graph(preference)
            route_b = ox.routing.shortest_path(
                G_weighted, start_node, end_node, weight="route_cost"
            )

            if route_a is None or route_b is None:
                raise ValueError(
                    "선택한 위치 사이의 보행 경로를 찾지 못했습니다. 다른 위치를 선택해 주세요."
                )

            edges_a = ox.routing.route_to_gdf(G, route_a)
            edges_b = ox.routing.route_to_gdf(G, route_b)

            distance_a = float(edges_a["length"].sum())
            distance_b = float(edges_b["length"].sum())

            line_a = unary_union(edges_a.to_crs(5179).geometry)
            line_b = unary_union(edges_b.to_crs(5179).geometry)
            snapped_start = [float(G.nodes[start_node]["y"]), float(G.nodes[start_node]["x"])]
            snapped_end = [float(G.nodes[end_node]["y"]), float(G.nodes[end_node]["x"])]
            lighting_a = analyze_route_lighting(line_a)
            lighting_b = analyze_route_lighting(line_b)
            buffer_a = line_a.buffer(BUFFER_DISTANCE)
            buffer_b = line_b.buffer(BUFFER_DISTANCE)

            lights_a, cctvs_a = analyze_facilities(buffer_a)
            lights_b, cctvs_b = analyze_facilities(buffer_b)

            la, lb = lighting_a["light_sites"], lighting_b["light_sites"]
            lat = int(lights_a["설치개수"].sum())
            lbt = int(lights_b["설치개수"].sum())
            ca, cb = len(cctvs_a), len(cctvs_b)
            cat = int(cctvs_a["카메라대수"].sum())
            cbt = int(cctvs_b["카메라대수"].sum())

            # 시설물 집계와 사고 노출 분석 모두 동일한 경로 A/B를 평가
            safety_a = analyze_route_safety(line_a)
            safety_b = analyze_route_safety(line_b)
            metrics_a = {
                "light_sites": la,
                "cctv_sites": ca,
                **safety_a
            }
            metrics_b = {
                "light_sites": lb,
                "cctv_sites": cb,
                **safety_b
            }
            judgement = judge_alternative(
                metrics_a, metrics_b, distance_a, distance_b
            )

            near_a = (
                f"{safety_a['nearest_accident_m']:.1f}"
                if safety_a["nearest_accident_m"] is not None else "자료 없음"
            )
            near_b = (
                f"{safety_b['nearest_accident_m']:.1f}"
                if safety_b["nearest_accident_m"] is not None else "자료 없음"
            )

            diff = distance_b - distance_a
            pct = (diff / distance_a * 100) if distance_a else 0

            same_route = route_a == route_b
            selected_is_b = (
                not same_route and preference > 0 and
                distance_b <= distance_a * 1.30 and (
                    lighting_b["light_density_100m"] > lighting_a["light_density_100m"] + 0.02
                    or lighting_b["max_light_gap_m"] < lighting_a["max_light_gap_m"] - 5
                    or safety_b["accident_exposed_pct"] < safety_a["accident_exposed_pct"] - 0.5
                )
            )
            selected_distance = distance_b if selected_is_b else distance_a
            selected_lights = lb if selected_is_b else la
            selected_light_total = lbt if selected_is_b else lat
            selected_cctvs = cb if selected_is_b else ca
            selected_cctv_total = cbt if selected_is_b else cat
            selected_safety = safety_b if selected_is_b else safety_a
            selected_lighting = lighting_b if selected_is_b else lighting_a
            selected_nearest = safety_b["nearest_accident_m"] if selected_is_b else safety_a["nearest_accident_m"]
            nearest_text = "자료 없음" if selected_nearest is None else f"약 {selected_nearest:.0f}m"
            light_change = lb - la
            gap_change = lighting_b["max_light_gap_m"] - lighting_a["max_light_gap_m"]
            if same_route:
                route_reason = "현재 설정에서는 A와 B 경로가 같습니다. 출발·도착 위치나 시설물 고려 비율을 바꿔 다시 확인해 보세요."
            elif selected_is_b:
                route_reason = (f"최단 경로보다 {max(0, int(round(diff))):,}m 더 걷지만, 보안등 위치가 {light_change:+d}곳이고 "
                                f"최대 보안등 공백은 {gap_change:+.0f}m 변화합니다. 시설물 분포 지표를 바탕으로 한 비교 결과이며 실제 안전을 의미하지 않습니다.")
            else:
                route_reason = "시설물 고려 경로가 설정한 우회 허용 범위 또는 개선 기준을 충족하지 않아 최단 경로를 표시합니다."
            result = {
                "route_name": "시설물 분포 및 사고 다발지역 고려 경로" if selected_is_b else "최단 거리 경로",
                "route_color": "#7a35a8" if selected_is_b else "#2864d7",
                "route_distance": f"{selected_distance:,.2f}",
                "route_distance_round": f"{int(round(selected_distance)):,}",
                "route_time_round": f"{int(round(selected_distance / 4000 * 60))}분",
                "route_light_density": f"{selected_lighting['light_density_100m']:.2f}",
                "route_max_gap": f"{selected_lighting['max_light_gap_m']:.0f}",
                "route_nearest_accident": nearest_text,
                "route_reason": route_reason,
                "accident_warning": "노출 구간 있음" if selected_safety["accident_exposed_pct"] > 0 else "노출 구간 확인되지 않음",
                "route_time": (
                    f"{int((selected_distance / 4000) * 60)}분"
                    + (
                        f" {int(round(((selected_distance / 4000) * 60 % 1) * 60))}초"
                        if round((selected_distance / 4000) * 60 % 1 * 60) > 0
                        else ""
                    )
                ),
                "route_lights": selected_lights,
                "route_light_total": selected_light_total,
                "route_cctvs": selected_cctvs,
                "route_cctv_total": selected_cctv_total,
                "route_accident_pct": f"{selected_safety['accident_exposed_pct']:.1f}",
                "route_message": ("시설물 분포와 사고 다발지역 노출 지표를 참고해 경로를 비교했습니다." if selected_is_b else ("A와 B가 같은 경로입니다." if same_route else "현재 설정에서는 최단 거리 경로를 표시합니다.")),
                "da": f"{distance_a:,.2f}",
                "db": f"{distance_b:,.2f}",
                "dan": round(distance_a, 2),
                "dbn": round(distance_b, 2),
                "la": la, "lb": lb,
                "lat": lat, "lbt": lbt,
                "ca": ca, "cb": cb,
                "cat": cat, "cbt": cbt,
                "acc_a": f"{safety_a['accident_exposed_m']:,.1f}",
                "acc_b": f"{safety_b['accident_exposed_m']:,.1f}",
                "acc_ap": f"{safety_a['accident_exposed_pct']:.1f}",
                "acc_bp": f"{safety_b['accident_exposed_pct']:.1f}",
                "near_a": near_a,
                "near_b": near_b,
                "da_round": f"{int(round(distance_a)):,}",
                "db_round": f"{int(round(distance_b)):,}",
                "ta_round": f"약 {int(round(distance_a / 4000 * 60))}분",
                "tb_round": f"약 {int(round(distance_b / 4000 * 60))}분",
                "light_density_a": f"{lighting_a['light_density_100m']:.2f}",
                "light_density_b": f"{lighting_b['light_density_100m']:.2f}",
                "gap_a": f"{lighting_a['max_light_gap_m']:.0f}",
                "gap_b": f"{lighting_b['max_light_gap_m']:.0f}",
                "judgement": judgement,
                "diff": f"{diff:,.2f}",
                "pct": f"{pct:.1f}"
            }

            map_html = create_result_map(
                [start_lat, start_lon], [end_lat, end_lon],
                snapped_start, snapped_end,
                edges_a, edges_b, buffer_a, buffer_b,
                lights_a, cctvs_a, lights_b, cctvs_b,
                selected_route="B" if selected_is_b else "A",
                selected_gap=selected_lighting["max_gap_segment"]
            )

        except Exception as exc:
            error = str(exc)
            print("경로 분석 오류:", repr(exc))

    return render_template_string(
        PAGE,
        center_lat=MAP_CENTER[0],
        center_lon=MAP_CENTER[1],
        result=result,
        map_html=map_html,
        error=error,
        selected_start=selected_start,
        selected_end=selected_end
    )


# =========================================================
# 자동 검증 기능
# =========================================================
def run_validation():
    """도로망의 대표 지점 쌍으로 경로 생성과 분석 함수의 기본 동작을 점검합니다."""
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    # 1. 입력 데이터 기본 상태
    record("보행 도로망 노드 존재", len(G.nodes) > 0, f"노드 {len(G.nodes):,}개")
    record("보행 도로망 도로 구간 존재", len(G.edges) > 0, f"구간 {len(G.edges):,}개")
    record("보안등 좌표 데이터 존재", len(lights) > 0, f"지점 {len(lights):,}곳")
    record("CCTV 좌표 데이터 존재", len(cctvs) > 0, f"지점 {len(cctvs):,}곳")
    record("사고 다발지역 좌표 데이터 존재", len(accidents) > 0, f"지점 {len(accidents):,}곳")

    # 2. 좌표 및 그래프 가중치 점검
    node_values = list(G.nodes(data=True))
    coords_ok = all(
        pd.notna(d.get("x")) and pd.notna(d.get("y"))
        and -180 <= float(d["x"]) <= 180
        and -90 <= float(d["y"]) <= 90
        for _, d in node_values
    )
    record("도로망 노드 좌표 유효성", coords_ok, "모든 노드의 위도·경도 범위 확인")

    safe_weights = [
        float(data.get("safe_weight", -1))
        for _, _, data in G_safe.edges(data=True)
    ]
    weights_ok = bool(safe_weights) and all(
        pd.notna(w) and w > 0 and w < float("inf") for w in safe_weights
    )
    record("안전 고려 가중치 유효성", weights_ok,
           f"가중치 {len(safe_weights):,}개 중 양수·유한 값 여부 확인")

    # 3. 가장 큰 연결 요소에서 공간적으로 떨어진 대표 노드 쌍 선택
    try:
        if G.is_directed():
            components = list(ox.utils_graph.get_largest_component(G, strongly=False).nodes)
            test_graph = G.subgraph(components).copy()
        else:
            components = list(max(nx.connected_components(G), key=len))
            test_graph = G.subgraph(components).copy()
    except Exception:
        # OSMnx 버전 차이를 고려한 대체 처리
        import networkx as nx
        if G.is_directed():
            component_nodes = max(nx.weakly_connected_components(G), key=len)
        else:
            component_nodes = max(nx.connected_components(G), key=len)
        test_graph = G.subgraph(component_nodes).copy()

    import networkx as nx
    nodes = list(test_graph.nodes(data=True))
    if len(nodes) >= 2:
        candidates = [
            min(nodes, key=lambda item: float(item[1]["x"])),
            max(nodes, key=lambda item: float(item[1]["x"])),
            min(nodes, key=lambda item: float(item[1]["y"])),
            max(nodes, key=lambda item: float(item[1]["y"]))
        ]
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if candidates[i][0] != candidates[j][0]:
                    pair = (candidates[i][0], candidates[j][0])
                    if pair not in pairs and (pair[1], pair[0]) not in pairs:
                        pairs.append(pair)
        # 최대 3개 조합을 검사
        for number, (start_node, end_node) in enumerate(pairs[:3], start=1):
            try:
                route_a = ox.routing.shortest_path(
                    G, start_node, end_node, weight="length"
                )
                route_b = ox.routing.shortest_path(
                    G_safe, start_node, end_node, weight="safe_weight"
                )
                paths_exist = route_a is not None and route_b is not None
                if not paths_exist:
                    record(f"대표 경로 {number} 생성", False, "경로 A 또는 B를 생성하지 못함")
                    continue

                edges_a = ox.routing.route_to_gdf(G, route_a)
                edges_b = ox.routing.route_to_gdf(G, route_b)
                distance_a = float(edges_a["length"].sum())
                distance_b = float(edges_b["length"].sum())
                distance_ok = (
                    pd.notna(distance_a) and pd.notna(distance_b)
                    and distance_a > 0 and distance_b > 0
                    and distance_a < float("inf") and distance_b < float("inf")
                )
                record(f"대표 경로 {number} 생성 및 거리", distance_ok,
                       f"A {distance_a:,.1f}m / B {distance_b:,.1f}m")

                line_a = unary_union(edges_a.to_crs(5179).geometry)
                line_b = unary_union(edges_b.to_crs(5179).geometry)
                valid_lines = (
                    line_a is not None and line_b is not None
                    and not line_a.is_empty and not line_b.is_empty
                    and line_a.is_valid and line_b.is_valid
                )
                record(f"대표 경로 {number} 선형 유효성", valid_lines,
                       "경로 선형의 빈 geometry 및 유효성 확인")

                area_a = line_a.buffer(BUFFER_DISTANCE)
                area_b = line_b.buffer(BUFFER_DISTANCE)
                la, ca = analyze_facilities(area_a)
                lb, cb = analyze_facilities(area_b)
                counts_ok = all(v >= 0 for v in [len(la), len(ca), len(lb), len(cb)])
                record(f"대표 경로 {number} 시설물 집계", counts_ok,
                       f"A 보안등 {len(la)}곳·CCTV {len(ca)}곳 / B 보안등 {len(lb)}곳·CCTV {len(cb)}곳")

                safety_a = analyze_route_safety(line_a)
                safety_b = analyze_route_safety(line_b)
                safety_ok = all(
                    pd.notna(m[k]) and m[k] >= 0
                    for m in [safety_a, safety_b]
                    for k in ["accident_exposed_m", "accident_exposed_pct"]
                )
                record(f"대표 경로 {number} 사고 노출 분석", safety_ok,
                       f"A {safety_a['accident_exposed_pct']:.1f}% / B {safety_b['accident_exposed_pct']:.1f}%")

                ma = {"light_sites": len(la), "cctv_sites": len(ca), **safety_a}
                mb = {"light_sites": len(lb), "cctv_sites": len(cb), **safety_b}
                judgment = judge_alternative(ma, mb, distance_a, distance_b)
                judgment_ok = (
                    isinstance(judgment.get("recommended"), bool)
                    and isinstance(judgment.get("conditions"), dict)
                    and judgment["recommended"] == all(judgment["conditions"].values())
                )
                record(f"대표 경로 {number} 추천 판정", judgment_ok,
                       judgment.get("status", "판정 결과 없음"))
            except Exception as exc:
                record(f"대표 경로 {number} 분석", False,
                       f"{type(exc).__name__}: {exc}")
    else:
        record("대표 경로 테스트 준비", False, "연결된 도로망 노드가 부족합니다.")

    # 4. 추천 로직의 통제된 입력 테스트
    sample_a = {
        "light_sites": 10, "cctv_sites": 5,
        "accident_exposed_m": 100, "accident_exposed_pct": 10,
    }
    sample_b = {
        "light_sites": 12, "cctv_sites": 5,
        "accident_exposed_m": 80, "accident_exposed_pct": 8,
    }
    sample_judgment = judge_alternative(sample_a, sample_b, 1000, 1100)
    record("추천 판정 기준 기본 테스트",
           sample_judgment["recommended"] is True,
           "개선 지표가 있고 우회율·감소 허용 기준을 충족하는 입력을 확인")

    return checks


VALIDATION_PAGE = r"""
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NightSafeRoute 분석 검증</title>
<style>
body{margin:0;background:#f4f6f9;color:#202733;font-family:Arial,'Malgun Gothic',sans-serif}
.wrap{max-width:950px;margin:auto;padding:25px 16px 45px}
.panel{background:white;border:1px solid #e3e7ee;border-radius:13px;padding:20px;margin:16px 0}
.summary{font-size:18px;font-weight:bold;margin:12px 0}
table{width:100%;border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #e8ebf1;text-align:left}
th{background:#f7f8fb}.pass{color:#16713a;font-weight:bold}.fail{color:#b42318;font-weight:bold}
a{color:#2457c5;font-weight:bold;text-decoration:none}.hint{color:#667085;line-height:1.6;font-size:14px}
button{background:#2457c5;color:white;border:0;border-radius:8px;padding:11px 16px;font-size:15px;font-weight:bold;cursor:pointer}
</style></head><body><div class="wrap">
<h1>NightSafeRoute 분석 기능 검증</h1>
<p><a href="/">← 경로 탐색으로 돌아가기</a></p>
<div class="panel"><h2>검증 결과</h2>
<p class="summary">전체 {{total}}개 항목 중 <span class="pass">통과 {{passed}}개</span> · <span class="fail">실패 {{failed}}개</span></p>
<p class="hint">현재 저장된 도로망과 시설물 데이터를 이용해 대표 노드 간 경로를 자동 생성하고 기본 분석 기능을 점검한 결과입니다. 실제 보행 안전성을 검증하거나 추천 기준의 타당성을 입증하는 테스트는 아닙니다.</p>
<table><tr><th>검증 항목</th><th>결과</th><th>상세 내용</th></tr>
{% for item in checks %}<tr><td>{{item.name}}</td><td class="{{'pass' if item.passed else 'fail'}}">{{'통과' if item.passed else '실패'}}</td><td>{{item.detail}}</td></tr>{% endfor %}
</table><p style="margin-top:18px"><button onclick="location.reload()">다시 검증하기</button></p>
</div></div></body></html>
"""

@app.route("/validate", methods=["GET"])
def validate():
    checks = run_validation()
    passed = sum(1 for item in checks if item["passed"])
    return render_template_string(
        VALIDATION_PAGE, checks=checks, total=len(checks),
        passed=passed, failed=len(checks) - passed
    )


if __name__ == "__main__":
    print("브라우저에서 http://127.0.0.1:5000 으로 접속하세요.")
    app.run(host="127.0.0.1", port=5000, debug=False)
