import json
recs = [json.loads(l) for l in open('processed/manifest.jsonl', encoding='utf-8')]

targets = [
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0012_0245.93_0265.16",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0017_0350.19_0370.36",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0018_0370.36_0389.22",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0020_0410.06_0431.51",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0021_0431.51_0451.04",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0024_0491.06_0510.44",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0025_0510.44_0530.18",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0036_0737.03_0757.75",
    "All_You_Need_To_Know_About_Exports_In_Pakistan__Episode_08_0214_4311.01_4311.25",
    "WhatsApp_Audio_2026-07-14_at_8.52.34_PM_0002_0031.35_0031.49",
]

for r in recs:
    if r.get('segment_id') in targets:
        print(r.get('segment_id'))
        print('  duration:', r.get('duration_seconds'), '| too_short:', r.get('too_short'))
        print('  transcript:', (r.get('transcript') or '')[:80])
        print()