import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import timedelta
import copy
import json


WARNING_LEVELS = {
    1: {'name': '蓝色', 'color': '#1E90FF', 'bg_color': '#E6F3FF'},
    2: {'name': '黄色', 'color': '#FFD700', 'bg_color': '#FFF9E6'},
    3: {'name': '橙色', 'color': '#FF8C00', 'bg_color': '#FFF0E6'},
    4: {'name': '红色', 'color': '#DC143C', 'bg_color': '#FFE6E6'}
}

OPERATORS = ['>', '>=', '<', '<=', '==', '!=']

PARAM_NAMES_CN = {
    'wind_speed': '风速(m/s)',
    'wind_dir': '风向(°)',
    'air_temp': '气温(°C)',
    'pressure': '气压(hPa)',
    'Hs': '有效波高(m)',
    'Tz': '平均波周期(s)',
    'wave_dir': '波向(°)',
    'SST': '海表温度(°C)',
    'salinity': '盐度(psu)',
    'current_speed': '流速(m/s)',
    'current_dir': '流向(°)'
}

AVAILABLE_PARAMS = list(PARAM_NAMES_CN.keys())


@dataclass
class Condition:
    param: str
    operator: str
    threshold: float

    def to_dict(self) -> Dict:
        return {
            'param': self.param,
            'operator': self.operator,
            'threshold': float(self.threshold)
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Condition':
        return cls(
            param=d['param'],
            operator=d['operator'],
            threshold=float(d['threshold'])
        )

    def describe(self) -> str:
        cn_name = PARAM_NAMES_CN.get(self.param, self.param)
        return f"{cn_name} {self.operator} {self.threshold}"

    def evaluate(self, value: float) -> bool:
        if pd.isna(value):
            return False
        try:
            v = float(value)
            t = float(self.threshold)
            if self.operator == '>':
                return v > t
            elif self.operator == '>=':
                return v >= t
            elif self.operator == '<':
                return v < t
            elif self.operator == '<=':
                return v <= t
            elif self.operator == '==':
                return abs(v - t) < 1e-9
            elif self.operator == '!=':
                return abs(v - t) >= 1e-9
        except (ValueError, TypeError):
            return False
        return False


@dataclass
class WarningRule:
    name: str
    level: int
    conditions: List[Condition] = field(default_factory=list)
    duration_minutes: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'level': int(self.level),
            'conditions': [c.to_dict() for c in self.conditions],
            'duration_minutes': int(self.duration_minutes),
            'enabled': bool(self.enabled)
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'WarningRule':
        return cls(
            name=d['name'],
            level=int(d['level']),
            conditions=[Condition.from_dict(c) for c in d.get('conditions', [])],
            duration_minutes=int(d.get('duration_minutes', 0)),
            enabled=bool(d.get('enabled', True))
        )

    def describe_conditions(self) -> str:
        parts = [c.describe() for c in self.conditions]
        desc = " 且 ".join(parts)
        if self.duration_minutes > 0:
            desc += f" 持续{self.duration_minutes}分钟"
        return desc

    def evaluate_row(self, row: pd.Series) -> bool:
        for cond in self.conditions:
            if cond.param not in row.index:
                return False
            if not cond.evaluate(row[cond.param]):
                return False
        return True


@dataclass
class WarningEvent:
    event_id: int
    rule_name: str
    level: int
    buoy_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: int
    param_snapshot: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'rule_name': self.rule_name,
            'level': int(self.level),
            'buoy_id': self.buoy_id,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': int(self.duration_minutes),
            'param_snapshot': {k: (float(v) if pd.notna(v) else None) for k, v in self.param_snapshot.items()}
        }


def get_builtin_templates() -> List[WarningRule]:
    templates = [
        WarningRule(
            name="台风预警",
            level=4,
            conditions=[
                Condition(param='wind_speed', operator='>', threshold=25.0),
                Condition(param='pressure', operator='<', threshold=990.0)
            ],
            duration_minutes=60,
            enabled=True
        ),
        WarningRule(
            name="大浪预警",
            level=3,
            conditions=[
                Condition(param='Hs', operator='>', threshold=4.0)
            ],
            duration_minutes=30,
            enabled=True
        ),
        WarningRule(
            name="低温预警",
            level=2,
            conditions=[
                Condition(param='SST', operator='<', threshold=5.0)
            ],
            duration_minutes=120,
            enabled=True
        )
    ]
    return templates


def make_unique_name(name: str, existing_names: List[str]) -> str:
    if name not in existing_names:
        return name
    counter = 1
    while f"{name}_{counter}" in existing_names:
        counter += 1
    return f"{name}_{counter}"


def import_templates(existing_rules: List[WarningRule]) -> List[WarningRule]:
    existing_names = [r.name for r in existing_rules]
    new_rules = []
    for tpl in get_builtin_templates():
        unique_name = make_unique_name(tpl.name, existing_names)
        existing_names.append(unique_name)
        tpl_copy = copy.deepcopy(tpl)
        tpl_copy.name = unique_name
        new_rules.append(tpl_copy)
    return existing_rules + new_rules


def scan_warnings(
    data: pd.DataFrame,
    rules: List[WarningRule],
    qc_codes: Optional[pd.DataFrame] = None
) -> List[WarningEvent]:
    if data is None or len(data) == 0:
        return []

    enabled_rules = [r for r in rules if r.enabled and len(r.conditions) > 0]
    if not enabled_rules:
        return []

    events = []
    event_id_counter = 1

    required_cols = set(['time', 'buoy_id'])
    for r in enabled_rules:
        for c in r.conditions:
            required_cols.add(c.param)
    missing = required_cols - set(data.columns)
    if missing:
        data = data.copy()
        for col in missing:
            data[col] = np.nan

    buoy_ids = data['buoy_id'].unique()

    for buoy_id in buoy_ids:
        buoy_data = data[data['buoy_id'] == buoy_id].sort_values('time').reset_index(drop=True)
        if len(buoy_data) == 0:
            continue

        for rule in enabled_rules:
            dur_min = rule.duration_minutes

            if dur_min <= 0:
                for idx, row in buoy_data.iterrows():
                    if rule.evaluate_row(row):
                        snapshot = {c.param: row.get(c.param, np.nan) for c in rule.conditions}
                        t = row['time']
                        events.append(WarningEvent(
                            event_id=event_id_counter,
                            rule_name=rule.name,
                            level=rule.level,
                            buoy_id=buoy_id,
                            start_time=t,
                            end_time=t,
                            duration_minutes=0,
                            param_snapshot=snapshot
                        ))
                        event_id_counter += 1
            else:
                match_start_idx = None
                match_count = 0
                expected_interval = None

                if len(buoy_data) >= 2:
                    diffs = buoy_data['time'].diff().dropna()
                    if len(diffs) > 0:
                        median_diff = diffs.median()
                        expected_interval = max(int(round(median_diff.total_seconds() / 60)), 1)
                if expected_interval is None:
                    expected_interval = 1

                required_samples = max(int(round(dur_min / expected_interval)), 1)

                for idx in range(len(buoy_data)):
                    row = buoy_data.iloc[idx]
                    satisfies = rule.evaluate_row(row)

                    if satisfies:
                        if match_start_idx is None:
                            match_start_idx = idx
                            match_count = 1
                        else:
                            prev_t = buoy_data.iloc[idx - 1]['time']
                            curr_t = row['time']
                            gap = (curr_t - prev_t).total_seconds() / 60
                            if gap <= expected_interval * 3:
                                match_count += 1
                            else:
                                match_start_idx = idx
                                match_count = 1
                    else:
                        if match_start_idx is not None and match_count >= required_samples:
                            start_row = buoy_data.iloc[match_start_idx]
                            end_row = buoy_data.iloc[idx - 1]
                            start_t = start_row['time']
                            end_t = end_row['time']
                            actual_dur = int(round((end_t - start_t).total_seconds() / 60))
                            snapshot = {c.param: end_row.get(c.param, np.nan) for c in rule.conditions}
                            events.append(WarningEvent(
                                event_id=event_id_counter,
                                rule_name=rule.name,
                                level=rule.level,
                                buoy_id=buoy_id,
                                start_time=start_t,
                                end_time=end_t,
                                duration_minutes=actual_dur,
                                param_snapshot=snapshot
                            ))
                            event_id_counter += 1
                        match_start_idx = None
                        match_count = 0

                if match_start_idx is not None and match_count >= required_samples:
                    start_row = buoy_data.iloc[match_start_idx]
                    end_row = buoy_data.iloc[-1]
                    start_t = start_row['time']
                    end_t = end_row['time']
                    actual_dur = int(round((end_t - start_t).total_seconds() / 60))
                    snapshot = {c.param: end_row.get(c.param, np.nan) for c in rule.conditions}
                    events.append(WarningEvent(
                        event_id=event_id_counter,
                        rule_name=rule.name,
                        level=rule.level,
                        buoy_id=buoy_id,
                        start_time=start_t,
                        end_time=end_t,
                        duration_minutes=actual_dur,
                        param_snapshot=snapshot
                    ))
                    event_id_counter += 1

    for i, ev in enumerate(events):
        ev.event_id = i + 1

    return events


def events_to_dataframe(events: List[WarningEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            'event_id', 'rule_name', 'level', 'level_name', 'buoy_id',
            'start_time', 'end_time', 'duration_minutes', 'param_snapshot'
        ])

    rows = []
    for ev in events:
        rows.append({
            'event_id': ev.event_id,
            'rule_name': ev.rule_name,
            'level': ev.level,
            'level_name': WARNING_LEVELS.get(ev.level, {}).get('name', f'L{ev.level}'),
            'buoy_id': ev.buoy_id,
            'start_time': ev.start_time,
            'end_time': ev.end_time,
            'duration_minutes': ev.duration_minutes,
            'param_snapshot': ev.param_snapshot
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('start_time', ascending=False).reset_index(drop=True)
        df['event_id'] = range(1, len(df) + 1)
    return df


def compute_level_counts(events_df: pd.DataFrame) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(columns=['level', 'level_name', 'count'])
    counts = events_df.groupby(['level', 'level_name']).size().reset_index(name='count')
    counts = counts.sort_values('level')
    return counts


def compute_buoy_counts(events_df: pd.DataFrame) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(columns=['buoy_id', 'count'])
    counts = events_df.groupby('buoy_id').size().reset_index(name='count')
    counts = counts.sort_values('count', ascending=False)
    return counts


def compute_hourly_heatmap(events_df: pd.DataFrame, all_buoys: List[str]) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(0, index=all_buoys if all_buoys else [], columns=list(range(24)))

    df = events_df.copy()
    df['hour'] = df['start_time'].dt.hour

    buoy_list = all_buoys if all_buoys else list(df['buoy_id'].unique())

    heatmap_data = {}
    for buoy in buoy_list:
        heatmap_data[buoy] = {}
        buoy_df = df[df['buoy_id'] == buoy]
        for hour in range(24):
            cnt = len(buoy_df[buoy_df['hour'] == hour])
            heatmap_data[buoy][hour] = cnt

    heatmap_df = pd.DataFrame(heatmap_data).T
    heatmap_df = heatmap_df.reindex(columns=list(range(24)), fill_value=0)
    return heatmap_df


def serialize_rules(rules: List[WarningRule]) -> str:
    return json.dumps([r.to_dict() for r in rules], ensure_ascii=False, indent=2)


def deserialize_rules(json_str: str) -> List[WarningRule]:
    try:
        data = json.loads(json_str)
        return [WarningRule.from_dict(d) for d in data]
    except Exception:
        return []
