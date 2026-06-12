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
    prerequisite_rule: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            'name': self.name,
            'level': int(self.level),
            'conditions': [c.to_dict() for c in self.conditions],
            'duration_minutes': int(self.duration_minutes),
            'enabled': bool(self.enabled),
        }
        if self.prerequisite_rule is not None:
            d['prerequisite_rule'] = self.prerequisite_rule
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'WarningRule':
        return cls(
            name=d['name'],
            level=int(d['level']),
            conditions=[Condition.from_dict(c) for c in d.get('conditions', [])],
            duration_minutes=int(d.get('duration_minutes', 0)),
            enabled=bool(d.get('enabled', True)),
            prerequisite_rule=d.get('prerequisite_rule', None)
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
    effective_level: int = 0
    level_tag: str = ""
    group_id: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            'event_id': self.event_id,
            'rule_name': self.rule_name,
            'level': int(self.level),
            'buoy_id': self.buoy_id,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': int(self.duration_minutes),
            'param_snapshot': {k: (float(v) if pd.notna(v) else None) for k, v in self.param_snapshot.items()},
            'effective_level': int(self.effective_level),
            'level_tag': self.level_tag,
            'group_id': self.group_id
        }
        return d


def detect_cycle_dependency(rules: List[WarningRule], new_rule_name: str, new_prerequisite: Optional[str]) -> bool:
    if new_prerequisite is None:
        return False
    rule_map = {r.name: r.prerequisite_rule for r in rules}
    rule_map[new_rule_name] = new_prerequisite
    visited = set()
    current = new_rule_name
    while current is not None:
        if current in visited:
            return True
        visited.add(current)
        current = rule_map.get(current)
    return False


def get_prerequisite_chain(rule_name: str, rules: List[WarningRule]) -> List[str]:
    chain = []
    rule_map = {r.name: r for r in rules}
    current = rule_name
    while current is not None:
        rule = rule_map.get(current)
        if rule is None:
            break
        if rule.prerequisite_rule is not None:
            chain.append(rule.prerequisite_rule)
            current = rule.prerequisite_rule
        else:
            break
    return chain


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
            enabled=True,
            prerequisite_rule=None
        ),
        WarningRule(
            name="大浪预警",
            level=3,
            conditions=[
                Condition(param='Hs', operator='>', threshold=4.0)
            ],
            duration_minutes=30,
            enabled=True,
            prerequisite_rule="台风预警"
        ),
        WarningRule(
            name="低温预警",
            level=2,
            conditions=[
                Condition(param='SST', operator='<', threshold=5.0)
            ],
            duration_minutes=120,
            enabled=True,
            prerequisite_rule=None
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


def _check_prerequisite_triggered(
    rule: WarningRule,
    buoy_id: str,
    all_events_by_buoy_rule: Dict[Tuple[str, str], List[WarningEvent]],
    current_time: pd.Timestamp
) -> bool:
    if rule.prerequisite_rule is None:
        return True
    key = (buoy_id, rule.prerequisite_rule)
    prereq_events = all_events_by_buoy_rule.get(key, [])
    for ev in prereq_events:
        if ev.start_time <= current_time <= ev.end_time:
            return True
    last_event = None
    for ev in prereq_events:
        if ev.start_time <= current_time:
            if last_event is None or ev.start_time > last_event.start_time:
                last_event = ev
    if last_event is not None and last_event.end_time >= current_time:
        return True
    return False


def topo_sort_rules(rules: List[WarningRule]) -> List[WarningRule]:
    rule_map = {r.name: r for r in rules}
    result = []
    visited = set()
    visiting = set()

    def _visit(name: str):
        if name in visited:
            return
        if name in visiting:
            return
        visiting.add(name)
        rule = rule_map.get(name)
        if rule and rule.prerequisite_rule and rule.prerequisite_rule in rule_map:
            _visit(rule.prerequisite_rule)
        visiting.discard(name)
        visited.add(name)
        if rule:
            result.append(rule)

    for r in rules:
        _visit(r.name)
    return result


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

    enabled_rules = topo_sort_rules(enabled_rules)

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

    base_events_by_buoy_rule: Dict[Tuple[str, str], List[WarningEvent]] = {}

    for buoy_id in buoy_ids:
        buoy_data = data[data['buoy_id'] == buoy_id].sort_values('time').reset_index(drop=True)
        if len(buoy_data) == 0:
            continue

        for rule in enabled_rules:
            dur_min = rule.duration_minutes

            if dur_min <= 0:
                for idx, row in buoy_data.iterrows():
                    t = row['time']
                    if rule.prerequisite_rule is not None:
                        prereq_key = (buoy_id, rule.prerequisite_rule)
                        prereq_events = base_events_by_buoy_rule.get(prereq_key, [])
                        prereq_triggered = False
                        for ev in prereq_events:
                            if ev.start_time <= t:
                                prereq_triggered = True
                                break
                        if not prereq_triggered:
                            continue

                    if rule.evaluate_row(row):
                        snapshot = {c.param: row.get(c.param, np.nan) for c in rule.conditions}
                        ev = WarningEvent(
                            event_id=event_id_counter,
                            rule_name=rule.name,
                            level=rule.level,
                            buoy_id=buoy_id,
                            start_time=t,
                            end_time=t,
                            duration_minutes=0,
                            param_snapshot=snapshot
                        )
                        events.append(ev)
                        key = (buoy_id, rule.name)
                        base_events_by_buoy_rule.setdefault(key, []).append(ev)
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
                    t = row['time']

                    if rule.prerequisite_rule is not None:
                        prereq_key = (buoy_id, rule.prerequisite_rule)
                        prereq_events = base_events_by_buoy_rule.get(prereq_key, [])
                        prereq_triggered = False
                        for ev in prereq_events:
                            if ev.start_time <= t:
                                prereq_triggered = True
                                break
                        if not prereq_triggered:
                            if match_start_idx is not None and match_count >= required_samples:
                                start_row = buoy_data.iloc[match_start_idx]
                                end_row = buoy_data.iloc[idx - 1]
                                start_t = start_row['time']
                                end_t = end_row['time']
                                actual_dur = int(round((end_t - start_t).total_seconds() / 60))
                                snapshot = {c.param: end_row.get(c.param, np.nan) for c in rule.conditions}
                                ev = WarningEvent(
                                    event_id=event_id_counter,
                                    rule_name=rule.name,
                                    level=rule.level,
                                    buoy_id=buoy_id,
                                    start_time=start_t,
                                    end_time=end_t,
                                    duration_minutes=actual_dur,
                                    param_snapshot=snapshot
                                )
                                events.append(ev)
                                key = (buoy_id, rule.name)
                                base_events_by_buoy_rule.setdefault(key, []).append(ev)
                                event_id_counter += 1
                            match_start_idx = None
                            match_count = 0
                            continue

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
                            ev = WarningEvent(
                                event_id=event_id_counter,
                                rule_name=rule.name,
                                level=rule.level,
                                buoy_id=buoy_id,
                                start_time=start_t,
                                end_time=end_t,
                                duration_minutes=actual_dur,
                                param_snapshot=snapshot
                            )
                            events.append(ev)
                            key = (buoy_id, rule.name)
                            base_events_by_buoy_rule.setdefault(key, []).append(ev)
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
                    ev = WarningEvent(
                        event_id=event_id_counter,
                        rule_name=rule.name,
                        level=rule.level,
                        buoy_id=buoy_id,
                        start_time=start_t,
                        end_time=end_t,
                        duration_minutes=actual_dur,
                        param_snapshot=snapshot
                    )
                    events.append(ev)
                    key = (buoy_id, rule.name)
                    base_events_by_buoy_rule.setdefault(key, []).append(ev)
                    event_id_counter += 1

    for i, ev in enumerate(events):
        ev.event_id = i + 1

    return events


def apply_escalation(
    events: List[WarningEvent],
    rules: List[WarningRule],
    all_buoys: List[str],
    persisted_state: Optional[Dict[Tuple[str, str], Dict]] = None
) -> Tuple[List[WarningEvent], Dict[Tuple[str, str], Dict]]:
    if persisted_state is None:
        persisted_state = {}

    new_state: Dict[Tuple[str, str], Dict] = {}
    rule_level_map = {r.name: r.level for r in rules}

    rule_names = [r.name for r in rules]
    all_keys: List[Tuple[str, str]] = []
    for b in all_buoys:
        for rn in rule_names:
            all_keys.append((b, rn))

    from collections import defaultdict
    event_groups = defaultdict(list)
    for ev in events:
        event_groups[(ev.buoy_id, ev.rule_name)].append(ev)

    for key in all_keys:
        buoy_id, rule_name = key
        original_level = rule_level_map.get(rule_name, 1)
        old = persisted_state.get(key, {'escalation_offset': 0, 'miss_scan_count': 0})
        offset = int(old.get('escalation_offset', 0))
        miss_count = int(old.get('miss_scan_count', 0))

        group_events = event_groups.get(key, [])

        if len(group_events) == 0:
            miss_count += 1
            if miss_count >= 3:
                offset = max(0, offset - 1)
                miss_count = 0
        else:
            miss_count = 0
            group_events.sort(key=lambda e: e.start_time)
            last_ev = None
            for i, ev in enumerate(group_events):
                if i > 0 and last_ev is not None:
                    prev_dur = max(last_ev.duration_minutes, 1)
                    gap_min = (ev.start_time - last_ev.end_time).total_seconds() / 60
                    if gap_min <= prev_dur * 2:
                        offset = min(offset + 1, 4 - original_level)
                last_ev = ev

            effective = min(original_level + offset, 4)
            for ev in group_events:
                ev.effective_level = effective
                if effective > original_level:
                    ev.level_tag = "↑升级"
                else:
                    ev.level_tag = "↓原始"

        new_state[key] = {
            'escalation_offset': offset,
            'miss_scan_count': miss_count
        }

    for ev in events:
        if ev.effective_level == 0:
            ev.effective_level = ev.level
            ev.level_tag = "↓原始"

    return events, new_state


def build_composite_groups(events: List[WarningEvent]) -> Tuple[List[WarningEvent], List[Dict]]:
    if not events:
        return events, []

    from collections import defaultdict
    buoy_events = defaultdict(list)
    for ev in events:
        buoy_events[ev.buoy_id].append(ev)

    group_counter = 1
    composite_groups = []
    assigned = set()

    for buoy_id, buoy_evts in buoy_events.items():
        buoy_evts.sort(key=lambda e: e.start_time)
        used = set()

        for i, ev in enumerate(buoy_evts):
            if i in used:
                continue

            group_members = [i]
            window_start = ev.start_time
            window_end = ev.start_time + timedelta(minutes=5)

            for j in range(i + 1, len(buoy_evts)):
                if j in used:
                    continue
                other = buoy_evts[j]
                if other.start_time <= window_end and other.rule_name != ev.rule_name:
                    group_members.append(j)
                    used.add(j)

            if len(group_members) > 1:
                used.add(i)
                gid = f"G-{group_counter}"
                group_counter += 1

                member_events = [buoy_evts[idx] for idx in group_members]
                max_level = max(e.effective_level for e in member_events)
                min_time = min(e.start_time for e in member_events)
                max_time = max(e.end_time for e in member_events)
                rule_names = list(set(e.rule_name for e in member_events))

                for idx in group_members:
                    buoy_evts[idx].group_id = gid

                member_dicts = []
                for e in member_events:
                    member_dicts.append({
                        'event_id': e.event_id,
                        'rule_name': e.rule_name,
                        'level': e.level,
                        'effective_level': e.effective_level,
                        'level_tag': e.level_tag,
                        'buoy_id': e.buoy_id,
                        'start_time': e.start_time,
                        'end_time': e.end_time,
                        'duration_minutes': e.duration_minutes,
                        'param_snapshot': e.param_snapshot.copy() if e.param_snapshot else {}
                    })

                composite_groups.append({
                    'group_id': gid,
                    'buoy_id': buoy_id,
                    'max_level': max_level,
                    'rule_count': len(rule_names),
                    'rule_names': rule_names,
                    'start_time': min_time,
                    'end_time': max_time,
                    'events': member_events,
                    'events_dicts': member_dicts
                })

    return events, composite_groups


def compute_daily_trend(events_df: pd.DataFrame) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(columns=['date', 'level', 'count'])

    df = events_df.copy()
    df['date'] = df['start_time'].dt.date

    level_counts = df.groupby(['date', 'effective_level']).size().reset_index(name='count')
    level_counts.columns = ['date', 'level', 'count']

    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D').date
    all_levels = sorted(df['effective_level'].unique())

    full_index = []
    for d in all_dates:
        for l in all_levels:
            full_index.append((d, l))

    full_df = pd.DataFrame(full_index, columns=['date', 'level'])
    result = full_df.merge(level_counts, on=['date', 'level'], how='left')
    result['count'] = result['count'].fillna(0).astype(int)

    return result


def compute_daily_totals(events_df: pd.DataFrame) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(columns=['date', 'total_count'])

    df = events_df.copy()
    df['date'] = df['start_time'].dt.date
    daily = df.groupby('date').size().reset_index(name='total_count')
    return daily


def compute_7day_moving_avg(daily_totals: pd.DataFrame) -> pd.DataFrame:
    if len(daily_totals) == 0:
        return pd.DataFrame(columns=['date', 'total_count', 'ma7'])

    result = daily_totals.copy()
    result = result.sort_values('date')
    result['ma7'] = result['total_count'].rolling(window=7, min_periods=1).mean()
    return result


def detect_anomaly_days(ma_df: pd.DataFrame) -> List:
    if len(ma_df) == 0 or 'ma7' not in ma_df.columns:
        return []

    anomalies = []
    for _, row in ma_df.iterrows():
        if pd.notna(row.get('ma7')) and row['ma7'] > 0:
            if row['total_count'] > row['ma7'] * 2:
                anomalies.append(row['date'])
    return anomalies


def events_to_dataframe(events: List[WarningEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            'event_id', 'rule_name', 'level', 'level_name', 'buoy_id',
            'start_time', 'end_time', 'duration_minutes', 'param_snapshot',
            'effective_level', 'effective_level_name', 'level_tag', 'group_id'
        ])

    rows = []
    for ev in events:
        eff = ev.effective_level if ev.effective_level > 0 else ev.level
        rows.append({
            'event_id': ev.event_id,
            'rule_name': ev.rule_name,
            'level': ev.level,
            'level_name': WARNING_LEVELS.get(ev.level, {}).get('name', f'L{ev.level}'),
            'buoy_id': ev.buoy_id,
            'start_time': ev.start_time,
            'end_time': ev.end_time,
            'duration_minutes': ev.duration_minutes,
            'param_snapshot': ev.param_snapshot,
            'effective_level': eff,
            'effective_level_name': WARNING_LEVELS.get(eff, {}).get('name', f'L{eff}'),
            'level_tag': ev.level_tag,
            'group_id': ev.group_id
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('start_time', ascending=False).reset_index(drop=True)
        df['event_id'] = range(1, len(df) + 1)
    return df


def compute_level_counts(events_df: pd.DataFrame, use_effective: bool = True) -> pd.DataFrame:
    if len(events_df) == 0:
        return pd.DataFrame(columns=['level', 'level_name', 'count'])

    level_col = 'effective_level' if use_effective and 'effective_level' in events_df.columns else 'level'
    level_name_col = 'effective_level_name' if use_effective and 'effective_level_name' in events_df.columns else 'level_name'

    if level_col not in events_df.columns:
        level_col = 'level'
        level_name_col = 'level_name'

    counts = events_df.groupby([level_col, level_name_col]).size().reset_index(name='count')
    counts.columns = ['level', 'level_name', 'count']
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
