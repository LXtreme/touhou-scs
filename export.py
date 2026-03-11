import re
from time import perf_counter
from typing import Any, cast

import orjson
from gmdkit.models.prop.hsv import HSV
from gmdbuilder import Level, ObjectType, obj_prop
from touhou_scs.utils import translate_remap_string

print("RUNNING EXPORT")

start_time = perf_counter()

PROPERTY_REMAP_STRING = "442"
PROPERTY_GROUPS = "57"
PROPERTY_PULSE_HSV_STRING = "49"
GROUP_FIELDS = {"51", "71", "76", "373", "395", "401"}


def is_group_property_field(key: str) -> bool:
    return key in GROUP_FIELDS


def to_gmdbuilder_key(key: str) -> str:
    return f"a{key}"


def collect_unknown_g(val: Any, unknown_g_set: set[int]) -> None:
    if isinstance(val, int) and val >= 10000:
        unknown_g_set.add(val)
    elif isinstance(val, str):
        for result in re.findall(r"\b\d{5,}\b", val):
            group_num = int(result)
            if group_num >= 10000:
                unknown_g_set.add(group_num)
    elif isinstance(val, list):
        for item in val:
            collect_unknown_g(cast(Any, item), unknown_g_set)


def replace_unknown_groups_in_string(value: str, unknown_g_dict: dict[int, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        group_num = int(match.group(0))
        replacement = unknown_g_dict.get(group_num)
        return str(replacement if replacement is not None else group_num)

    return re.sub(r"\b\d{5,}\b", repl, value)



# level = Level.from_live_editor()
level = Level.from_file("touhou scs.gmd")



with open("triggers.json", "rb") as f:
    trigger_json = orjson.loads(f.read())

unknown_g_set: set[int] = set()
trigger_count = 0

for trigger in trigger_json["triggers"]:
    trigger_count += 1
    for value in trigger.values():
        collect_unknown_g(value, unknown_g_set)

unknown_g_dict: dict[int, int] = {}

for g in sorted(unknown_g_set):
    new_group = int(level.new.group())
    unknown_g_dict[g] = new_group
    print(f"Registered {g} -> Group {new_group}")

print(f"Unknown group registry complete: {len(unknown_g_dict)} groups registered\n")

for trigger in trigger_json["triggers"]:
    new_obj: ObjectType = {"a1": trigger["1"]}

    for key, value in trigger.items():
        if key == PROPERTY_GROUPS:
            group_data = value
            group_array: list[Any] = group_data if isinstance(group_data, list) else [group_data]

            resolved_groups: set[int] = set()
            for group_value in group_array:
                if isinstance(group_value, int) and group_value >= 10000:
                    resolved_groups.add(unknown_g_dict[group_value])
                elif isinstance(group_value, int):
                    resolved_groups.add(group_value)
                else:
                    raise TypeError(f"Invalid group value in property 57: {group_value!r}")

            new_obj["a57"] = resolved_groups

        elif is_group_property_field(key):
            target = value
            if isinstance(target, int) and target >= 10000 and target in unknown_g_dict:
                target = unknown_g_dict[target]
            new_obj[to_gmdbuilder_key(key)] = target

        elif key == PROPERTY_REMAP_STRING:
            if isinstance(value, str):
                remap_string = replace_unknown_groups_in_string(value, unknown_g_dict)
                remap_dict, _ = translate_remap_string(remap_string)
                new_obj[obj_prop.Trigger.Spawn.REMAPS] = remap_dict
            else:
                new_obj[obj_prop.Trigger.Spawn.REMAPS] = value

        elif key == PROPERTY_PULSE_HSV_STRING:
            if isinstance(value, str):
                new_obj[to_gmdbuilder_key(key)] = HSV.from_string(value)
            else:
                new_obj[to_gmdbuilder_key(key)] = value

        else:
            new_obj[to_gmdbuilder_key(key)] = value

    level.objects.append(new_obj)



level.export_to_file("touhou scs updated.gmd")
# level.export_to_live_editor()


elapsed = perf_counter() - start_time
print(f"EXPORT FINISHED IN {elapsed:.3f} SECONDS")
print(f"Processed {trigger_count} triggers")