def recursive_parse_settings(given_setting, modify_setting):
    for key, val in modify_setting.items():
        if isinstance(val, dict):
            recursive_parse_settings(given_setting[key], val)
        else:
            given_setting[key] = val
    return given_setting