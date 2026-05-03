def safe_label_encode(encoder, value, field_name):
    value = str(value).strip()
    if value not in encoder.classes_:
        raise ValueError(f"{field_name} 값이 학습 데이터에 없습니다: {value}")
    return int(encoder.transform([value])[0])


def make_peak_level(hour):
    if hour == 7:
        return 1
    elif hour == 8:
        return 2
    elif hour == 9:
        return 3
    elif hour == 17:
        return 4
    elif hour == 18:
        return 5
    elif hour == 19:
        return 6
    return 0