"""dev 태스크 보조 함수 (설정 선택 전용 분할, test와 겹치지 않음)."""


def first_40(dataset):
    """KMMLU train 분할에서 과목별 앞 40문항 (결정적)."""
    return dataset.select(range(min(40, len(dataset))))
