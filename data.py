from konlpy.tag import Okt


def load_data(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]
        
    return data


def process_data(raw_data):
    tokenizer = Okt()
    data_processed = [tokenizer.morphs(data, norm=True) for data in raw_data]
    
    return data_processed
    