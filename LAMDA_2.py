import json
import re

json_path = "./LAMDA.json"

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for i, line in enumerate(data["Q2"]):
    orig = line['original']
    prep_list = re.findall('\([%.0-9a-zA-Z가-힣\s]+\)/\([%.0-9a-zA-Z가-힣\s]+\)', orig)
    prep_list = list(set(prep_list))
    
    # print(prep_list)
    modified = orig
    # 1차 필터링: ()/() 형태에서 숫자 있는 부분은 제거하고
    for idx, sentence in enumerate(prep_list):
        a, b = sentence.split('/')
        if re.search('[0-9]+', b) is None:
            modified = modified.replace(sentence, b[1:-1])
        else:
            modified = modified.replace(sentence, a[1:-1])
    
    # 2차 필터링: 특수문자, b/, o/ 이런것들 제거
    modified = re.sub('[0-9A-Za-z]*/[\s]*', '',modified)
    modified = re.sub('[^0-9A-Za-z가-힣.,?!\s]', '', modified)

    # 3차 필터링: 영어 단어 ex) NCS, IV, npc
    eng_to_kor = {
        'a': '에이',
        'b': '비',
        'c': '씨',
        'd': '디',
        'e': '이',
        'f': '에프',
        'g': '쥐',
        'h': '에이치',
        'i': '아이',
        'j': '제이',
        'k': '케이',
        'l': '엘',
        'm': '엠',
        'n': '엔',
        'o': '오',
        'p': '피',
        'q': '큐',
        'r': '알',
        's': '에스',
        't': '티',
        'u': '유',
        'v': '브이',
        'w': '더블유',
        'x': '엑스',
        'y': '와이',
        'z': '지'
    }
    for alphabet in eng_to_kor:
        modified = modified.replace(alphabet, eng_to_kor[alphabet])
        modified = modified.replace(alphabet.upper(), eng_to_kor[alphabet])

    data["Q2"][i]["new"] = modified
    # print(modified)

with open(json_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent='\t', ensure_ascii=False)