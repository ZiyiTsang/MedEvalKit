"""
Medical Keywords for Filter Framework

This module contains predefined keyword sets for filtering medical benchmark datasets.
Each keyword set is designed for specific medical domains or conditions.
"""

# Default dental/oral health keywords for filtering
DENTAL_KEYWORDS = {
    # English keywords
    'tooth', 'teeth', 'dental', 'dentist', 'oral', 'cavity', 'caries', 
    'enamel', 'dentin', 'gum', 'gingiva', 'molar', 'incisor', 'canine',
    'premolar', 'wisdom tooth', 'root canal', 'crown', 'bridge', 
    'implant', 'orthodontic', 'braces', 'periodontal', 'plaque',
    'tartar', 'filling', 'extraction', 'denture', 'prosthesis',
    
    # Chinese keywords  
    '牙齿', '牙', '口腔', '牙科', '牙医', '龋齿', '蛀牙', '牙釉质',
    '牙本质', '牙龈', '臼齿', '门牙', '犬齿', '智齿', '根管', '牙冠',
    '牙桥', '种植牙', '正畸', '牙套', '牙周', '牙菌斑', '牙石',
    '补牙', '拔牙', '假牙', '义齿'
}

# Cardiovascular-related keywords
CARDIAC_KEYWORDS = {
    # English keywords
    'heart', 'cardiac', 'cardiovascular', 'coronary', 'myocardial',
    'artery', 'vein', 'aorta', 'valve', 'rhythm', 'pulse', 'blood pressure',
    'hypertension', 'hypotension', 'arrhythmia', 'tachycardia', 'bradycardia',
    'infarction', 'ischemia', 'stenosis', 'embolism', 'thrombosis',
    
    # Chinese keywords
    '心脏', '心血管', '冠状动脉', '心肌', '动脉', '静脉', '主动脉',
    '心律', '血压', '高血压', '低血压', '心律失常', '心动过速',
    '心动过缓', '心肌梗死', '缺血', '狭窄', '血栓'
}

# Cancer-related keywords
CANCER_KEYWORDS = {
    # English keywords
    'cancer', 'tumor', 'malignant', 'benign', 'metastasis', 'oncology',
    'carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'chemotherapy',
    'radiation', 'biopsy', 'histology', 'grade', 'stage',
    
    # Chinese keywords
    '癌症', '肿瘤', '恶性', '良性', '转移', '肿瘤学', '癌',
    '肉瘤', '淋巴瘤', '白血病', '化疗', '放疗', '活检',
    '组织学', '分级', '分期'
}

# Neurological keywords
NEUROLOGICAL_KEYWORDS = {
    # English keywords
    'brain', 'neurological', 'stroke', 'seizure', 'epilepsy', 'migraine',
    'dementia', 'alzheimer', 'parkinson', 'multiple sclerosis', 'neuropathy',
    'neuron', 'cortex', 'spinal cord', 'cranial nerve',
    
    # Chinese keywords
    '大脑', '神经', '中风', '癫痫', '偏头痛', '痴呆',
    '阿尔茨海默', '帕金森', '多发性硬化', '神经病',
    '神经元', '皮层', '脊髓', '脑神经'
}

# Respiratory keywords
RESPIRATORY_KEYWORDS = {
    # English keywords
    'lung', 'respiratory', 'pulmonary', 'asthma', 'bronchitis', 'pneumonia',
    'copd', 'emphysema', 'tuberculosis', 'cough', 'breathing', 'airway',
    
    # Chinese keywords
    '肺', '呼吸', '肺部', '哮喘', '支气管炎', '肺炎',
    '慢阻肺', '肺气肿', '结核病', '咳嗽', '呼吸', '气道'
}

# General medical keywords (fallback)
GENERAL_MEDICAL_KEYWORDS = {
    # English keywords
    'medical', 'diagnosis', 'treatment', 'patient', 'symptom', 'disease',
    'medicine', 'therapy', 'clinical', 'hospital', 'doctor', 'physician',
    'surgery', 'medication', 'prescription', 'examination', 'test',
    
    # Chinese keywords
    '医学', '诊断', '治疗', '病人', '症状', '疾病',
    '药物', '治疗', '临床', '医院', '医生', '医师',
    '手术', '药物', '处方', '检查', '测试'
}

# Predefined keyword sets for common medical domains
KEYWORD_SETS = {
    'dental': DENTAL_KEYWORDS,
    'cardiac': CARDIAC_KEYWORDS,
    'cancer': CANCER_KEYWORDS,
    'neurological': NEUROLOGICAL_KEYWORDS,
    'respiratory': RESPIRATORY_KEYWORDS,
    'general': GENERAL_MEDICAL_KEYWORDS
}


def get_keywords_for_category(category: str) -> set:
    """
    Get keyword set for a specific medical category
    
    Args:
        category: Medical category name (dental, cardiac, cancer, etc.)
        
    Returns:
        Set of keywords for the specified category
    """
    return KEYWORD_SETS.get(category.lower(), GENERAL_MEDICAL_KEYWORDS)


def get_all_keywords() -> set:
    """
    Get all available keywords combined
    
    Returns:
        Set containing all keywords from all categories
    """
    all_keywords = set()
    for keyword_set in KEYWORD_SETS.values():
        all_keywords.update(keyword_set)
    return all_keywords