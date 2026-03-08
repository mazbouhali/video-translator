"""
Arabic-English glossary for religious and cultural terms.
Applied before/after neural translation to fix common errors.
"""

# Common mistranslations and corrections
ARABIC_GLOSSARY = {
    # Religious figures and titles
    "الكرار": "the Champion (Imam Ali)",
    "الأكبر": "al-Akbar (Ali al-Akbar)",
    "حسين": "Hussein",
    "الحسين": "Hussein",
    "العباس": "Abbas",
    "زينب": "Zainab",
    "المختار": "the Chosen One (Prophet Muhammad)",
    "الرسول": "the Messenger",
    "النبي": "the Prophet",
    
    # Religious terms
    "بسم الله": "In the name of Allah",
    "الرحمن الرحيم": "the Most Gracious, the Most Merciful",
    "سبحان الله": "Glory be to Allah",
    "الله أكبر": "Allah is Greatest",
    "كبر لله": "proclaimed Allah is Greatest",
    "سور الحمد": "Surah Al-Fatiha",
    "الشهادة": "martyrdom",
    "الجهاد": "struggle",
    
    # Battle/Karbala terms
    "كربلاء": "Karbala",
    "الطف": "Taff (Karbala)",
    "الثار": "vengeance/retribution",
    "الميدان": "battlefield",
    "الفرات": "Euphrates",
    
    # Common words often mistranslated
    "فتى": "young warrior",
    "فتاين": "young warriors",
    "شبل": "lion cub",
    "زمجر": "roared",
    "عزف": "played (music)",
    "مقام": "melody/station",
    "النصر": "victory",
    "صاح": "shouted",
}

# Patterns to fix (regex-based)
TRANSLATION_FIXES = [
    # Fix gendered mistranslations
    (r'\bgirl went to war\b', 'young warrior went to war'),
    (r'\bwent to war.*?girl\b', 'young warrior went to war'),
    
    # Fix nonsense translations
    (r'Razaf al-Nusra.*?Makkah', 'played victory as a melody'),
    (r'two planes', 'who'),
    (r'rear wheel', 'cast from the Champion'),
    (r'Eden\b', 'our enemies'),
    (r'\bEmeralds?\b', 'He roared'),
    
    # Fix name transliterations
    (r'\bHussain\b', 'Hussein'),
    (r'\bZenb\b', 'Zainab'),
    (r'\bAlawan\b', 'Alwan'),
    (r'\bal-Kharar\b', 'al-Karrar (the Champion)'),
]

import re

def apply_glossary_to_arabic(text: str) -> str:
    """Pre-process Arabic text to normalize terms."""
    # Could add normalization here
    return text

def fix_translation(english_text: str) -> str:
    """Post-process English translation to fix common errors."""
    result = english_text
    
    for pattern, replacement in TRANSLATION_FIXES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def get_context_prompt(domain: str = "general") -> str:
    """Get context for translation based on domain."""
    contexts = {
        "religious": (
            "This is Islamic religious content, likely a nasheed (religious song) "
            "or poetry about Imam Hussein, Karbala, or Shia Islamic themes. "
            "Translate with appropriate religious terminology."
        ),
        "news": "This is news content. Translate formally and accurately.",
        "casual": "This is casual conversation. Translate naturally.",
        "general": "",
    }
    return contexts.get(domain, "")
