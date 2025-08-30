import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional
from debugger import DebugMixin

class PhoneticDictionary(DebugMixin):
    """
    Phonetic dictionary class for managing word-to-phoneme mappings.
    Supports multiple languages, custom dictionaries, and extensibility.
    """
    
    # Default Dutch phonetic dictionary
    DUTCH_PHONETIC_DICT = {
            'aan': 'aːn',
            'al': 'ɑl',
            'als': 'ɑls',
            'alsof': 'ˈɑlsɔf',
            '8': 'ɑxt',
            'bak': 'bɑk',
            'bakker': 'ˈbɑkər',
            'betovering': 'bəˈtoːvərɪŋ',
            'bevrijd': 'bəˈvrɛit',
            'bij': 'bɛi',
            'bloedrode': 'ˈblutˌrodə',
            'braadde': 'ˈbradə',            
            'een': 'ən',  # unstressed form
            '1': 'eːn',
            'elf': 'ɛlf',
            '11': 'ɛlf',
            'er': 'ɛr',            
            'daarna': 'dɑrˈna',
            'dak': 'dɑk',
            'dakker': 'ˈdɑkər',
            'dan': 'dɑn',
            'dat': 'dɑt',
            'dauwdruppel': 'ˈdɑuˌdrʏpəl',
            'de': 'də',
            'deur': 'døːr',
            'die': 'di',
            'direct': 'diˈrɛkt',
            'dit': 'dɪt',
            'donkere': 'ˈdɔŋkərə',
            'doodsbang': 'ˈdotsbɑŋ',
            'door': 'doːr',
            'drie': 'dri',
            '3': 'dri',
            'en': 'ɛn',
            'erheen': "ɛrˈheːn",
            'geen': 'ɣeːn',
            'gefluit': 'xəˈflœyt',
            'groen': 'ɣruːn',             
            'haar': 'haːr',
            'had': 'hɑt',  # final devoicing
            'helemaal': 'ˌheləˈmaːl',
            'helft': 'hɛlft',
            'hem': 'hɛm',
            'het': 'hɛt',  # stressed form
            'hij': 'hɛi',
            'hierop': 'ˈhirɔp',
            'hoe': 'hu',
            'hun': 'hʏn',
            'je': 'jə',
            'in': 'ɪn',
            'juist': 'jœyst', 
            'kasteel': 'kɑsˈteːl',
            'komt': 'kɔmt',
            'kin': 'kɪn',            
            'kwamen': 'ˈkʋaːmən',
            'lij': 'lɛi',
            'maantje': 'ˈmaːntjə',
            'maar': 'maːr',
            'meisjes': 'ˈmɛiʃəs',
            'met': 'mɛt',
            'mij': 'mɛi',
            'mijn': 'mɛin',
            'moment': 'moˈmɛnt',
            'mooi': 'moːi',
            'mooie': 'ˈmoːiə',
            'naar': 'naːr',
            'nachtegalen': 'ˈnɑxtəˌɣaːlən',
            'negen': 'ˈneːɣən',
            '9': 'ˈneːɣən',
            'niet': 'nit',
            'nog': 'nɔx', 
            'noordenwind': 'ˈnoːrdənˌʋɪnt',
            'nu': 'ny',    
            'of': 'ɔf',
            'om': 'ɔm',
            'onmiddellijk': 'ɔnˈmɪdələk',
            'onschuldig': 'ˈɔnsxʏldəx',
            'ook': 'oːk',
            'over': 'ˈoːvər',
            'redetwisten': 'ˌreːdəˈtʋɪstən', 
            '`s': 's',
            'schold': 'sxɔlt',  # final devoicing
            'smeekte': 'ˈsmeːktə',
            'sok': 'sɔk',
            'spreuk': 'sprøːk',
            'sprong': 'sprɔŋ',
            'sterkste': 'ˈstɛrkstə',
            'stiekem': 'ˈstiːkəm',
            'stilstaan': 'ˈstɪlstaːn',
            'struik': 'strœyk',
            'te': 'tə',
            'teruggekregen': 'təˈrʏxɣəˌkreːɣən',
            'terugvinden': 'təˈrʏxˌfɪndən',
            'tien': 'tin',
            '10': 'tin',
            'tot': 'tɔt',
            'twaalf': 'tʋaːlf',
            '12': 'tʋaːlf',
            'twee': 'tʋeː',
            '2': 'tʋeː',
            'tussen': 'ˈtʏsən',
            'tuwiet': 'tyˈʋit',
            'uit': 'œyt',
            'uittrekken': 'ˈœytˌtrɛkən',    
            'vak': 'vɑk',
            'van': 'vɑn',
            'veel': 'veːl',
            'verdwaald': 'vərˈdʋaːlt',
            'verlost': 'vərˈlɔst', 
            'verstijfde': "vərˈstɛivdə",
            'vier': 'viːr',
            '4': 'viːr',
            'vijf': 'vɛif',
            '5': 'vɛif',
            'vlakbij': "ˈvlɑkbɛi",
            'vloog': 'vloːx',
            'vogeltje': 'ˈvoːɣəltjə',
            'vogelkooitje': 'ˈvoːɣəlˌkoːitjə',
            'voor': 'voːr',
            'wak': 'ʋɑk',
            'was': 'ʋɑs',
            'waren': 'ˈʋaːrən',
            'wat': 'ʋɑt',
            'wegpakte': 'ˈʋɛxpɑktə',
            'wel': 'ʋɛl',
            'werd': 'ʋɛrt',
            'zandbak': 'ˈzɑntbɑk',  # final devoicing of d 
            'zanddak': 'ˈzɑndɑk',  # gemination typically results in single consonant
            'zich': 'zɪx',
            'zijn': 'zɛin',
            'ze': 'zə',
            'zei': 'zɛi',
            'zeven': 'ˈzeːvən',
            '7': 'ˈzeːvən',
            'zevenduizend': 'ˌzeːvənˈdœyzənt',
            'zo': 'zoː',
            'zonlicht': 'ˈzɔnlɪxt',
            'zou': 'zɑu'  # not 'zʏlə(n)' which would be 'zullen'
                }
    
    # Dictionary of common Dutch phoneme mappings (letter to phoneme)
    DUTCH_PHONEME_MAP = {
        # Basic vowels
        'a': 'a', 'e': 'ə', 'i': 'i', 'o': 'o', 'u': 'u', 
        # Diphthongs
        'ij': 'ɛi', 'ei': 'ɛi', 'ui': 'œy', 'eu': 'ø', 'oe': 'u', 
        'ou': 'ɑu', 'au': 'ɑu', 'ie': 'i', 'ee': 'e', 'oo': 'o', 'uu': 'y',
        # Consonants
        'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'x', 'h': 'h', 'j': 'j', 'k': 'k',
        'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'v': 'v',
        'w': 'w', 'z': 'z', 'ch': 'x', 'ng': 'ŋ', 'nk': 'ŋk', 'sch': 'sx',
        # Additional special cases
        'aa': 'a:', 'ee': 'e:', 'oo': 'o:', 'uu': 'y:',  # Long vowels
    }
    
    # List of complex phonemes that count as single units
    COMPLEX_PHONEMES = [
            # Existing complex phonemes
            'ɛi', 'œy', 'ɑu', 'ɵ:', 
            
            # Long vowels with IPA length markers
            'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'ɑː', 'ɔː', 'ɛː', 'ɛ:',
            
            # Additional complex sequences
            'tʋ', 'ʋɑ', 'ʋɪ', 'ʋə', 'ŋk', 'sx', 'tʃ', 'dʒ',
        ]
    
    def __init__(self, custom_dict_path: Optional[str] = None, debug_mode=False):
        """
        Initialize the phonetic dictionary.
        
        Parameters:
        -----------
        custom_dict_path : str or None
            Path to a custom dictionary file (JSON, TSV, or TXT format)
        """
        # Initialize the DebugMixin
        super().__init__(class_name="PhoneticDictionary", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        # Initialize base dictionary based on language
        self.dictionary = self.DUTCH_PHONETIC_DICT.copy()
        self.phoneme_map = self.DUTCH_PHONEME_MAP.copy()
        
        # Initialize custom entries
        self.custom_entries = {}
        
        # Load custom dictionary if provided
        if custom_dict_path is not None:
            self.load_custom_entries(custom_dict_path)
            
        # Create reverse phoneme map
        self.reverse_phoneme_map = self._create_reverse_phoneme_map()
        
    def __getitem__(self, word: str) -> Optional[str]:
        """
        Get phonetic transcription for a word 

        """
        # Check custom entries first, then fall back to base dictionary
        return self.custom_entries.get(word, self.dictionary.get(word))
    
    def __contains__(self, word: str) -> bool:
        """
        Check if a word is in the dictionary.

        """
        return word in self.custom_entries or word in self.dictionary
    
    def get_transcription(self, word: str) -> Optional[str]:
        return self[word]
    
    def add_word(self, word: str, transcription: str) -> None:

        self.custom_entries[word] = transcription
    
    def add_words(self, word_dict: Dict[str, str]) -> None:

        self.custom_entries.update(word_dict)    
    
    def count_phonemes(self, word: str) -> int:
        
        if word not in self:
            # Rough estimation based on Dutch spelling rules
            return max(1, len(word) - word.count('oe') - word.count('ie') - 
                      word.count('ui') - word.count('ij') - word.count('eu'))
        
        # Get phonetic transcription
        transcription = self[word]
        
        cleaned = self.clean_transcription(transcription)
        
        # Initialize phoneme count
        phoneme_count = len(cleaned)
        
        # Adjust for complex phonemes
        for cp in self.COMPLEX_PHONEMES:
            phoneme_count -= cleaned.count(cp)
        
        return max(1, phoneme_count)
    
    def extract_phonemes(self, word: str) -> List[str]:
        """Extract individual phonemes from a word's transcription."""
        if word not in self:
            # Workaround! If word not in dictionary, return characters as approximation
            return list(word)
        
        # Get phonetic transcription
        transcription = self[word]
        
        # Clean transcription - CENTRALIZED CLEANING
        cleaned = self.clean_transcription(transcription)
        
        # Extract phonemes
        phonemes = []
        i = 0
        
        while i < len(cleaned):
            # Check for complex phonemes in descending order of length
            complex_found = False
            # Sort complex phonemes by length (longest first) to prevent substring matches
            sorted_complex = sorted(self.COMPLEX_PHONEMES, key=len, reverse=True)
            
            for cp in sorted_complex:
                if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                    phonemes.append(cp)
                    i += len(cp)
                    complex_found = True
                    break
            
            # Check for length markers (IMPORTANT: this should be part of the previous phoneme)
            if not complex_found and i + 1 < len(cleaned) and cleaned[i+1] == 'ː':
                phonemes.append(cleaned[i:i+2])  # Include the length marker with the vowel
                i += 2
                complex_found = True
            
            if not complex_found:
                phonemes.append(cleaned[i])
                i += 1
        
        return phonemes
        
    def clean_transcription(self, transcription: str) -> str:
        """Remove stress markers and other diacritics from transcription."""
        cleaned = transcription
        # Remove ALL stress and prosodic markers consistently
        markers_to_remove = ['ˈ', 'ˌ', '.', '|', '‖', '(', ')', "'"]
        for marker in markers_to_remove:
            cleaned = cleaned.replace(marker, '')
        return cleaned
    
    def _create_reverse_phoneme_map(self) -> Dict[str, str]:
        """
        Create a mapping from phonemes to letters.

        """
        reverse_map = {}
        for letter, phoneme in self.phoneme_map.items():
            if phoneme not in reverse_map:
                reverse_map[phoneme] = letter
        return reverse_map
    
    def get_phoneme_letter(self, phoneme: str) -> Optional[str]:

        return self.reverse_phoneme_map.get(phoneme)
 
    # Phoneme group level
    def add_phoneme_groups(self):
        """
        Add phoneme group mappings to the dictionary.
        """
        # Define phoneme groups
        self.phoneme_groups = {
            'alveolar': ['t', 'd', 's', 'z', 'n', 'l'],
            ##'alveolar plosive': ['t', 'd',],
            ##'alveolar fricative': ['s', 'z'], 
            ##'alveolar other': ['n', 'l'],
            'back_vowels': ['u', 'o', 'ɔ', 'a', 'ɑ', 'ɑu', 'œy', 'ə', 'oː', 'aː', 'ɔː', 'ɑː'],
            ##'back vowels close': ['u'],
            ##'back vowels mid-close': ['u', 'ɑu', 'œy', 'o', 'ɔ', 'ə'],
            ##'back vowels open': ['a', 'ɑ', 'œy'],
            'dorsal': ['k', 'g', 'x', 'ɣ', 'ŋ', 'χ', 'ʁ', 'ŋk'],
            'front_vowels': ['i', 'ɪ', 'e', 'ɛ', 'ɛi', 'y', 'ʏ', 'eː', 'iː', 'ɪː'],
            'labial': ['p', 'b', 'f', 'v', 'm', 'ʋ'],
            'palatal': ['ŋ', 'j', 'r', 'tʃ', 'dʒ', 'tɕ', 'dʑ'], 
            'glottal': ['h', 'ɦ', 'ʔ'],
            ##'plosive': ['p', 'b', 't', 'd', 'k', 'g'],
            ##'fricative': ['f', 'v', 's', 'z', 'x', 'h', 'ʃ', 'ʒ', 'sx'],
            ##'nasal': ['m', 'n', 'ŋ', 'ŋk'],
            ##'approximant': ['l', 'j', 'w'],
            ##'rhotic': ['r']
        }
        
        # Create reverse mapping from phoneme to group
        self.phoneme_to_group = {}
        for group, phonemes in self.phoneme_groups.items():
            for phoneme in phonemes:
                self.phoneme_to_group[phoneme] = group
        
        self.phoneme_to_group['ˌ'] = 'marker'
        self.phoneme_to_group['?'] = 'unknown'
    
        return self.phoneme_groups

    #def get_phoneme_group(self, phoneme):
        """
        Get the group a phoneme belongs to.
        """
     #   if not hasattr(self, 'phoneme_to_group'):
      #      self.add_phoneme_groups()
        
       # return self.phoneme_to_group.get(phoneme, None)
        
    def get_all_group_names(self):
        """Get all phoneme group names."""
        if not hasattr(self, 'phoneme_groups'):
            self.add_phoneme_groups()
        return list(self.phoneme_groups.keys()) + ['marker', 'unknown']
    

    def get_word_phoneme_groups(self, word):
        """
        Get the sequence of phoneme groups for a word.
        """
        if not hasattr(self, 'phoneme_groups'):
            self.add_phoneme_groups()
            
        # Get individual phonemes
        phonemes = self.extract_phonemes(word)
        
        # Map to groups
        groups = []
        for phoneme in phonemes:
            group = self.get_phoneme_group(phoneme)
            if group:
                groups.append(group)
            else:
                groups.append('unknown')
        
        return groups
        
        
    def map_phonemes_to_groups(self, phoneme_labels):
        """
        Map individual phoneme labels to their corresponding groups.
        """
        group_labels = []
        unknown_phonemes = set()
        
        for phoneme in phoneme_labels:
            # Skip unknown phonemes
            if phoneme == '?':
                group_labels.append('unknown')
                continue
                
            # Map phoneme to group
            if phoneme in self.phoneme_to_group:
                group_labels.append(self.phoneme_to_group[phoneme])
            else:
                unknown_phonemes.add(phoneme)
                group_labels.append('unknown')
        
        if unknown_phonemes:
            self.log(f"Warning: Found {len(unknown_phonemes)} phonemes without group mapping: {unknown_phonemes}")
        
        return group_labels