import os
import json
import pandas as pd
import string
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
            'avonds': 'ˈaːvɔnts',
            '8': 'ɑxt',
            'bak': 'bɑk',
            'bakker': 'ˈbɑkər',
            'ballon': 'bɑˈlɔn',
            'bedreigt': 'bəˈdrɛixt',
            'betovering': 'bəˈtoːvərɪŋ',
            'bekent': 'bəˈkɛnt',
            'bevrijd': 'bəˈvrɛit',
            'bij': 'bɛi',
            'binnenplaats': 'ˈbɪnənˌplaːts',
            'bloedrode': 'ˈblutˌrodə',
            'boomstammen': 'ˈboːmˌstɑmən',
            'braadde': 'ˈbradə',  
            'brievenbus': 'ˈbrivə(n)bʏs',            
            'buurt': 'byːrt',            
            'canule': 'kaˈnylə',
            'daarna': 'dɑrˈna',
            'dak': 'dɑk',
            'dakker': 'ˈdɑkər',
            'dan': 'dɑn',
            'dat': 'dɑt',
            'dauwdruppel': 'ˈdɑuˌdrʏpəl',
            'de': 'də',
            'deken': 'ˈdeːkən',
            'deur': 'døːr',
            'dichtbij': 'ˈdɪxtbɛi',
            'die': 'di',
            'direct': 'diˈrɛkt',
            'dit': 'dɪt',
            'doei': 'dui',
            'donkere': 'ˈdɔŋkərə',
            'doodsbang': 'ˈdotsbɑŋ',
            'door': 'doːr',
            'drie': 'dri',
            '3': 'dri',      
            'duurder':'dyrdə',
            'een': 'ən',  # unstressed form
            'en': 'ɛn',
            'er': 'ər',
            'erheen': 'ərˈheːn',
            '1': 'eːn',
            'elf': 'ɛlf',
            '11': 'ɛlf',
            'fijn': 'fɛin',
            'gebeurt': 'ɣəˈbøːrt',
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
            'in': 'ɪn',
            'ja': 'jaː',
            'jammer': 'ˈjɑmər',
            'je': 'jə',
            'jeuk': 'jøːk',
            'juist': 'jœyst',
            'kasteel': 'kɑsˈteːl',
            'keer':'ker',
            'kin': 'kɪn',  
            'kleine': 'klɛinə',            
            'komt': 'kɔmt',
            'koud': 'kɑut',
            'kwamen': 'ˈkʋaːmən',
            'lachte':'ˈlɑxtə',
            'leeg': 'leːx',
            'lij': 'lɛi',
            'leuk': 'løːk',
            'longen': 'ˈlɔŋən',
            'maantje': 'ˈmaːntjə',
            'maar': 'maːr',
            'meisjes': 'ˈmɛiʃəs',
            'met': 'mɛt',
            'mij': 'mɛi',
            'mijn': 'mɛin',
            'moment': 'moˈmɛnt',
            'mond': 'mɔnt',
            'mooi': 'moːi',
            'mooie': 'ˈmoːiə',
            'morgens': 'ˈmɔrɣəns',
            'naar': 'naːr',
            'nachtegalen': 'ˈnɑxtəˌɣaːlən',
            'nee': 'neː',
            'negen': 'ˈneːɣən',
            '9': 'ˈneːɣən',
            'niet': 'nit',
            'nog': 'nɔx', 
            'noordenwind': 'ˈnoːrdənˌʋɪnt',
            'nu': 'ny',    
            'ochtends': 'ˈɔxtənts',
            'of': 'ɔf',
            'ogen': 'ˈoːɣən',
            'om': 'ɔm',
            'onmiddellijk': 'ɔnˈmɪdələk',
            'onschuldig': 'ˈɔnsxʏldəx',
            'ook': 'oːk',
            'op': 'ɔp',
            'over': 'ˈoːvər',
            'pak': 'pɑk',
            'politieagenten':'poˈlitsiaxɛntən',
            'radio': 'ˈraːdiˌoː',
            'redetwisten': 'ˌreːdəˈtʋɪstən', 
            's': 's',
            'schold': 'sxɔlt',  # final devoicing
            'smeekte': 'ˈsmeːktə',
            'sok': 'sɔk',
            'speeksel': 'ˈspeːksəl',
            'spreuk': 'sprøːk',
            'sprong': 'sprɔŋ',
            'sterkste': 'ˈstɛrkstə',
            'starten':'ˈstɑrtən',
            'stiekem': 'ˈstiːkəm',
            'stilstaan': 'ˈstɪlstaːn',
            'struik': 'strœyk',
            'tak': 'tɑk',
            'te': 'tə',
            'teruggekregen': 'təˈrʏxɣəˌkreːɣən',
            'terugvinden': 'təˈrʏxˌfɪndən',
            'tien': 'tin',
            'tuin':'tœyn',
            '10': 'tin',
            'tot': 'tɔt',
            'totdat': 'tɔˈdɑt',
            'twaalf': 'tʋaːlf',
            '12': 'tʋaːlf',
            'twee': 'tʋeː',
            '2': 'tʋeː',
            'tussen': 'ˈtʏsən',
            'tuwiet': 'tyˈʋit',
            'uit': 'œyt',
            'uittrekken': 'ˈœytrɛkən',    
            'uitzuigen': 'ˈœytˌzœyɣən',
            'vak': 'vɑk',
            'van': 'vɑn',
            'veel': 'veːl',
            'verdiend': 'vərˈdint',
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
            'wanneer': 'ʋɑˈneːr',
            'waren': 'ˈʋaːrən',
            'warm': 'ʋɑrm',
            'was': 'ʋɑs',
            'wat': 'ʋɑt',
            'wegpakte': 'ˈʋɛxpɑktə',
            'wel': 'ʋɛl',
            'werd': 'ʋɛrt',
            'woorden': 'wordən',
            'zandbak': 'ˈzɑnbɑk',  # final devoicing of d 
            'zanddak': 'ˈzɑndɑk',  # gemination typically results in single consonant
            'zich': 'zɪx',
            'zijn': 'zɛin',
            'ze': 'zə',
            '6': 'zɛs',
            'zei': 'zɛi',
            'zeven': 'ˈzeːvən',
            '7': 'ˈzeːvən',
            'zevenduizend': 'ˌzeːvənˈdœyzənt',
            'zo': 'zoː',
            'zonlicht': 'ˈzɔnlɪxt',
            'zou': 'zɑu',  # not 'zʏlə(n)' which would be 'zullen'
            'Één bol vanille en één bol stracciatella': 'eːn bɔl vaˈnɪlə ɛn eːn bɔl ˌstrɑtʃaˈtɛla',
            '"Welterusten" zei de moeder tegen haar kinderen.': 'ˈʋɛltəˌrʏstən zɛi də ˈmudər ˈteːɣən haːr ˈkɪndərən',
            'Als het kalf verdronken is, dempt men de put.': 'ɑls hət kɑlf vərˈdrɔŋkən ɪs dɛmpt mɛn də pʏt',
            'Brisbane, Melbourne en Perth zijn steden in Australië.': 'ˈbrɪzbən ˈmɛlbərn ɛn pɛrt zɛin ˈsteːdən ɪn ɑuˈstraːlijə',
            'De vakantie was voorbij, de kinderen gingen weer naar school.': 'də vaˈkɑnsi ʋɑs voːrˈbɛi də ˈkɪndərən ˈɣɪŋən ʋeːr naːr sxoːl',
            'Een atoom bestaat uit protonen, elektronen en neutronen.': 'ən aˈtoːm bəˈstaːt œyt proˈtoːnən eˈlɛktroːnən ɛn nøˈtroːnən',
            'Geld heeft niet mijn interesse, macht wel!': 'ɣɛlt heːft nit mɛin ɪntəˈrɛsə mɑxt ʋɛl',
            'Hallo, mijn naam is Brenda.': 'hɑˈloː mɛin naːm ɪs ˈbrɛnda',
            'Het is de D van Daimler, Dacia of Dodge.': 'hət ɪs də deː vɑn ˈdɛimlər ˈdaːsia ɔf dɔdʒ',
            'Het is de L van Land Rover, Lexus of Lotus.': 'hət ɪs də ɛl vɑn lɛnt ˈroːvər ˈlɛksʏs ɔf ˈloːtʏs',
            'Het is wat saai, kan je een stukje doorspoelen?': 'hət ɪs ʋɑt saːi kɑn jə ən ˈstʏkjə ˈdoːrˌspulən',
            'Hij spreekt Frans, Nederlands, Engels en Duits.': 'hɛi spreːkt frɑns ˈneːdərlɑnts ˈɛŋəls ɛn dœyts',
            'Inderdaad, we hadden meer kunnen doen om het te voorkomen.': 'ɪndərˈdaːt ʋə ˈhɑdən meːr ˈkʏnən dun ɔm hət tə voːrˈkoːmən',
            'Ja, we gaan dat programma hervatten.': 'jaː ʋə ɣaːn dɑt proˈɣrɑma hɛrˈvɑtən',
            'Kan je "Ik neem je mee" van Gers Pardoel afspelen?': 'kɑn jə ɪk neːm jə meː vɑn ɣɛrs pɑrˈdul ˈɑfˌspeːlən',
            'Kijk, ze hebben net een Sushi restaurant geopend.': 'kɛik zə ˈhɛbən nɛt ən ˈsuʃi rɛstoˈrɑnt ɣəˈoːpənt',
            'Nee, dat is niet meer het geval.': 'neː dɑt ɪs nit meːr hət ɣəˈvɑl',
            'Niemand nam op, dus ik ben zelf even langs gegaan.': 'ˈnimɑnt nɑm ɔp dʏs ɪk bɛn zɛlf ˈeːvən lɑŋs ɣəˈɣaːn',
            'Oh heerlijk, een salade met feta.': 'oː ˈheːrlək ən saˈlaːdə mɛt ˈfeːta',
            'Pech, we zullen moeten wachten op de volgende bus.': 'pɛx ʋə ˈzʏlən ˈmutən ˈʋɑxtən ɔp də ˈvɔlɣəndə bʏs',
            'Sorry, er zijn geen tafels meer vrij.': 'ˈsɔri ər zɛin ɣeːn ˈtaːfəls meːr vrɛi',
            'Sorry, maar die bewering vind ik toch wel belachelijk.': 'ˈsɔri maːr di bəˈʋeːrɪŋ vɪnt ɪk tɔx ʋɛl bəˈlɑxələk',
            'Spreken is zilver, zwijgen is oud.': 'ˈspreːkən ɪs ˈzɪlvər ˈzʋɛiɣən ɪs ɑut',
            'Wie niet waagt, die niet wint.': 'ʋi nit ʋaːxt di nit ʋɪnt',
            'Zalig, heb je dat zelf voorgesteld?': 'ˈzaːləx hɛp jə dɑt zɛlf voːrɣəˈstɛlt',
            'Ze betaalt niet in gulden, maar in Belgische frank.': 'zə bəˈtaːlt nit ɪn ˈɣʏldən maːr ɪn ˈbɛlɣisə frɑŋk',
            'Zij heeft prachtige, donkergroene ogen.': 'zɛi heːft ˈprɑxtəɣə ˈdɔŋkərˌɣrunə ˈoːɣən',            
            'Aalst staat bekend om de viering van carnaval.': 'aːlst staːt bəˈkɛnt ɔm də ˈviːrɪŋ vɑn ˈkɑrnavɑl',
            'Aan de kassa moet je de barcode scannen.': 'aːn də ˈkɑsa mut jə də ˈbɑrˌkoːdə ˈskɛnən',
            'Aan de overkant van die berg ligt een stadje toch?': 'aːn də ˈoːvərkɑnt vɑn di bɛrx lɪxt ən ˈstɑtjə tɔx',
            'Aardappelen groeien onder de grond.': 'ˈaːrdɑpələn ˈɣruiən ˈɔndər də ɣrɔnt',
            'Achilles is uiteindelijk geveld door een aanval op zijn achillespees.': 'aˈxɪləs ɪs œytˈɛindələk ɣəˈvɛlt doːr ən ˈaːnvɑl ɔp zɛin aˈxɪləsˌpeːs',
            'Acht min vijf is drie.': 'ɑxt mɪn vɛif ɪs dri',
            'Al een paar weken wordt die wijk door inbraken geteisterd.': 'ɑl ən paːr ˈʋeːkən ʋɔrt di ʋɛik doːr ˈɪnbraːkən ɣəˈtɛistərt',
            'Al jarenlang verkleinen de gletsjers overal in de wereld.': 'ɑl ˈjaːrənlɑŋ vərˈklɛinən də ˈɣlɛtsjərs ˈoːvərɑl ɪn də ˈʋeːrəlt',
            'Als de lijst is afgedrukt mag je hem lamineren.': 'ɑls də lɛist ɪs ˈɑfɣədrʏkt mɑx jə hɛm lamiˈneːrən',
            'Als dessert krijgen we tiramisu.': 'ɑls dɛˈsɛrt ˈkrɛiɣən ʋə tiramiˈsu',
            'Als fietser schoor hij zijn beenhaar.': 'ɑls ˈfitsər sxoːr hɛi zɛin ˈbeːnhaːr',
            'Als het groen is mag je doorrijden.': 'ɑls hət ɣrun ɪs mɑx jə ˈdoːrˌrɛidən',
            'Als het morgen meer gaat waaien kunnen we leuk vliegeren.': 'ɑls hət ˈmɔrɣən meːr ɣaːt ˈʋaːiən ˈkʏnən ʋə løːk ˈvliɣərən',
            'Als je iets wil moet je het pakken.': 'ɑls jə its ʋɪl mut jə hət ˈpɑkən',
            'Als je luide muziek beluisterd draag je een hoofdtelefoon.': 'ɑls jə ˈlœydə myˈzik bəˈlœystərt draːx jə ən ˈhoːftˌteːləˌfoːn',
            'Als kind hadden zijn ouders hem leren schaatsen.': 'ɑls kɪnt ˈhɑdən zɛin ˈɑudərs hɛm ˈleːrən ˈsxaːtsən',
            'Als tiener had hij veel last van acne.': 'ɑls ˈtinər hɑt hɛi veːl lɑst vɑn ˈɑkne',
            'Als zij samen praten gaat het vaak over vrouwen.': 'ɑls zɛi ˈsaːmən ˈpraːtən ɣaːt hət vaːk ˈoːvər ˈvrɑuʋən',
            'Barack Obama was de vorige Amerikaanse president.': 'baˈrɑk oˈbaːma ʋɑs də ˈvoːrəɣə ameriˈkaːnsə presiˈdɛnt',
            'Ben je wel eens naar een theater geweest?': 'bɛn jə ʋɛl eːns naːr ən teˈaːtər ɣəˈʋeːst',
            'Bepalen welke hardloper won was een kwestie van milliseconden.': 'bəˈpaːlən ˈʋɛlkə ˈhɑrtˌloːpər ʋɔn ʋɑs ən ˈkʋɛsti vɑn miliˈseːkɔndən',
            'Bewust belasting ontduiken is niet toegestaan.': 'bəˈʋʏst bəˈlɑstɪŋ ɔntˈdœykən ɪs nit tuɣəˈstaːn',
            'Bij het ontbijt werden ook croissants geserveerd.': 'bɛi hət ˈɔntbɛit ˈʋɛrdən oːk krʋaˈsɑnts ɣəsɛrˈveːrt',
            'Bij het ziekenhuis kwam ik een heel aardige verpleger tegen.': 'bɛi hət ˈzikənˌhœys kʋɑm ɪk ən heːl ˈaːrdəɣə vərˈpleːɣər ˈteːɣən',
            'Bijen kunnen sterven van bepaalde pesticiden.': 'ˈbɛiən ˈkʏnən ˈstɛrvən vɑn bəˈpaːldə pɛstiˈsidən',
            'Breakdancen is nu weer in de mode bij jongeren.': 'breːkˈdɑnsən ɪs ny ʋeːr ɪn də ˈmoːdə bɛi ˈjɔŋərən',
            'Breng zeker je zwembroek of bikini mee.': 'brɛŋ ˈzeːkər jə ˈzʋɛmbruk ɔf biˈkini meː',
            'Caïro is de hoofdstad van Egypte.': 'kaˈiːro ɪs də ˈhoːftstɑt vɑn eˈɣɪptə',
            'Charles Darwin reisde de wereld rond als bioloog.': 'tʃɑrls ˈdɑrʋɪn ˈrɛizdə də ˈʋeːrəlt rɔnt ɑls bioˈloːx',
            'Crème brûlée is mijn favoriete dessert.': 'krɛm bryˈleː ɪs mɛin favoˈritə dɛˈsɛrt',
            'Dat boek bevat het volledige werk van die dichter.': 'dɑt buk bəˈvɑt hət vɔˈleːdəɣə ʋɛrk vɑn di ˈdɪxtər',
            'Dat boek over kwantummechanica leest toch niet zo vlot.': 'dɑt buk ˈoːvər kʋɑntʏmmeˈxaːnika leːst tɔx nit zoː vlɔt',
            'Dat brengt ons dichter bij elkaar.': 'dɑt brɛŋt ɔns ˈdɪxtər bɛi ɛlˈkaːr',
            'Dat ga ik morgen benadrukken.': 'dɑt ɣaː ɪk ˈmɔrɣən bəˈnaːdrʏkən',
            'Dat is de beste chocolaterie van België.': 'dɑt ɪs də ˈbɛstə ʃokolaːtəˈri vɑn ˈbɛlɣijə',
            'Dat is een fikse boete.': 'dɑt ɪs ən ˈfɪksə ˈbutə',
            'Dat is een goed nummer.': 'dɑt ɪs ən ɣut ˈnʏmər',
            'Dat is een ingenieus systeem.': 'dɑt ɪs ən ɪnʒeˈnjøːs sisˈteːm',
            'Dat is een retorische vraag.': 'dɑt ɪs ən reˈtoːrisə vraːx',
            'Dat is het neusje van de zalm.': 'dɑt ɪs hət ˈnøːsjə vɑn də zɑlm',
            'Dat is maar een flauw mopje.': 'dɑt ɪs maːr ən flɑu ˈmɔpjə',
            'Dat is slechts een klein akkefietje.': 'dɑt ɪs slɛxts ən klɛin ɑkəˈfitjə',
            'Dat kadert in een lopend onderzoek.': 'dɑt ˈkaːdərt ɪn ən ˈloːpənt ˈɔndərˌzuk',
            'Dat kan je aanpassen in je instellingen.': 'dɑt kɑn jə ˈaːnpɑsən ɪn jə ɪnˈstɛlɪŋən',
            'Dat kan je wel op je buik schrijven!': 'dɑt kɑn jə ʋɛl ɔp jə bœyk ˈsxrɛivən',
            'Dat prachtige landhuis staat eindelijk te koop.': 'dɑt ˈprɑxtəɣə ˈlɑnthœys staːt ˈɛindələk tə koːp',
            'Dat was een harde noot om te kraken.': 'dɑt ʋɑs ən ˈhɑrdə noːt ɔm tə ˈkraːkən',
            'Dat was een nobele daad.': 'dɑt ʋɑs ən noˈbeːlə daːt',
            'De Beatles waren van Liverpool.': 'də ˈbitəls ˈʋaːrən vɑn ˈlɪvərpul',
            'De Christelijke Bijbel is vaak veranderd in de geschiedenis.': 'də ˈxrɪstələkə ˈbɛibəl ɪs vaːk vərˈɑndərt ɪn də ɣəˈsxidənɪs',
            'De Eiffeltoren is het hoogste gebouw in Parijs.': 'də ˈɛifəlˌtoːrən ɪs hət ˈhoːxstə ɣəˈbɑu ɪn paˈrɛis',
            'De Houtstraat ligt twee straten verder.': 'də ˈhɑutstraːt lɪxt tʋeː ˈstraːtən ˈvɛrdər',
            'De Mont Blanc is de hoogste berg in de Alpen.': 'də mɔn blɑŋ ɪs də ˈhoːxstə bɛrx ɪn də ˈɑlpən',
            'De Thaise keuken kan heel lekker zijn.': 'də ˈtaːisə ˈkøːkən kɑn heːl ˈlɛkər zɛin',
            'De Thalys rijdt frequent tussen Parijs en Brussel.': 'də taˈlis rɛit freˈkʋɛnt ˈtʏsən paˈrɛis ɛn ˈbrʏsəl',
            'De Titanic is gezonken in negentienhonderdentwaalf.': 'də tiˈtaːnɪk ɪs ɣəˈzɔŋkən ɪn ˌneːɣənˈtinˌhɔndərtənˈtʋaːlf',
            'De adelaar cirkelde hoog boven onze hoofden.': 'də ˈaːdəlaːr ˈsɪrkəldə hoːx ˈboːvən ˈɔnzə ˈhoːfdən',
            'De ambassadeur werd opgeroepen om uitleg te geven.': 'də ɑmbasaˈdøːr ʋɛrt ˈɔpɣəˌrupən ɔm ˈœytlɛx tə ˈɣeːvən',
            'De ambulance kwam snel om hulp te verlenen.': 'də ɑmbyˈlɑnsə kʋɑm snɛl ɔm hʏlp tə vərˈleːnən',
            'De arbeider was bezig met het afvegen van de buis.': 'də ˈɑrbɛidər ʋɑs ˈbeːzəx mɛt hət ˈɑfˌveːɣən vɑn də bœys',
            'De architect maakte de ontwerpen eerst met potlood.': 'də ɑrxiˈtɛkt ˈmaːktə də ɔntˈʋɛrpən eːrst mɛt ˈpɔtloːt',
            'De baby was inmiddels 5 maanden oud.': 'də ˈbeːbi ʋɑs ɪnˈmɪdəls vɛif ˈmaːndən ɑut',
            'De band kreeg een staande ovatie na hun optreden.': 'də bɛnt kreːx ən ˈstaːndə oˈvaːtsi naː hʏn ˈɔpˌtreːdən',
            'De bassen van mijn nieuwe geluidsinstallatie doen alle ruiten trillen.': 'də ˈbɑsən vɑn mɛin ˈniuʋə ɣəˈlœytsˌɪnstɑˌlaːtsi dun ˈɑlə ˈrœytən ˈtrɪlən',
            'De begrafenisondernemer stelde een eiken doodskist voor.': 'də bəˈɣraːfənɪsˌɔndərˌneːmər ˈstɛldə ən ˈɛikən ˈdoːtskɪst voːr',
            'De benzine is een stuk duurder geworden.': 'də bɛnˈzinə ɪs ən stʏk ˈdyːrdər ɣəˈʋɔrdən',
            'De beul liep naar de galg.': 'də bøːl lip naːr də ɣɑlx',
            'De biefstuk moest op de juiste manier worden gebakken.': 'də ˈbifstʏk must ɔp də ˈjœystə maˈnir ˈʋɔrdən ɣəˈbɑkən',
            'De bliksemschicht was te zien aan de horizon.': 'də ˈblɪksəmˌsxɪxt ʋɑs tə zin aːn də horiˈzɔn',
            'De boer is gespecialiseerd in veeteelt.': 'də bur ɪs ɣəspeːsjaliˈzeːrt ɪn ˈveːˌteːlt',
            'De boer ploegt zijn veld om met een tractor.': 'də bur pluxt zɛin vɛlt ɔm mɛt ən ˈtrɑktɔr',
            'De bokser lag knock-out op de mat.': 'də ˈbɔksər lɑx nɔkˈɑut ɔp də mɑt',
            'De burgemeester werd meermalen bedreigd.': 'də ˈbʏrɣəˌmeːstər ʋɛrt ˈmeːrmaːlən bəˈdrɛixt',
            'De clown had een grote rode neus.': 'də klɑun hɑt ən ˈɣroːtə ˈroːdə nøːs',
            'De conducteur passeerde om de kaartjes te controleren.': 'də kɔndʏkˈtøːr pɑˈseːrdə ɔm də ˈkaːrtjəs tə kɔntroˈleːrən',
            'De criminelen treiterden de politieagenten.': 'də krimiˈneːlən ˈtrɛitərdən də poˈlitsiˌaːɣɛntən',
            'De databases hadden elk een eigen back-up.': 'də ˈdaːtaˌbeːsəs ˈhɑdən ɛlk ən ˈɛiɣən bɛkˈʏp',
            'De databestanden waren per ongeluk gewist door de beheerder.': 'də ˈdaːtabəˌstɑndən ˈʋaːrən pɛr ˈɔnɣəlʏk ɣəˈʋɪst doːr də bəˈheːrdər',
            'De dief klom langs de regenpijp naar boven.': 'də dif klɔm lɑŋs də ˈreːɣənˌpɛip naːr ˈboːvən',
            'De discussie duurde langer dan gedacht.': 'də dɪsˈkʏsi ˈdyːrdə ˈlɑŋər dɑn ɣəˈdɑxt',
            'De douaneagent had hem opgepakt.': 'də duˈaːnəˌaːɣɛnt hɑt hɛm ˈɔpɣəˌpɑkt',
            'De draadloze boor van Jelle is kapot gegaan.': 'də ˈdraːtˌloːzə boːr vɑn ˈjɛlə ɪs kaˈpɔt ɣəˈɣaːn',
            'De droge lucht zorgt voor veel statische elektriciteit.': 'də ˈdroːɣə lʏxt zɔrxt voːr veːl ˈstaːtisə eːlɛktriˈsitɛit',
            'De economie zal zich geleidelijk weer herstellen.': 'də eːkonoˈmi zɑl zɪx ɣəˈlɛidələk ʋeːr hɛrˈstɛlən',
            'De een zijn dood is de ander zijn brood.': 'də eːn zɛin doːt ɪs də ˈɑndər zɛin broːt',
            'De frisdrank was vooral populair bij jongeren.': 'də ˈfrɪsdrɑŋk ʋɑs voːrˈɑl poˈpyleːr bɛi ˈjɔŋərən',
            'De ganzen vliegen naar het noorden.': 'də ˈɣɑnzən ˈvliɣən naːr hət ˈnoːrdən',
            'De geest lachte op een vreemde manier.': 'də ɣeːst ˈlɑxtə ɔp ən ˈvreːmdə maˈnir',
            'De gefaalde acquisitie had twijfel gezaaid over het management.': 'də ɣəˈfaːldə ɑkʋiˈzitsi hɑt ˈtʋɛifəl ɣəˈzaːit ˈoːvər hət ˈmɛnətʃmənt',
            'De gemiddelde vrouw in België heeft één komma vierenzeventig kinderen.': 'də ɣəˈmɪdəldə vrɑu ɪn ˈbɛlɣijə heːft eːn ˈkɔma ˌvirənˈzeːvəntəx ˈkɪndərən',
            'De generaal had de leiding overgenomen.': 'də ɣeneˈraːl hɑt də ˈlɛidɪŋ ˈoːvərɣəˌnoːmən',
            'De generator had de geest gegeven.': 'də ɣeneˈraːtɔr hɑt də ɣeːst ɣəˈɣeːvən',
            'De gereserveerde plaatsen kan je vinden in wagon vijf.': 'də ɣəreːsɛrˈveːrdə ˈplaːtsən kɑn jə ˈvɪndən ɪn ˈʋaːɣɔn vɛif',
            'De geschiedenis van de Egyptenaren is enorm interessant.': 'də ɣəˈsxidənɪs vɑn də eˈɣɪptəˌnaːrən ɪs eˈnɔrm ɪntəreˈsɑnt',
            'De gesuikerde frisdrank plakte aan de vloer.': 'də ɣəˈsœykərdə ˈfrɪsdrɑŋk ˈplɑktə aːn də vlur',
            'De geur van de versgebakken wafels deed mij watertanden.': 'də ɣøːr vɑn də ˈvɛrsɣəˌbɑkən ˈʋaːfəls deːt mɛi ˈʋaːtərˌtɑndən',
            'De graanoogst was lager dan verwacht dit jaar.': 'də ˈɣraːnˌoːxst ʋɑs ˈlaːɣər dɑn vərˈʋɑxt dɪt jaːr',
            'De groeicijfers gaan in een stijgende lijn.': 'də ˈɣruiˌsɛifərs ɣaːn ɪn ən ˈstɛiɣəndə lɛin',
            'De haaien hebben de zeehond opgegeten.': 'də ˈhaːiən ˈhɛbən də ˈzeːhɔnt ˈɔpɣəˌeːtən',
            'De honden blaffen in de verte.': 'də ˈhɔndən ˈblɑfən ɪn də ˈvɛrtə',
            'De hondenmand staat in de hoek van de kamer.': 'də ˈhɔndənˌmɑnt staːt ɪn də huk vɑn də ˈkaːmər',
            'De houthakker kapt de boom om met een bijl.': 'də ˈhɑutˌhɑkər kɑpt də boːm ɔm mɛt ən bɛil',
            'De jager was op zoek naar roodkapje.': 'də ˈjaːɣər ʋɑs ɔp zuk naːr ˈroːtˌkɑpjə',
            'De jongen had zwarte krullen.': 'də ˈjɔŋən hɑt ˈzʋɑrtə ˈkrʏlən',
            'De jurk was gemaakt van zijde.': 'də jʏrk ʋɑs ɣəˈmaːkt vɑn ˈzɛidə',
            'De kast staat voor de deur.': 'də kɑst staːt voːr də døːr',
            'De kastanjeboom was al bijna honderd jaar oud.': 'də kɑsˈtɑnjəˌboːm ʋɑs ɑl ˈbɛinaː ˈhɔndərt jaːr ɑut',
            'De kat sprong in het gordijn.': 'də kɑt sprɔŋ ɪn hət ɣɔrˈdɛin',
            'De kinderen zaten allemaal in kleermakerszit op de grond.': 'də ˈkɪndərən ˈzaːtən ˈɑləmaːl ɪn ˈkleːrmaːkərsˌzɪt ɔp də ɣrɔnt',
            'De kip at de regenworm op.': 'də kɪp ɑt də ˈreːɣənˌʋɔrm ɔp',
            'De klinkers van de straat werden vervangen door asfalt.': 'də ˈklɪŋkərs vɑn də straːt ˈʋɛrdən vərˈvɑŋən doːr ɑsˈfɑlt',
            'De koers is snel gedaald.': 'də kurs ɪs snɛl ɣəˈdaːlt',
            'De kroonjuwelen van de koningin zijn gestolen.': 'də ˈkroːnˌjuˌʋeːlən vɑn də koˈnɪŋɪn zɛin ɣəˈstoːlən',
            'De kussens in dat hotel waren niet dik genoeg.': 'də ˈkʏsəns ɪn dɑt hoˈtɛl ˈʋaːrən nit dɪk ɣəˈnux',
            'De leverancier komt langs om kwart over zes.': 'də leːvərɑnˈsir kɔmt lɑŋs ɔm kʋɑrt ˈoːvər zɛs',
            'De lucht ziet helemaal blauw.': 'də lʏxt zit ˈheːləmaːl blɑu',
            'De luier zat opnieuw vol met uitwerpselen.': 'də ˈlœyər zɑt ɔpˈniu vɔl mɛt ˈœytʋɛrpsələn',
            'De man stond zonder schaamte te loeren naar de vrouw.': 'də mɑn stɔnt ˈzɔndər ˈsxaːmtə tə ˈlurən naːr də vrɑu',
            'De meeste mensen stofzuigen hun tapijt niet iedere dag.': 'də ˈmeːstə ˈmɛnsən ˈstɔfˌzœyɣən hʏn taˈpɛit nit ˈidərə dɑx',
            'De meningen over dat onderwerp zijn niet onverdeeld.': 'də ˈmeːnɪŋən ˈoːvər dɑt ˈɔndərˌʋɛrp zɛin nit ɔnvərˈdeːlt',
            'De mol groef tunnels in het mulle zand.': 'də mɔl ɣruf ˈtʏnəls ɪn hət ˈmʏlə zɑnt',
            'De motor bromde zachtjes op de achtergrond.': 'də ˈmoːtɔr ˈbrɔmdə ˈzɑxtjəs ɔp də ˈɑxtərˌɣrɔnt',
            'De muntstukken waren op in die automaat.': 'də ˈmʏntstʏkən ˈʋaːrən ɔp ɪn di ɑutoˈmaːt',
            'De olifant was even bang voor de muis.': 'də ˈoːlifɑnt ʋɑs ˈeːvən bɑŋ voːr də mœys',
            'De opbouw van het nummer kan beter.': 'də ˈɔpbɑu vɑn hət ˈnʏmər kɑn ˈbeːtər',
            'De oranje gloed van de straatverlichting verlichtte de kamer.': 'də oˈrɑnjə ɣlut vɑn də ˈstraːtvərˌlɪxtɪŋ vərˈlɪxtə də ˈkaːmər',
            'De organisatie van het evenement is chaotisch verlopen.': 'də ɔrɣaniˈzaːtsi vɑn hət eːvəneˈmɛnt ɪs xaˈoːtis vərˈloːpən',
            'De paashaas wordt steeds populairder tijdens Pasen.': 'də ˈpaːshaːs ʋɔrt steːts popyleːrdər ˈtɛidəns ˈpaːsən',
            'De peuter stapelde de blokken op elkaar.': 'də ˈpøːtər ˈstaːpəldə də ˈblɔkən ɔp ɛlˈkaːr',
            'De plant van de aardappel is giftig.': 'də plɑnt vɑn də ˈaːrdɑpəl ɪs ˈɣɪftəx',
            'De politie heeft een dossier geopend over die onrustwekkende verdwijning.': 'də poˈlitsi heːft ən dɔˈsir ɣəˈoːpənt ˈoːvər di ˈɔnrʏstˌʋɛkəndə vərˈdʋɛinɪŋ',
            'De politieagent volgde de wet tot op de letter.': 'də poˈlitsiˌaːɣɛnt ˈvɔlxdə də ʋɛt tɔt ɔp də ˈlɛtər',
            'De prijs is gebaseerd op vraag en aanbod.': 'də prɛis ɪs ɣəbaˈzeːrt ɔp vraːx ɛn ˈaːnbɔt',
            'De productie van olie is al jaren aan het dalen.': 'də proˈdʏktsi vɑn ˈoːli ɪs ɑl ˈjaːrən aːn hət ˈdaːlən',
            'De rechter heeft het vonnis uitgesproken.': 'də ˈrɛxtər heːft hət ˈvɔnɪs ˈœytɣəˌsproːkən',
            'De regisseur schreeuwde actie om de scene te starten.': 'də reʒiˈsøːr ˈsxreːudə ˈɑktsi ɔm də seːn tə ˈstɑrtən',
            'De reporter bracht live verslag van het evenement.': 'də reˈpɔrtər brɑxt laif vərˈslɑx vɑn hət eːvəneˈmɛnt',
            'De rijpe pruimen vielen van de boom.': 'də ˈrɛipə ˈprœymən ˈvilən vɑn də boːm',
            'De rijstvelden lagen er prachtig bij.': 'də ˈrɛistˌvɛldən ˈlaːɣən ər ˈprɑxtəx bɛi',
            'De rivier was niet geschikt om in te zwemmen.': 'də riˈvir ʋɑs nit ɣəˈsxɪkt ɔm ɪn tə ˈzʋɛmən',
            'De rolluiken zijn nog dicht.': 'də ˈrɔlˌœykən zɛin nɔx dɪxt',
            'De ruitenwasser hing aan de buitenkant van het flatgebouw.': 'də ˈrœytənˌʋɑsər hɪŋ aːn də ˈbœytənkɑnt vɑn hət ˈflɛtɣəˌbɑu',
            'De schrijver van dat boek is internationaal bekend.': 'də ˈsxrɛivər vɑn dɑt buk ɪs ɪntərnatsiˈonaːl bəˈkɛnt',
            'De schuifdeur zit geblokkeerd en gaat niet meer dicht.': 'də ˈsxœyfdøːr zɪt ɣəblɔˈkeːrt ɛn ɣaːt nit meːr dɪxt',
            'De schutter kon ongezien wegkomen.': 'də ˈsxʏtər kɔn ˈɔnɣəˌzin ˈʋɛxˌkoːmən',
            'De schuur stond helemaal vol met spullen.': 'də sxyːr stɔnt ˈheːləmaːl vɔl mɛt ˈspʏlən',
            'De slaapzakken liggen ook al klaar.': 'də ˈslaːpˌzɑkən ˈlɪɣən oːk ɑl klaːr',
            'De slager gaf het kind een stukje worst.': 'də ˈslaːɣər ɣɑf hət kɪnt ən ˈstʏkjə ʋɔrst',
            'De stewardess vraagt aandacht voor de noodprocedure.': 'də stjuˈɑrdɛs vraːxt ˈaːndɑxt voːr də ˈnoːtproˌseːˌdyːrə',
            'De tabaksindustrie is toch haar flair verloren.': 'də taˈbɑksˌɪndʏsˌtri ɪs tɔx haːr flɛːr vərˈloːrən',
            'De televisie hangt aan de muur met een beugel.': 'də teːləˈvizi hɑŋt aːn də myːr mɛt ən ˈbøːɣəl',
            'De tent was nochtans goed verankerd.': 'də tɛnt ʋɑs ˈnɔxtɑns ɣut vərˈɑŋkərt',
            'De topsnelheid is meer dan tweehonderd kilometer per uur.': 'də ˈtɔpsnɛlˌhɛit ɪs meːr dɑn ˌtʋeːˈhɔndərt kiloˈmeːtər pɛr yːr',
            'De trams rijden op een afgescheiden baan.': 'də trɛms ˈrɛidən ɔp ən ˈɑfɣəˌsxɛidən baːn',
            'De trap van de Domtoren heeft vierhonderdvijfenzestig treden.': 'də trɑp vɑn də ˈdɔmˌtoːrən heːft ˌvirˈhɔndərtˌvɛifənˈzɛstəx ˈtreːdən',
            'De trein van de NMBS had weer vertraging.': 'də trɛin vɑn də ɛnɛmbeːˈɛs hɑt ʋeːr vərˈtraːɣɪŋ',
            'De treinrit richting Bern was genieten.': 'də ˈtrɛinrɪt ˈrɪxtɪŋ bɛrn ʋɑs ɣəˈnitən',
            'De veelpleger bleek een pleegkind te zijn.': 'də ˈveːlˌpleːɣər bleːk ən ˈpleːxkɪnt tə zɛin',
            'De ventilator bromde in de hoek van de kamer.': 'də vɛntiˈlaːtɔr ˈbrɔmdə ɪn də huk vɑn də ˈkaːmər',
            'De verkoper was net als de rest niet betrouwbaar.': 'də vərˈkoːpər ʋɑs nɛt ɑls də rɛst nit bəˈtrɑubaːr',
            'De villa stond op het einde van de straat.': 'də ˈvila stɔnt ɔp hət ˈɛində vɑn də straːt',
            'De vissersboot had zich vastgevaren in het ondiepe water.': 'də ˈvɪsərsˌboːt hɑt zɪx ˈvɑstɣəˌvaːrən ɪn hət ˈɔndipə ˈʋaːtər',
            'De vlag wapperde in de wind.': 'də vlɑx ˈʋɑpərdə ɪn də ʋɪnt',
            'De vleermuis vloog door de donkere grot.': 'də ˈvleːrmœys vloːx doːr də ˈdɔŋkərə ɣrɔt',
            'De volgende halte is over zestien minuten.': 'də ˈvɔlɣəndə ˈhɑltə ɪs ˈoːvər ˈzɛstin miˈnytən',
            'De weerwolf is enkel actief als het volle maan is.': 'də ˈʋeːrʋɔlf ɪs ˈɛŋkəl ɑkˈtif ɑls hət ˈvɔlə maːn ɪs',
            'De wegversmalling zorgde voor een grote opstopping in het verkeer.': 'də ˈʋɛxvərˌsmɑlɪŋ ˈzɔrxdə voːr ən ˈɣroːtə ˈɔpˌstɔpɪŋ ɪn hət vərˈkeːr',
            'De wieken van de molen stonden stil.': 'də ˈʋikən vɑn də ˈmoːlən ˈstɔndən stɪl',
            'De winkel was vierentwintig op zeven geopend.': 'də ˈʋɪŋkəl ʋɑs ˌvirənˈtʋɪntəx ɔp ˈzeːvən ɣəˈoːpənt',
            'De wond heelde sneller dan iedereen verwachtte.': 'də ʋɔnt ˈheːldə ˈsnɛlər dɑn ˈidəreːn vərˈʋɑxtə',
            'De zon is niet de enige ster in het universum.': 'də zɔn ɪs nit də ˈeːnəɣə stɛr ɪn hət yniˈvɛrsʏm',
            'De zuurstof was ontsnapt uit de buis.': 'də ˈzyːrstɔf ʋɑs ɔntˈsnɑpt œyt də bœys',
            'De zware tocht had zijn tol geëist.': 'də ˈzʋaːrə tɔxt hɑt zɛin tɔl ɣəˈɛist',
            'Deze website kan wel wat meer interactie gebruiken.': 'ˈdeːzə ˈʋɛpsɑit kɑn ʋɛl ʋɑt meːr ɪntərˈɑktsi ɣəˈbrœykən',
            'Deze wollen trui is kriebelig.': 'ˈdeːzə ˈʋɔlən trœy ɪs ˈkribələx',
            'Die doos bevat honderd verschillende kleurpotloden.': 'di doːs bəˈvɑt ˈhɔndərt vərˈsxɪləndə ˈkløːrˌpɔtˌloːdən',
            'Die inspanning heeft mij volledig afgemat.': 'di ˈɪnspɑnɪŋ heːft mɛi vɔˈleːdəx ˈɑfɣəˌmɑt',
            'Die inzichten zijn al achterhaald.': 'di ˈɪnzɪxtən zɛin ɑl ˈɑxtərˌhaːlt',
            'Discriminatie is nog steeds een groot probleem.': 'dɪskrimiˈnaːtsi ɪs nɔx steːts ən ɣroːt proˈbleːm',
            'Dolfijnen communiceren en navigeren met behulp van ultrasoon geluid.': 'dɔlˈfɛinən kɔmyniˈkeːrən ɛn naviˈɣeːrən mɛt bəˈhʏlp vɑn ʏltraˈsoːn ɣəˈlœyt',
            'Donald Trump is de huidige Amerikaanse president.': 'ˈdɔnɑlt trʏmp ɪs də ˈhœydəɣə ameriˈkaːnsə presiˈdɛnt',
            'Door de afbeelding te downloaden kon Bas het bewerken.': 'doːr də ˈɑfˌbeːldɪŋ tə ˈdɑunˌloːdən kɔn bɑs hət bəˈʋɛrkən',
            'Door de globalisering kan ik nu iets in China bestellen.': 'doːr də ɣloːbaliˈzeːrɪŋ kɑn ɪk ny its ɪn ˈʃina bəˈstɛlən',
            'Door de keelpijn deed slikken veel pijn.': 'doːr də ˈkeːlpɛin deːt ˈslɪkən veːl pɛin',
            'Door een gebrek aan bewijs is de zaak geseponeerd.': 'doːr ən ɣəˈbrɛk aːn bəˈʋɛis ɪs də zaːk ɣəsepoˈneːrt',
            'Door een probleem met het internet ben ik technisch werkloos.': 'doːr ən proˈbleːm mɛt hət ˈɪntərnɛt bɛn ɪk ˈtɛxnis ˈʋɛrkloːs',
            'Door het werkverkeer was er geluidsoverlast ontstaan.': 'doːr hət ˈʋɛrkvərˌkeːr ʋɑs ər ɣəˈlœytsˌoːvərˌlɑst ɔntˈstaːn',
            'Door zelf de tomaten te kweken bespaarde Amber veel geld.': 'doːr zɛlf də toˈmaːtən tə ˈkʋeːkən bəˈspaːrdə ˈɑmbər veːl ɣɛlt',
            'Doordat we te laat waren moesten we dubbel betalen.': 'doːrˈdɑt ʋə tə laːt ˈʋaːrən ˈmustən ʋə ˈdʏbəl bəˈtaːlən',
            'Drumt Jasper nog steeds in de groep van Daniël?': 'drʏmt ˈjɑspər nɔx steːts ɪn də ɣrup vɑn daˈnijɛl',
            'Een cactus kan je beter niet aanraken!': 'ən ˈkɑktʏs kɑn jə ˈbeːtər nit ˈaːnraːkən',
            'Een concentratiekamp is de hel op aarde.': 'ən kɔnsɛnˈtraːtsiˌkɑmp ɪs də hɛl ɔp ˈaːrdə',
            'Een dag bestaat uit vierentwintig uren.': 'ən dɑx bəˈstaːt œyt ˌvirənˈtʋɪntəx ˈyːrən',
            'Een groot deel van Oostenrijk is eigenlijk vrij vlak.': 'ən ɣroːt deːl vɑn ˈoːstənrɛik ɪs ˈɛiɣənlək vrɛi vlɑk',
            'Een hittegolf is een periode van extreem warme dagen.': 'ən ˈhɪtəˌɣɔlf ɪs ən periˈoːdə vɑn ɛksˈtreːm ˈʋɑrmə ˈdaːɣən',
            'Een houten pilaar stond in het midden.': 'ən ˈhɑutən piˈlaːr stɔnt ɪn hət ˈmɪdən',
            'Een mededelende zin eindigt op een punt.': 'ən ˈmeːdəˌdeːləndə zɪn ˈɛindəxt ɔp ən pʏnt',
            'Een mens kan een paar minuten overleven in een vacuüm.': 'ən mɛns kɑn ən paːr miˈnytən ˈoːvərˌleːvən ɪn ən vaˈkyːʏm',
            'Een van de spelmogelijkheden bij Risk is wereldverovering.': 'eːn vɑn də ˈspɛlˌmoːɣələkˌheːdən bɛi rɪsk ɪs ˈʋeːrəltvərˌoːvərɪŋ',
            'Een van de symptomen van griep is koorts.': 'eːn vɑn də sɪmpˈtoːmən vɑn ɣrip ɪs koːrts',
            'Eisden is een deelgemeente van Maasmechelen.': 'ˈɛizdən ɪs ən ˈdeːlɣəˌmeːntə vɑn ˈmaːsˌmɛxələn',
            'Elke maakt \'s avonds de brievenbus leeg.': 'ˈɛlkə maːkt ˈsaːvənts də ˈbrivənbʏs leːx',
            'Els heeft recent ontslag genomen.': 'ɛls heːft reˈsɛnt ˈɔntslɑx ɣəˈnoːmən',
            'Els is kleiner dan Elke.': 'ɛls ɪs ˈklɛinər dɑn ˈɛlkə',
            'Els neemt iedere ochtend de bus om te gaan werken.': 'ɛls neːmt ˈidərə ˈɔxtənt də bʏs ɔm tə ɣaːn ˈʋɛrkən',
            'Er hangt mist in de vallei.': 'ər hɑŋt mɪst ɪn də vɑˈlɛi',
            'Er is niets aan de hand.': 'ər ɪs nits aːn də hɑnt',
            'Er is over een wet gestemd vanmorgen.': 'ər ɪs ˈoːvər ən ʋɛt ɣəˈstɛmt vɑnˈmɔrɣən',
            'Er komt pus uit de wonde.': 'ər kɔmt pʏs œyt də ˈʋɔndə',
            'Er kwam met regelmaat een man met zijn hond langsgelopen.': 'ər kʋɑm mɛt ˈreːɣəlˌmaːt ən mɑn mɛt zɛin hɔnt ˈlɑŋsɣəˌloːpən',
            'Er lag weer een grote hondendrol op het voetpad.': 'ər lɑx ʋeːr ən ˈɣroːtə ˈhɔndənˌdrɔl ɔp hət ˈvutpɑt',
            'Er liepen enkel zwervers over de straat.': 'ər ˈlipən ˈɛŋkəl ˈzʋɛrvərs ˈoːvər də straːt',
            'Er ligt een meubelzaak op die steenweg.': 'ər lɪxt ən ˈmøːbəlˌzaːk ɔp di ˈsteːnʋɛx',
            'Er mogen nog wat kruiden in de puree.': 'ər ˈmoːɣən nɔx ʋɑt ˈkrœydən ɪn də pyˈreː',
            'Er plakte een kauwgom onder de stoel.': 'ər ˈplɑktə ən ˈkɑuɣɔm ˈɔndər də stul',
            'Er staan nog twee flessen in de wijnkelder.': 'ər staːn nɔx tʋeː ˈflɛsən ɪn də ˈʋɛinˌkɛldər',
            'Er staat een barcode op de achterzijde.': 'ər staːt ən ˈbɑrˌkoːdə ɔp də ˈɑxtərˌzɛidə',
            'Er staat nog melk in de rek in de voorraadkamer.': 'ər staːt nɔx mɛlk ɪn də rɛk ɪn də ˈvoːrˌaːtˌkaːmər',
            'Er stond een kilometer lange file van zuid naar noord.': 'ər stɔnt ən kiloˈmeːtər ˈlɑŋə ˈfilə vɑn zœyt naːr noːrt',
            'Er stond een kilometerslange file van zuid naar noord.': 'ər stɔnt ən kiloˈmeːtərsˌlɑŋə ˈfilə vɑn zœyt naːr noːrt',
            'Er tekende zich een flauwe glimlach af op haar gezicht.': 'ər ˈteːkəndə zɪx ən ˈflɑuʋə ˈɣlɪmlɑx ɑf ɔp haːr ɣəˈzɪxt',
            'Er waren geen gordels voorzien op de achterbank.': 'ər ˈʋaːrən ɣeːn ˈɣɔrdəls voːrˈzin ɔp də ˈɑxtərˌbɑŋk',
            'Er was een probleem met de afvoer van hun bad.': 'ər ʋɑs ən proˈbleːm mɛt də ˈɑfvur vɑn hʏn bɑt',
            'Er was een vermoeden van een zware misdaad.': 'ər ʋɑs ən vərˈmudən vɑn ən ˈzʋaːrə mɪsˈdaːt',
            'Er was grote belangstelling voor de politicus na het schandaal.': 'ər ʋɑs ˈɣroːtə bəˈlɑŋˌstɛlɪŋ voːr də poˈlitikʏs naː hət sxɑnˈdaːl',
            'Er was veel schade na de tropische storm.': 'ər ʋɑs veːl ˈsxaːdə naː də ˈtroːpisə stɔrm',
            'Er zat een dun laagje paneermeel op de schnitzel.': 'ər zɑt ən dʏn ˈlaːxjə paˈneːrmeːl ɔp də ˈʃnɪtsəl',
            'Er zat een vlek op de witte lakens.': 'ər zɑt ən vlɛk ɔp də ˈʋɪtə ˈlaːkəns',
            'Er zaten verstekelingen in het ruim van het schip.': 'ər ˈzaːtən vərˈsteːkəlɪŋən ɪn hət rœym vɑn hət sxɪp',
            'Er zijn vreselijke dingen gebeurd tijdens de kolonisatieperiode.': 'ər zɛin ˈvreːsələkə ˈdɪŋən ɣəˈbøːrt ˈtɛidəns də koːloniˈzaːtsiˌperiˌoːdə',
            'Er zitten vierentwintig uren in een dag.': 'ər ˈzɪtən ˌvirənˈtʋɪntəx ˈyːrən ɪn ən dɑx',
            'Er zwommen nog steeds vissen onder het ijs.': 'ər ˈzʋɔmən nɔx steːts ˈvɪsən ˈɔndər hət ɛis',
            'Even verderop staat er een flitspaal.': 'ˈeːvən vərˈdeːrɔp staːt ər ən ˈflɪtspaːl',
            'Eén van de vier straalmotoren van het vliegtuig was uitgevallen.': 'eːn vɑn də vir ˈstraːlˌmoːˌtoːrən vɑn hət ˈvlixtœyx ʋɑs ˈœytɣəˌvɑlən',
            'Farao\'s werden zowel in piramides als in uitgegraven rotsen begraven.': 'faˈraːoːs ˈʋɛrdən zoˈʋɛl ɪn piraˈmidəs ɑls ɪn ˈœytɣəˌɣraːvən ˈrɔtsən bəˈɣraːvən',
            'Finland grenst in het oosten aan Rusland.': 'ˈfɪnlɑnt ɣrɛnst ɪn hət ˈoːstən aːn ˈrʏslɑnt',
            'Ga je dit jaar naar Pukkelpop?': 'ɣaː jə dɪt jaːr naːr ˈpʏkəlpɔp',
            'Ga je wel eens kamperen met een tent?': 'ɣaː jə ʋɛl eːns kɑmˈpeːrən mɛt ən tɛnt',
            'Gaan jullie ieder jaar met de auto op reis?': 'ɣaːn ˈjʏli ˈidər jaːr mɛt də ˈɑutoː ɔp rɛis',
            'Gaat er een hogesnelheidstrein richting Kyoto?': 'ɣaːt ər ən ˈhoːɣəsnɛlˌhɛitsˌtrɛin ˈrɪxtɪŋ kiˈoːtoː',
            'Gelukkig kon hij het vuur doven met een brandblusser.': 'ɣəˈlʏkəx kɔn hɛi hət vyːr ˈdoːvən mɛt ən ˈbrɑntˌblʏsər',
            'Gelukkig was er niemand gewond geraakt bij dat incident.': 'ɣəˈlʏkəx ʋɑs ər ˈnimɑnt ɣəˈʋɔnt ɣəˈraːkt bɛi dɑt ɪnsiˈdɛnt',
            'Geruisloos beweegt ze zich door de kamer.': 'ɣəˈrœysloːs bəˈʋeːxt zə zɪx doːr də ˈkaːmər',
            'Geënerveerd nam ze nog een trek van haar sigaret.': 'ɣəeːnɛrˈveːrt nɑm zə nɔx ən trɛk vɑn haːr siɣaˈrɛt',
            'Gisteravond is op deze straat een ongeluk gebeurd.': 'ˈɣɪstərˌaːvɔnt ɪs ɔp ˈdeːzə straːt ən ˈɔnɣəlʏk ɣəˈbøːrt',
            'Goede wijn behoeft geen krans.': 'ˈɣudə ʋɛin bəˈhuft ɣeːn krɑns',
            'Goele werkt nu in het buitenland.': 'ˈɣulə ʋɛrkt ny ɪn hət ˈbœytənlɑnt',
            'Haar kinderen hebben een zandkasteel gebouwd op het strand.': 'haːr ˈkɪndərən ˈhɛbən ən ˈzɑntˌkɑsˌteːl ɣəˈbɑut ɔp hət strɑnt',
            'Haar moed moet beloond worden.': 'haːr mut mut bəˈloːnt ˈʋɔrdən',
            'Haar opmerking was de genadeslag.': 'haːr ˈɔpˌmɛrkɪŋ ʋɑs də ɣəˈnaːdəˌslɑx',
            'Haar ouders hebben de geldkraan dichtgedraaid.': 'haːr ˈɑudərs ˈhɛbən də ˈɣɛltkraːn ˈdɪxtɣəˌdraːit',
            'Haar tante is kolonel geworden.': 'haːr ˈtɑntə ɪs koloˈnɛl ɣəˈʋɔrdən',
            'Haar tattoo is nauwelijks zichtbaar.': 'haːr tɑˈtu ɪs ˈnɑuʋələks ˈzɪxtbaːr',
            'Heb je de tafel gedekt?': 'hɛp jə də ˈtaːfəl ɣəˈdɛkt',
            'Heb je een licentie voor die software?': 'hɛp jə ən liˈsɛnsi voːr di ˈsɔftʋɛːr',
            'Heb je een plan van het gebouw?': 'hɛp jə ən plɑn vɑn hət ɣəˈbɑu',
            'Heb jij al eens een escape game gedaan?': 'hɛp jɛi ɑl eːns ən ɛsˈkeːp ɣeːm ɣəˈdaːn',
            'Heb jij nog albums op platen?': 'hɛp jɛi nɔx ˈɑlbʏms ɔp ˈplaːtən',
            'Heb jij nog een papieren rijbewijs?': 'hɛp jɛi nɔx ən paˈpirən ˈrɛibəˌʋɛis',
            'Hebben jullie dit appartement gekocht of gehuurd?': 'ˈhɛbən ˈjʏli dɪt ɑpɑrtəˈmɛnt ɣəˈkɔxt ɔf ɣəˈhyːrt',
            'Heeft Laurens eindelijk zijn rijbewijs gehaald?': 'heːft ˈlɑurəns ˈɛindələk zɛin ˈrɛibəˌʋɛis ɣəˈhaːlt',
            'Heeft Nico eindelijk zijn doctoraat afgemaakt?': 'heːft ˈnikoː ˈɛindələk zɛin dɔktoˈraːt ˈɑfɣəˌmaːkt',
            'Heeft de beveiligingscamera iets geregistreerd?': 'heːft də bəˈvɛiləɣɪŋsˌkaːməra its ɣəreːɣɪsˈtreːrt',
            'Hercule Poirot is een detective.': 'hɛrˈkyl pʋaˈroː ɪs ən deˈtɛktif',
            'Herinner mij eraan om je nog een mailtje te sturen.': 'hɛrˈɪnər mɛi ərˈaːn ɔm jə nɔx ən ˈmeːltjə tə ˈstyːrən',
            'Het I-profiel had de schok geabsorbeerd.': 'hət ˈiˌproˌfil hɑt də sxɔk ɣəɑpsɔrˈbeːrt',
            'Het aantal mensen met obesitas neemt jaar na jaar toe.': 'hət ˈaːntɑl ˈmɛnsən mɛt obeˈzitɑs neːmt jaːr naː jaːr tu',
            'Het aantal politieagenten was buiten proportie.': 'hət ˈaːntɑl poˈlitsiˌaːɣɛntən ʋɑs ˈbœytən proˈpɔrtsi',
            'Het andere team heeft dat doelpunt afgedwongen.': 'hət ˈɑndərə tim heːft dɑt ˈdulpʏnt ˈɑfɣəˌdʋɔŋən',
            'Het bedrijf kon de rentekosten niet meer opbrengen.': 'hət bəˈdrɛif kɔn də ˈrɛntəˌkɔstən nit meːr ˈɔpbrɛŋən',
            'Het beest bleek een beer te zijn.': 'hət beːst bleːk ən beːr tə zɛin',
            'Het bijtend product had de parket permanent beschadigd.': 'hət ˈbɛitənt proˈdʏkt hɑt də pɑrˈkɛt pɛrmaˈnɛnt bəˈsxaːdəxt',
            'Het bos brandde voor dertien dagen.': 'hət bɔs ˈbrɑndə voːr ˈdɛrtin ˈdaːɣən',
            'Het containerschip lag aangemeerd in de haven.': 'hət kɔnˈteːnərˌsxɪp lɑx ˈaːnɣəˌmeːrt ɪn də ˈhaːvən',
            'Het examen bestond uit vijftig meerkeuzevragen.': 'hət ɛkˈsaːmən bəˈstɔnt œyt ˈvɛiftəx ˈmeːrkøːzəˌvraːɣən',
            'Het gemeentehuis kan je vinden op de markt.': 'hət ɣəˈmeːntəˌhœys kɑn jə ˈvɪndən ɔp də mɑrkt',
            'Het heeft tien uur geduurd.': 'hət heːft tin yːr ɣəˈdyːrt',
            'Het hertje verschool zich achter de eenzame boom.': 'hət ˈhɛrtjə vərˈsxoːl zɪx ˈɑxtər də ˈeːnzaːmə boːm',
            'Het hoofdgerecht is een koninginnenhapje.': 'hət ˈhoːftɣəˌrɛxt ɪs ən koˈnɪŋɪnənˌhɑpjə',
            'Het huis stond te huur.': 'hət hœys stɔnt tə hyːr',
            'Het internet lijkt niet meer te werken.': 'hət ˈɪntərnɛt lɛikt nit meːr tə ˈʋɛrkən',
            'Het irrigatie systeem moet binnen een week worden gemaakt.': 'hət iriˈɣaːtsi sisˈteːm mut ˈbɪnən ən ʋeːk ˈʋɔrdən ɣəˈmaːkt',
            'Het is de B van Bugatti.': 'hət ɪs də beː vɑn byˈɣɑti',
            'Het is de E van Eagle.': 'hət ɪs də eː vɑn ˈiɣəl',
            'Het is de K van Kia.': 'hət ɪs də kaː vɑn ˈkia',
            'Het is de Q van Quinten.': 'hət ɪs də ky vɑn ˈkʋɪntən',
            'Het is een kleine gemeenschap en iedereen kent iedereen.': 'hət ɪs ən ˈklɛinə ɣəˈmeːnsxɑp ɛn ˈidəreːn kɛnt ˈidəreːn',
            'Het is moeilijk om skibotten te vinden die goed passen.': 'hət ɪs ˈmuilək ɔm ˈskiˌbɔtən tə ˈvɪndən di ɣut ˈpɑsən',
            'Het is moeilijk om skischoenen te vinden die goed passen.': 'hət ɪs ˈmuilək ɔm ˈskiˌsxunən tə ˈvɪndən di ɣut ˈpɑsən',
            'Het is niet fraai maar het voldoet.': 'hət ɪs nit fraːi maːr hət vɔlˈdut',
            'Het is officieel de grootste ramp uit de geschiedenis.': 'hət ɪs ɔfiˈsjeːl də ˈɣroːtstə rɑmp œyt də ɣəˈsxidənɪs',
            'Het is onze gewoonte om op zaterdag friet te eten.': 'hət ɪs ˈɔnzə ɣəˈʋoːntə ɔm ɔp ˈzaːtərdɑx frit tə ˈeːtən',
            'Het is tien over acht.': 'hət ɪs tin ˈoːvər ɑxt',
            'Het jachtseizoen is weer begonnen.': 'hət ˈjɑxtˌsɛizun ɪs ʋeːr bəˈɣɔnən',
            'Het kost slechts een tientje.': 'hət kɔst slɛxts ən ˈtintjə',
            'Het land is pas in de jaren zestig onafhankelijk geworden.': 'hət lɑnt ɪs pɑs ɪn də ˈjaːrən ˈzɛstəx ˈɔnɑfˌhɑŋkələk ɣəˈʋɔrdən',
            'Het land van de rijzende zon.': 'hət lɑnt vɑn də ˈrɛizəndə zɔn',
            'Het materiaal was ontworpen om waterdicht te zijn.': 'hət matəriˈaːl ʋɑs ɔntˈʋɔrpən ɔm ˈʋaːtərdɪxt tə zɛin',
            'Het meisje had blond haar.': 'hət ˈmɛisjə hɑt blɔnt haːr',
            'Het nummer pi is ongeveer drie punt vier één vijf.': 'hət ˈnʏmər pi ɪs ˈɔnɣəveːr dri pʏnt vir eːn vɛif',
            'Het ontbijt is tussen zeven en tien uur \'s ochtends.': 'hət ˈɔntbɛit ɪs ˈtʏsən ˈzeːvən ɛn tin yːr ˈsɔxtənts',
            'Het paard was te oud geworden om bereden te worden.': 'hət paːrt ʋɑs tə ɑut ɣəˈʋɔrdən ɔm bəˈreːdən tə ˈʋɔrdən',
            'Het politiek akkoord heeft zonet groen licht gekregen.': 'hət poliˈtik ɑˈkɔːrt heːft zoˈnɛt ɣrun lɪxt ɣəˈkreːɣən',
            'Het restaurant ligt op de Oratoriënberg.': 'hət rɛstoˈrɑnt lɪxt ɔp də oraˈtoːrijənˌbɛrx',
            'Het schip was gezonken na een zware storm.': 'hət sxɪp ʋɑs ɣəˈzɔŋkən naː ən ˈzʋaːrə stɔrm',
            'Het snoer zit ergens vast.': 'hət snur zɪt ˈɛrɣəns vɑst',
            'Het stekkerblok was overbelast en veroorzaakte de brand.': 'hət ˈstɛkərˌblɔk ʋɑs ˈoːvərbəˌlɑst ɛn vərˈoːrzaːktə də brɑnt',
            'Het stuk metaal was gedraaid in een draaibank.': 'hət stʏk meˈtaːl ʋɑs ɣəˈdraːit ɪn ən ˈdraːibɑŋk',
            'Het systeemplafond was ingezakt door de waterschade.': 'hət sisˈteːmˌplaˌfɔn ʋɑs ˈɪnɣəˌzɑkt doːr də ˈʋaːtərˌsxaːdə',
            'Het volume staat te laag.': 'hət voˈlymə staːt tə laːx',
            'Het was een chaotisch begin van de week.': 'hət ʋɑs ən xaˈoːtis bəˈɣɪn vɑn də ʋeːk',
            'Het was een leuke vakantie.': 'hət ʋɑs ən ˈløːkə vaˈkɑnsi',
            'Het was een zware nacht.': 'hət ʋɑs ən ˈzʋaːrə nɑxt',
            'Het was muisstil in de kamer.': 'hət ʋɑs ˈmœysstɪl ɪn də ˈkaːmər',
            'Het water is al aan het koken.': 'hət ˈʋaːtər ɪs ɑl aːn hət ˈkoːkən',
            'Het weer is ideaal om terrasjes te doen.': 'hət ʋeːr ɪs ideˈaːl ɔm tɛˈrɑsjəs tə dun',
            'Het ziekenhuis zoekt naar nieuwe verpleegsters.': 'hət ˈzikənˌhœys zukt naːr ˈniuʋə vərˈpleːxstərs',
            'Het zuur van een zuurstok is eigenlijk heel zoet.': 'hət zyːr vɑn ən ˈzyːrstɔk ɪs ˈɛiɣənlək heːl zut',
            'Hier geldt voorrang van rechts.': 'hir ɣɛlt ˈvoːrɑŋ vɑn rɛxts',
            'Hij gaat het morgen uitmaken met zijn vriendin.': 'hɛi ɣaːt hət ˈmɔrɣən ˈœytˌmaːkən mɛt zɛin ˈvrindɪn',
            'Hij had een satijnen kostuum gekocht.': 'hɛi hɑt ən saˈtɛinən kɔsˈtym ɣəˈkɔxt',
            'Hij had hen alles verteld onder hypnose.': 'hɛi hɑt hɛn ˈɑləs vərˈtɛlt ˈɔndər hipˈnoːzə',
            'Hij heeft de velgen van zijn wagen gepoetst.': 'hɛi heːft də ˈvɛlɣən vɑn zɛin ˈʋaːɣən ɣəˈputst',
            'Hij heeft ooit een moord gepleegd.': 'hɛi heːft oːit ən moːrt ɣəˈpleːxt',
            'Hij heeft spierpijn aan zijn buikspieren.': 'hɛi heːft ˈspirpɛin aːn zɛin ˈbœykˌspirən',
            'Hij heeft zijn identiteitskaart verloren.': 'hɛi heːft zɛin idɛntiˈtɛitsˌkaːrt vərˈloːrən',
            'Hij is administratief medewerker bij de stad Brussel.': 'hɛi ɪs ɑtminɪstraˈtif ˈmeːdəˌʋɛrkər bɛi də stɑt ˈbrʏsəl',
            'Hij kreeg last van hoogtevrees op de hangbrug.': 'hɛi kreːx lɑst vɑn ˈhoːxtəˌvreːs ɔp də ˈhɑŋbrʏx',
            'Hij liep het warenhuis binnen en kocht een nieuw laken.': 'hɛi lip hət ˈʋaːrənˌhœys ˈbɪnən ɛn kɔxt ən niu ˈlaːkən',
            'Hij liep op de puntjes van zijn tenen.': 'hɛi lip ɔp də ˈpʏntjəs vɑn zɛin ˈteːnən',
            'Hij ligt met zijn armen gekruist op de zetel.': 'hɛi lɪxt mɛt zɛin ˈɑrmən ɣəˈkrœyst ɔp də ˈzeːtəl',
            'Hij luistert naar muziek op zijn kamer met zijn hoofdtelefoon.': 'hɛi ˈlœystərt naːr myˈzik ɔp zɛin ˈkaːmər mɛt zɛin ˈhoːftˌteːləˌfoːn',
            'Hij maakt kunstwerken met kleurpotloden.': 'hɛi maːkt ˈkʏnstˌʋɛrkən mɛt ˈkløːrˌpɔtˌloːdən',
            'Hij plaatste een DVD in de speler.': 'hɛi ˈplaːtstə ən deːveːˈdeː ɪn də ˈspeːlər',
            'Hij schroefde de laatste schroef in het apparaat.': 'hɛi ˈsxruvdə də ˈlaːtstə sxruf ɪn hət ɑpaˈraːt',
            'Hij strooide een grote hoeveelheid poedersuiker over zijn pannenkoek.': 'hɛi ˈstroːidə ən ˈɣroːtə ˈhuvəlhɛit ˈpudərˌsœykər ˈoːvər zɛin ˈpɑnənˌkuk',
            'Hij was een gedecoreerd militair.': 'hɛi ʋɑs ən ɣədekoˈreːrt miliˈtɛːr',
            'Hij was moe van de lange tocht.': 'hɛi ʋɑs mu vɑn də ˈlɑŋə tɔxt',
            'Hij was onschuldig en is vrijgesproken.': 'hɛi ʋɑs ˈɔnsxʏldəx ɛn ɪs ˈvrɛiɣəˌsproːkən',
            'Hij was vergeten hoe de stelling van Pythagoras werkt.': 'hɛi ʋɑs vərˈɣeːtən hu də ˈstɛlɪŋ vɑn piˈtaːɣoraːs ʋɛrkt',
            'Hij wikkelde de verse vis in een bundel krantenpapier.': 'hɛi ˈʋɪkəldə də ˈvɛrsə vɪs ɪn ən ˈbʏndəl ˈkrɑntənˌpaˌpir',
            'Hij zat niet goed op de kruk.': 'hɛi zɑt nit ɣut ɔp də krʏk',
            'Hij zou zich omdraaien in zijn graf.': 'hɛi zɑu zɪx ˈɔmˌdraːiən ɪn zɛin ɣrɑf',
            'Hoe hoog is de hoogste wolkenkrabber ter wereld?': 'hu hoːx ɪs də ˈhoːxstə ˈʋɔlkənˌkrɑbər tɛr ˈʋeːrəlt',
            'Hoeveel dingen kan je tegelijkertijd in je geheugen houden?': 'huˈveːl ˈdɪŋən kɑn jə ˈteːɣələkərˌtɛit ɪn jə ɣəˈhøːɣən ˈhɑudən',
            'Hoeveel groente eet jij per dag?': 'huˈveːl ˈɣruntə eːt jɛi pɛr dɑx',
            'Hoeveel tegoed heb jij nog?': 'huˈveːl təˈɣut hɛp jɛi nɔx',
            'Hoeveel terawattuur is er afgelopen jaar geconsumeerd in België?': 'huˈveːl ˈteːraˌʋɑtˌyːr ɪs ər ˈɑfɣəˌloːpən jaːr ɣəkɔnsyˈmeːrt ɪn ˈbɛlɣijə',
            'Hoeveel verschillende woorden zou het Nederlands bevatten?': 'huˈveːl vərˈsxɪləndə ˈʋoːrdən zɑu hət ˈneːdərlɑnts bəˈvɑtən',
            'Hoeveel zinnen kan je maken?': 'huˈveːl ˈzɪnən kɑn jə ˈmaːkən',
            'Ijs met stukjes chocolade is het beste dat er is.': 'ɛis mɛt ˈstʏkjəs ʃokoˈlaːdə ɪs hət ˈbɛstə dɑt ər ɪs',
            'Ieder jaar valt de ramadan op een andere datum.': 'ˈidər jaːr vɑlt də rɑmaˈdɑn ɔp ən ˈɑndərə ˈdaːtʏm',
            'Iedere avond maak ik tijd voor mijn hobby\'s.': 'ˈidərə ˈaːvɔnt maːk ɪk tɛit voːr mɛin ˈhɔbis',
            'Iedere zondag maken we een boswandeling.': 'ˈidərə ˈzɔndɑx ˈmaːkən ʋə ən ˈbɔsˌʋɑndəlɪŋ',
            'Ik accepteer die diagnose niet.': 'ɪk ɑksɛpˈteːr di diɑxˈnoːzə nit',
            'Ik ben geld gaan afhalen in de bank.': 'ɪk bɛn ɣɛlt ɣaːn ˈɑfˌhaːlən ɪn də bɑŋk',
            'Ik ben het grootste deel van mei niet thuis.': 'ɪk bɛn hət ˈɣroːtstə deːl vɑn mɛi nit tœys',
            'Ik ben niet blij dat het zo lang duurt.': 'ɪk bɛn nit blɛi dɑt hət zoː lɑŋ dyːrt',
            'Ik ben vergeten een cadeau te kopen.': 'ɪk bɛn vərˈɣeːtən ən kaˈdoː tə ˈkoːpən',
            'Ik ben wat verkouden dus ik ga niet werken.': 'ɪk bɛn ʋɑt vərˈkɑudən dʏs ɪk ɣaː nit ˈʋɛrkən',
            'Ik eet dagelijks een banaan als vieruurtje.': 'ɪk eːt ˈdaːɣələks ən baˈnaːn ɑls ˈvirˌyːrtjə',
            'Ik ga de rand afwerken met silicone.': 'ɪk ɣaː də rɑnt ˈɑfˌʋɛrkən mɛt siliˈkoːnə',
            'Ik ga de tafel afschuren.': 'ɪk ɣaː də ˈtaːfəl ˈɑfˌsxyːrən',
            'Ik ga eens kijken of ik iets zie.': 'ɪk ɣaː eːns ˈkɛikən ɔf ɪk its zi',
            'Ik ga nu naar boven om mijn tanden te poetsen.': 'ɪk ɣaː ny naːr ˈboːvən ɔm mɛin ˈtɑndən tə ˈputsən',
            'Ik geraak er niet aan uit.': 'ɪk ɣəˈraːk ər nit aːn œyt',
            'Ik had vroeger een postzegelverzameling.': 'ɪk hɑt ˈvruɣər ən ˈpɔstˌzeːɣəlvərˌzaːməlɪŋ',
            'Ik heb altijd schrik in dat griezelige bos.': 'ɪk hɛp ˈɑltɛit sxrɪk ɪn dɑt ˈɣrizələɣə bɔs',
            'Ik heb artisanale producten gekocht voor hem.': 'ɪk hɛp ɑrtizaˈnaːlə proˈdʏktən ɣəˈkɔxt voːr hɛm',
            'Ik heb dat spel gekocht op de PlayStation.': 'ɪk hɛp dɑt spɛl ɣəˈkɔxt ɔp də ˈpleːˌsteːʃən',
            'Ik heb de kast beige geverfd.': 'ɪk hɛp də kɑst beːʒ ɣəˈvɛrft',
            'Ik heb de kleren opgeborgen in de kast.': 'ɪk hɛp də ˈkleːrən ˈɔpɣəˌbɔrɣən ɪn də kɑst',
            'Ik heb die tocht afgelegd op stapschoenen.': 'ɪk hɛp di tɔxt ˈɑfɣəˌlɛxt ɔp ˈstɑpˌsxunən',
            'Ik heb die tocht afgelegd op wandelschoenen.': 'ɪk hɛp di tɔxt ˈɑfɣəˌlɛxt ɔp ˈʋɑndəlˌsxunən',
            'Ik heb drie gemiste oproepen.': 'ɪk hɛp dri ɣəˈmɪstə ˈɔpˌrupən',
            'Ik heb een blog over kunst.': 'ɪk hɛp ən blɔx ˈoːvər kʏnst',
            'Ik heb enkele lampen van Philips Hue gekocht.': 'ɪk hɛp ˈɛŋkələ ˈlɑmpən vɑn ˈfilɪps hju ɣəˈkɔxt',
            'Ik heb geen bekerhouders in mijn wagen.': 'ɪk hɛp ɣeːn ˈbeːkərˌhɑudərs ɪn mɛin ˈʋaːɣən',
            'Ik heb geen inspiratie meer.': 'ɪk hɛp ɣeːn ɪnspiˈraːtsi meːr',
            'Ik heb gisteren nieuwe schoenen gekocht in de winkel.': 'ɪk hɛp ˈɣɪstərən ˈniuʋə ˈsxunən ɣəˈkɔxt ɪn də ˈʋɪŋkəl',
            'Ik heb gsm van het merk Nokia.': 'ɪk hɛp ɣeːɛsˈɛm vɑn hət mɛrk ˈnoːkia',
            'Ik heb haar geklopt in de spurt.': 'ɪk hɛp haːr ɣəˈklɔpt ɪn də spʏrt',
            'Ik heb hoofdpijn en een pijnlijke keel.': 'ɪk hɛp ˈhoːftpɛin ɛn ən ˈpɛinləkə keːl',
            'Ik heb mij dat altijd al afgevraagd.': 'ɪk hɛp mɛi dɑt ˈɑltɛit ɑl ˈɑfɣəˌvraːxt',
            'Ik heb mijn enkel verstuikt.': 'ɪk hɛp mɛin ˈɛŋkəl vərˈstœykt',
            'Ik heb mijn knie tegen de tafel gestoten.': 'ɪk hɛp mɛin kni ˈteːɣən də ˈtaːfəl ɣəˈstoːtən',
            'Ik heb nog geen orderbevestiging gekregen in mijn mailbox.': 'ɪk hɛp nɔx ɣeːn ˈɔrdərbəˌvɛstəɣɪŋ ɣəˈkreːɣən ɪn mɛin ˈmeːlbɔks',
            'Ik heb nog nooit in een luchtballon gevlogen.': 'ɪk hɛp nɔx noːit ɪn ən ˈlʏxtbɑˌlɔn ɣəˈvloːɣən',
            'Ik heb twee gaten in mijn sokken.': 'ɪk hɛp tʋeː ˈɣaːtən ɪn mɛin ˈsɔkən',
            'Ik hoop dat deze relatie veel oplevert.': 'ɪk hoːp dɑt ˈdeːzə reˈlaːtsi veːl ˈɔpˌleːvərt',
            'Ik kan helaas niet voldoen aan je eis.': 'ɪk kɑn heˈlaːs nit vɔlˈdun aːn jə ɛis',
            'Ik kan je een Leffe of een Grimbergen aanbieden.': 'ɪk kɑn jə ən ˈlɛfə ɔf ən ˈɣrɪmbɛrɣən ˈaːnˌbidən',
            'Ik kan niet geloven dat het al december is.': 'ɪk kɑn nit ɣəˈloːvən dɑt hət ɑl deˈsɛmbər ɪs',
            'Ik kan niet zwemmen zonder een duikbril.': 'ɪk kɑn nit ˈzʋɛmən ˈzɔndər ən ˈdœykbrɪl',
            'Ik kan vijftig keer pompen.': 'ɪk kɑn ˈvɛiftəx keːr ˈpɔmpən',
            'Ik krijg er kippenvel van.': 'ɪk krɛix ər ˈkɪpənˌvɛl vɑn',
            'Ik lust wel een Duvel.': 'ɪk lʏst ʋɛl ən ˈdyvəl',
            'Ik moet dringend mijn nagels eens knippen.': 'ɪk mut ˈdrɪŋənt mɛin ˈnaːɣəls eːns ˈknɪpən',
            'Ik moet hem nog een paar duizend euro terugbetalen.': 'ɪk mut hɛm nɔx ən paːr ˈdœyzənt ˈøːro təˈrʏxbəˌtaːlən',
            'Ik moet mijn haar nog föhnen.': 'ɪk mut mɛin haːr nɔx ˈføːnən',
            'Ik moet nog boodschappen doen vanavond.': 'ɪk mut nɔx ˈboːtsxɑpən dun vɑnˈaːvɔnt',
            'Ik passeer Utrecht dus ik kan wel even stoppen.': 'ɪk pɑˈseːr ˈytrɛxt dʏs ɪk kɑn ʋɛl ˈeːvən ˈstɔpən',
            'Ik studeer op kot tijdens de examens.': 'ɪk styˈdeːr ɔp kɔt ˈtɛidəns də ɛkˈsaːməns',
            'Ik was volledig verkleumd van de kou.': 'ɪk ʋɑs vɔˈleːdəx vərˈkløːmt vɑn də kɑu',
            'Ik weet niet of we deze misdaad kunnen bewijzen.': 'ɪk ʋeːt nit ɔf ʋə ˈdeːzə mɪsˈdaːt ˈkʏnən bəˈʋɛizən',
            'Ik woon in duizend Brussel.': 'ɪk ʋoːn ɪn ˈdœyzənt ˈbrʏsəl',
            'Ik woon liever op het platteland dan in stedelijk gebied.': 'ɪk ʋoːn ˈlivər ɔp hət ˈplɑtəlɑnt dɑn ɪn ˈsteːdələk ɣəˈbit',
            'Ik zag mijn weerspiegeling in het water.': 'ɪk zɑx mɛin ˈʋeːrˌspiɣəlɪŋ ɪn hət ˈʋaːtər',
            'Ik zal een uitzondering aanvragen.': 'ɪk zɑl ən ˈœytˌzɔndərɪŋ ˈaːnˌvraːɣən',
            'Ik zal het eens opzoeken op Wikipedia.': 'ɪk zɑl hət eːns ˈɔpˌzukən ɔp ʋikiˈpeːdia',
            'Ik zal je helpen met je bagage.': 'ɪk zɑl jə ˈhɛlpən mɛt jə baˈɣaːʒə',
            'Ik zie het patroon niet.': 'ɪk zi hət paˈtroːn nit',
            'Ik zit nog steeds opgezadeld met een overschot aan appels.': 'ɪk zɪt nɔx steːts ˈɔpɣəˌzaːdəlt mɛt ən ˈoːvərsxɔt aːn ˈɑpəls',
            'Ik zoek een aansluiting van het type USB-C.': 'ɪk zuk ən ˈaːnˌslœytɪŋ vɑn hət ˈtipə yɛsbeːˈseː',
            'Ik zoek een synoniem voor portemonnee in mijn kruiswoordraadsel.': 'ɪk zuk ən sinoˈnim voːr pɔrtəmɔˈneː ɪn mɛin ˈkrœysˌʋoːrtˌraːtsəl',
            'Ik zoek het kabeltje van mijn hoofdtelefoon.': 'ɪk zuk hət kaˈbɛltjə vɑn mɛin ˈhoːftˌteːləˌfoːn',
            'Ik zou die opsomming laten inspringen.': 'ɪk zɑu di ˈɔpˌsɔmɪŋ ˈlaːtən ˈɪnˌsprɪŋən',
            'Ik zou het kader toch een paar centimeter hoger hangen.': 'ɪk zɑu hət ˈkaːdər tɔx ən paːr ˈsɛntiˌmeːtər ˈhoːɣər ˈhɑŋən',
            'Ik zwem twee keer per week.': 'ɪk zʋɛm tʋeː keːr pɛr ʋeːk',
            'In Australië leven kangoeroes en koala\'s in het wild.': 'ɪn ɑuˈstraːlijə ˈleːvən kɑŋɣəˈrus ɛn koˈaːlaːs ɪn hət ʋɪlt',
            'In België gebruiken ze een azerty toetsenbord.': 'ɪn ˈbɛlɣijə ɣəˈbrœykən zə ən aˈzɛrti ˈtutsənˌbɔrt',
            'In Canada hebben we ijsberen gezien.': 'ɪn ˈkɑnadaː ˈhɛbən ʋə ˈɛizˌbeːrən ɣəˈzin',
            'In Hilversum worden veel televisieprogramma\'s gemaakt.': 'ɪn ˈhɪlvərsʏm ˈʋɔrdən veːl teːləˈviziˌproˌɣrɑmaːs ɣəˈmaːkt',
            'In dat kanaal is ooit een meisje verdronken.': 'ɪn dɑt kaˈnaːl ɪs oːit ən ˈmɛisjə vərˈdrɔŋkən',
            'In de Verenigde Staten is de wapenlobby enorm machtig.': 'ɪn də vərˈeːnəxdə ˈstaːtən ɪs də ˈʋaːpənˌlɔbi eˈnɔrm ˈmɑxtəx',
            'In de Westerse landen zijn mannen steeds vaker obees.': 'ɪn də ˈʋɛstərsə ˈlɑndən zɛin ˈmɑnən steːts ˈvaːkər oˈbeːs',
            'In de bakkerij rook het naar vers gebakken brood.': 'ɪn də bɑkəˈrɛi roːk hət naːr vɛrs ɣəˈbɑkən broːt',
            'In de dierentuin hebben we een koala gezien.': 'ɪn də ˈdirənˌtœyn ˈhɛbən ʋə ən koˈaːla ɣəˈzin',
            'In de namiddag volgen er opklaringen na de buien.': 'ɪn də ˈnaːˌmɪdɑx ˈvɔlɣən ər ˈɔpˌklaːrɪŋən naː də ˈbœyən',
            'In de vallei heb je nauwelijks bereik.': 'ɪn də vɑˈlɛi hɛp jə ˈnɑuʋələks bəˈrɛik',
            'In de verste verte waren er enkel velden te zien.': 'ɪn də ˈvɛrstə ˈvɛrtə ˈʋaːrən ər ˈɛŋkəl ˈvɛldən tə zin',
            'In de webshop laten ze gerelateerde producten zien.': 'ɪn də ˈʋɛpˌʃɔp ˈlaːtən zə ɣəreːlaˈteːrdə proˈdʏktən zin',
            'In een grootstad moeten verschillende culturen samenleven.': 'ɪn ən ˈɣroːtstɑt ˈmutən vərˈsxɪləndə kʏlˈtyːrən ˈsaːmənˌleːvən',
            'In een loods heeft tweeduizend ton nikkelsulfide vlamgevat.': 'ɪn ən loːts heːft ˌtʋeːˈdœyzənt tɔn ˈnɪkəlsʏlˌfidə ˈvlɑmɣəˌvɑt',
            'In een oogwenk lag ze gezellig te ronken.': 'ɪn ən ˈoːxʋɛŋk lɑx zə ɣəˈzɛləx tə ˈrɔŋkən',
            'In het Verenigd Koninkrijk betalen de mensen met de pond.': 'ɪn hət vərˈeːnəxt ˈkoːnɪŋkrɛik bəˈtaːlən də ˈmɛnsən mɛt də pɔnt',
            'In het buitenland kan je gemakkelijk betalen met een kredietkaart.': 'ɪn hət ˈbœytənlɑnt kɑn jə ɣəˈmɑkələk bəˈtaːlən mɛt ən kreˈditkaːrt',
            'In het graf hadden ze ook enkele relieken gevonden.': 'ɪn hət ɣrɑf ˈhɑdən zə oːk ˈɛŋkələ reˈlikən ɣəˈvɔndən',
            'In het oerwoud ben je enkel op jezelf aangewezen.': 'ɪn hət ˈurʋɑut bɛn jə ˈɛŋkəl ɔp jəˈzɛlf ˈaːnɣəˌʋeːzən',
            'In het ruim staan de machines die de boot voortstuwen.': 'ɪn hət rœym staːn də maˈʃinəs di də boːt ˈvoːrtˌstyʋən',
            'In onze badkamer hebben we handzeep van Sunlight staan.': 'ɪn ˈɔnzə ˈbɑtkaːmər ˈhɛbən ʋə ˈhɑntzeːp vɑn ˈsʏnlɑit staːn',
            'Ine had haar Renault weer verkocht.': 'ˈinə hɑt haːr reˈnoː ʋeːr vərˈkɔxt',
            'Is dat rundvlees of varkensvlees?': 'ɪs dɑt ˈrʏntvleːs ɔf ˈvɑrkənsvleːs',
            'Is de elektricien al lang geweest?': 'ɪs də eːlɛkˈtrisijən ɑl lɑŋ ɣəˈʋeːst',
            'Is de straatnaam hier Bleumerstraat?': 'ɪs də ˈstraːtnaːm hir ˈbløːmərˌstraːt',
            'Is die rode Mercedes van Linda?': 'ɪs di ˈroːdə mɛrˈseːdəs vɑn ˈlɪnda',
            'Is er een geschiedenis van hartziekten in je familie?': 'ɪs ər ən ɣəˈsxidənɪs vɑn ˈhɑrtˌziktən ɪn jə faˈmili',
            'Is het echt waar dat jullie in China gewoond hebben?': 'ɪs hət ɛxt ʋaːr dɑt ˈjʏli ɪn ˈʃina ɣəˈʋoːnt ˈhɛbən',
            'Is vijftig meter de Olympische afstand?': 'ɪs ˈvɛiftəx ˈmeːtər də oˈlɪmpisə ˈɑfstɑnt',
            'Is ze goed in honkbal?': 'ɪs zə ɣut ɪn ˈhɔŋkbɑl',
            'Isabelle is sneller dan Jef.': 'izaˈbɛl ɪs ˈsnɛlər dɑn jɛf',
            'Istanboel is de enige stad die op twee continenten ligt.': 'ɪstɑnˈbul ɪs də ˈeːnəɣə stɑt di ɔp tʋeː kɔntiˈnɛntən lɪxt',
            'Je bent je jasje vergeten.': 'jə bɛnt jə ˈjɑsjə vərˈɣeːtən',
            'Je drijft beter in de Dode Zee.': 'jə drɛift ˈbeːtər ɪn də ˈdoːdə zeː',
            'Je hebt slechts een paar millimeter marge.': 'jə hɛpt slɛxts ən paːr ˈmiliˌmeːtər ˈmɑrʒə',
            'Je hebt toch nog relatief snel extra informatie gevonden.': 'jə hɛpt tɔx nɔx reːlaˈtif snɛl ˈɛkstra ɪnfɔrˈmaːtsi ɣəˈvɔndən',
            'Je kan blikjes kopen in de automaat in het cafetaria.': 'jə kɑn ˈblɪkjəs ˈkoːpən ɪn də ɑutoˈmaːt ɪn hət kafeˈtaːria',
            'Je kan dat per definitie niet op die manier uitwerken.': 'jə kɑn dɑt pɛr defiˈnitsi nit ɔp di maˈnir ˈœytˌʋɛrkən',
            'Je kan flink aankomen als je geblesseerd bent.': 'jə kɑn flɪŋk ˈaːnˌkoːmən ɑls jə ɣəblɛˈseːrt bɛnt',
            'Je kan het werk hervatten.': 'jə kɑn hət ʋɛrk hɛrˈvɑtən',
            'Je kan je computer bedienen met een toetsenbord en muis.': 'jə kɑn jə kɔmˈpjutər bəˈdinən mɛt ən ˈtutsənˌbɔrt ɛn mœys',
            'Je kan wat snacks gaan kopen in de nachtwinkel.': 'jə kɑn ʋɑt snɛks ɣaːn ˈkoːpən ɪn də ˈnɑxtˌʋɪŋkəl',
            'Je mag naar het volgend liedje gaan.': 'jə mɑx naːr hət ˈvɔlɣənt ˈlitjə ɣaːn',
            'Je mag nu overal in de Europese Unie gratis roamen.': 'jə mɑx ny ˈoːvərɑl ɪn də øːroˈpeːsə ˈyni ˈɣraːtɪs ˈroːmən',
            'Je mag roeren in achtjes.': 'jə mɑx ˈrurən ɪn ˈɑxtjəs',
            'Je moet geen angst hebben voor spinnen.': 'jə mut ɣeːn ɑŋst ˈhɛbən voːr ˈspɪnən',
            'Je moet geen schrik hebben van spinnen.': 'jə mut ɣeːn sxrɪk ˈhɛbən vɑn ˈspɪnən',
            'Je moet in achtjes roeren.': 'jə mut ɪn ˈɑxtjəs ˈrurən',
            'Je moet je broer niet zo na-apen.': 'jə mut jə brur nit zoː ˈnaːˌaːpən',
            'Je moet je rug rechter houden.': 'jə mut jə rʏx ˈrɛxtər ˈhɑudən',
            'Je moet opletten als je wilde bessen eet.': 'jə mut ˈɔpˌlɛtən ɑls jə ˈʋɪldə ˈbɛsən eːt',
            'Je moet soms gewoon pragmatisch zijn.': 'jə mut sɔms ɣəˈʋoːn prɑxˈmaːtis zɛin',
            'Je slaat de spijker op zijn kop.': 'jə slaːt də ˈspɛikər ɔp zɛin kɔp',
            'Jullie hond heeft een schattig snoetje.': 'ˈjʏli hɔnt heːft ən ˈsxɑtəx ˈsnutjə',
            'Kan het zijn dat mijn grafische kaart het begeven heeft?': 'kɑn hət zɛin dɑt mɛin ˈɣraːfisə kaːrt hət bəˈɣeːvən heːft',
            'Kan ik een drankje bestellen?': 'kɑn ɪk ən ˈdrɑŋkjə bəˈstɛlən',
            'Kan je dat land aanduiden op een wereldkaart?': 'kɑn jə dɑt lɑnt ˈaːnˌdœydən ɔp ən ˈʋeːrəltˌkaːrt',
            'Kan je dat nogmaals herhalen?': 'kɑn jə dɑt ˈnɔxmaːls hɛrˈhaːlən',
            'Kan je de flyers uitdelen aan de klanten?': 'kɑn jə də ˈflɑiərs ˈœytˌdeːlən aːn də ˈklɑntən',
            'Kan je de oven even voorverwarmen?': 'kɑn jə də ˈoːvən ˈeːvən ˈvoːrvərˌʋɑrmən',
            'Kan je de serie even op pauze zetten?': 'kɑn jə də ˈseːri ˈeːvən ɔp ˈpɑuzə ˈzɛtən',
            'Kan je de temperatuur wat hoger zetten?': 'kɑn jə də tɛmpəraˈtyːr ʋɑt ˈhoːɣər ˈzɛtən',
            'Kan je dertig seconden terugspoelen?': 'kɑn jə ˈdɛrtəx seˈkɔndən təˈrʏxˌspulən',
            'Kan je morgen wat eten meebrengen naar de borrel?': 'kɑn jə ˈmɔrɣən ʋɑt ˈeːtən ˈmeːˌbrɛŋən naːr də ˈbɔrəl',
            'Kan je nog wat maandverband kopen?': 'kɑn jə nɔx ʋɑt ˈmaːntvərˌbɑnt ˈkoːpən',
            'Kan je vanavond meegaan naar het concert?': 'kɑn jə vɑnˈaːvɔnt ˈmeːɣaːn naːr hət kɔnˈsɛrt',
            'Kan jij dat raadsel oplossen?': 'kɑn jɛi dɑt ˈraːtsəl ˈɔpˌlɔsən',
            'Kan jij het raadsel oplossen?': 'kɑn jɛi hət ˈraːtsəl ˈɔpˌlɔsən',
            'Ken je slechts twee akkoorden op de gitaar?': 'kɛn jə slɛxts tʋeː ɑˈkɔːrdən ɔp də ɣiˈtaːr',
            'Ken jij alle hoofdsteden van Europa?': 'kɛn jɛi ˈɑlə ˈhoːftstɑdən vɑn øːˈroːpa',
            'Kijk je soms Flikken op de televisie?': 'kɛik jə sɔms ˈflɪkən ɔp də teːləˈvizi',
            'Klittenband en velcro zijn hetzelfde.': 'ˈklɪtənˌbɑnt ɛn ˈvɛlkroː zɛin hɛtˈzɛlfdə',
            'Komend weekend gaan we het autosalon bezoeken.': 'ˈkoːmənt ˈʋiːkɛnt ɣaːn ʋə hət ˈɑutoˌsaˌlɔn bəˈzukən',
            'Kuifje is een bekend stripfiguur in België.': 'ˈkœyfjə ɪs ən bəˈkɛnt ˈstrɪpfiˌɣyːr ɪn ˈbɛlɣijə',
            'Kunnen jullie allemaal in een kring gaan staan?': 'ˈkʏnən ˈjʏli ˈɑləmaːl ɪn ən krɪŋ ɣaːn staːn',
            'Kwallen kunnen hun vorm enkel behouden in het water.': 'ˈkʋɑlən ˈkʏnən hʏn vɔrm ˈɛŋkəl bəˈhɑudən ɪn hət ˈʋaːtər',
            'Laura en Matthias speelden in het lange gras.': 'ˈlɑuraː ɛn mɑˈtiaːs ˈspeːldən ɪn hət ˈlɑŋə ɣrɑs',
            'Laura pakte een sigaret en plaatste die in haar mond.': 'ˈlɑuraː ˈpɑktə ən siɣaˈrɛt ɛn ˈplaːtstə di ɪn haːr mɔnt',
            'Leon en Finn kennen haar.': 'ˈleːɔn ɛn fɪn ˈkɛnən haːr',
            'Let op voor de hoge kosten op een visakaart.': 'lɛt ɔp voːr də ˈhoːɣə ˈkɔstən ɔp ən ˈvizaˌkaːrt',
            'Leuk je ontmoet te hebben.': 'løːk jə ɔntˈmut tə ˈhɛbən',
            'Leveren ze nog zo laat?': 'leˈveːrən zə nɔx zoː laːt',
            'Lid worden van de vereniging was heel makkelijk.': 'lɪt ˈʋɔrdən vɑn də vərˈeːnəɣɪŋ ʋɑs heːl ˈmɑkələk',
            'Lien zette de pan op het fornuis.': 'lin ˈzɛtə də pɑn ɔp hət fɔrˈnœys',
            'Maastricht bevindt zich aan beide kanten van de Maas.': 'maːˈstrɪxt bəˈvɪnt zɪx aːn ˈbɛidə ˈkɑntən vɑn də maːs',
            'Madrid en Barcelona zijn de grootste steden van Spanje.': 'maˈdrɪt ɛn bɑrsəˈloːna zɛin də ˈɣroːtstə ˈsteːdən vɑn ˈspɑnjə',
            'Margarine is eigenlijk boter met een minderwaardigheidscomplex.': 'mɑrɣaˈrinə ɪs ˈɛiɣənlək ˈboːtər mɛt ən ˈmɪndərˌʋaːrdəxhɛitsˌkɔmplɛks',
            'Marleen geeft sinds kort les op de middelbare school.': 'mɑrˈleːn ɣeːft sɪnts kɔrt lɛs ɔp də ˈmɪdəlˌbaːrə sxoːl',
            'Max keek mij recht in de ogen.': 'mɑks keːk mɛi rɛxt ɪn də ˈoːɣən',
            'Merkwaardig genoeg staan de bloemen nu al in bloei.': 'ˈmɛrkˌʋaːrdəx ɣəˈnux staːn də ˈblumən ny ɑl ɪn blui',
            'Met dit weer heb ik meer last van astma.': 'mɛt dɪt ʋeːr hɛp ɪk meːr lɑst vɑn ˈɑstma',
            'Met dit winterweer heb ik zin in een warme appeltaart.': 'mɛt dɪt ˈʋɪntərˌʋeːr hɛp ɪk zɪn ɪn ən ˈʋɑrmə ˈɑpəlˌtaːrt',
            'Met een schoenlepel kreeg ik de schoen uiteindelijk aan.': 'mɛt ən ˈsxunˌleːpəl kreːx ɪk də sxun œytˈɛindələk aːn',
            'Met jachtgeweren werd er op kleiduiven geschoten.': 'mɛt ˈjɑxtɣəˌʋeːrən ʋɛrt ər ɔp ˈklɛiˌdœyvən ɣəˈsxoːtən',
            'Met knikkende knieën liep ik het kantoor binnen.': 'mɛt ˈknɪkəndə ˈkniən lip ɪk hət kɑnˈtoːr ˈbɪnən',
            'Met naald en draad kan je naaien.': 'mɛt naːlt ɛn draːt kɑn jə ˈnaːiən',
            'Met welke snelheid verplaatst geluid zich onderwater?': 'mɛt ˈʋɛlkə ˈsnɛlhɛit vərˈplaːtst ɣəˈlœyt zɪx ˈɔndərˌʋaːtər',
            'Michiel dronk van zijn drinkbus.': 'miˈxil drɔŋk vɑn zɛin ˈdrɪŋkbʏs',
            'Mijn baas stond op mijn vingers te kijken.': 'mɛin baːs stɔnt ɔp mɛin ˈvɪŋərs tə ˈkɛikən',
            'Mijn buurvrouw is aan het klagen.': 'mɛin ˈbyːrvrɑu ɪs aːn hət ˈklaːɣən',
            'Mijn collega heeft te maken met burn-out klachten.': 'mɛin kɔˈleːɣa heːft tə ˈmaːkən mɛt ˈbøːrnɑut ˈklɑxtən',
            'Mijn grafische kaart is van Nvidia.': 'mɛin ˈɣraːfisə kaːrt ɪs vɑn ɛnˈvidia',
            'Mijn grootmoeder heeft kanker gehad en een borst laten amputeren.': 'mɛin ˈɣroːtˌmudər heːft ˈkɑŋkər ɣəˈhɑt ɛn ən bɔrst ˈlaːtən ɑmpyˈteːrən',
            'Mijn mama maakt zelf verse soep.': 'mɛin ˈmaːma maːkt zɛlf ˈvɛrsə sup',
            'Mijn ouders wonen dicht bij Hoofddorp.': 'mɛin ˈɑudərs ˈʋoːnən dɪxt bɛi ˈhoːftɔrp',
            'Mijn trekrugzak woog achttien kilo toen we die trektocht maakten.': 'mɛin ˈtrɛkˌrʏxˌzɑk ʋoːx ˈɑxtin ˈkiloː tun ʋə di ˈtrɛktɔxt ˈmaːktən',
            'Mijn wachtwoord is zonder mijn toestemming veranderd.': 'mɛin ˈʋɑxtˌʋoːrt ɪs ˈzɔndər mɛin tuˈstɛmɪŋ vərˈɑndərt',
            'Na de woordenwisseling was de sfeer grim.': 'naː də ˈʋoːrdənˌʋɪsəlɪŋ ʋɑs də sfeːr ɣrɪm',
            'Na die regenbui was hij nat van kop tot teen.': 'naː di ˈreːɣənˌbœy ʋɑs hɛi nɑt vɑn kɔp tɔt teːn',
            'Na een lange droogte viel er weer een bui.': 'naː ən ˈlɑŋə ˈdroːxtə vil ər ʋeːr ən bœy',
            'Na het skiën voelde ik het serieus in mijn beenspieren.': 'naː hət ˈskiən ˈvuldə ɪk hət seˈrijøːs ɪn mɛin ˈbeːnˌspirən',
            'Nederland heeft een multiculturele samenleving.': 'ˈneːdərlɑnt heːft ən mʏltiˌkʏltyˈreːlə ˈsaːmənˌleːvɪŋ',
            'Nee dat denk ik niet.': 'neː dɑt dɛŋk ɪk nit',
            'Niet alle zinnen worden gebruikt om het model te trainen.': 'nit ˈɑlə ˈzɪnən ˈʋɔrdən ɣəˈbrœykt ɔm hət moˈdɛl tə ˈtreːnən',
            'Niet met je schoenen in bed!': 'nit mɛt jə ˈsxunən ɪn bɛt',
            'Niet veel mensen hebben last van hoogtevrees in een reuzenrad.': 'nit veːl ˈmɛnsən ˈhɛbən lɑst vɑn ˈhoːxtəˌvreːs ɪn ən ˈrøːzənˌrɑt',
            'Nike en Adidas sponsoren veel topsporters.': 'ˈnɑiki ɛn ˈaːdidɑs ˈspɔnsɔrən veːl ˈtɔpˌspɔrtərs',
            'Noorwegen heeft een grote bron met olie en gas.': 'ˈnoːrˌʋeːɣən heːft ən ˈɣroːtə brɔn mɛt ˈoːli ɛn ɣɑs',
            'Om half negen heb ik een afspraak bij de dokter.': 'ɔm hɑlf ˈneːɣən hɛp ɪk ən ˈɑfspraːk bɛi də ˈdɔktər',
            'Om te winnen zal je veel moeten trainen.': 'ɔm tə ˈʋɪnən zɑl jə veːl ˈmutən ˈtreːnən',
            'Online communiceren ze via de webcam.': 'ɔnˈlɑin kɔmyniˈkeːrən zə ˈvia də ˈʋɛpkɛm',
            'Onze computers zijn niet sterk genoeg voor die simulaties.': 'ˈɔnzə kɔmˈpjutərs zɛin nit stɛrk ɣəˈnux voːr di simyˈlaːtsis',
            'Onze verre voorouders waren holbewoners.': 'ˈɔnzə ˈvɛrə voːrˈɑudərs ˈʋaːrən ˈhɔlbəˌʋoːnərs',
            'Onze yogalerares is enorm lenig.': 'ˈɔnzə ˈjoːɣaˌleːˌraːrəs ɪs eˈnɔrm ˈleːnəx',
            'Op de rotonde neem je de vierde afslag.': 'ɔp də roˈtɔndə neːm jə də ˈvirdə ˈɑfslɑx',
            'Op die snelweg starten we een pilootproject voor de trajectcontrole.': 'ɔp di ˈsnɛlʋɛx ˈstɑrtən ʋə ən piˈloːtˌproˌjɛkt voːr də traˈjɛktkɔnˌtroːlə',
            'Op een bouwwerf moet iedereen verplicht een helm dragen.': 'ɔp ən ˈbɑuˌʋɛrf mut ˈidəreːn vərˈplɪxt ən hɛlm ˈdraːɣən',
            'Op het broodje zat ham en kaas.': 'ɔp hət ˈbroːtjə zɑt hɑm ɛn kaːs',
            'Op het label stond: uitsluitend in gesloten verpakking bewaren.': 'ɔp hət ˈleːbəl stɔnt ˈœytslœytənt ɪn ɣəˈsloːtən vərˈpɑkɪŋ bəˈʋaːrən',
            'Op mijn kamer heb ik een grote wereldkaart hangen.': 'ɔp mɛin ˈkaːmər hɛp ɪk ən ˈɣroːtə ˈʋeːrəltˌkaːrt ˈhɑŋən',
            'Op vrijdag ging iedereen naar de disco.': 'ɔp ˈvrɛidɑx ɣɪŋ ˈidəreːn naːr də ˈdɪskoː',
            'Op zijn blote voeten liep hij over de hete kolen.': 'ɔp zɛin ˈbloːtə ˈvutən lip hɛi ˈoːvər də ˈheːtə ˈkoːlən',
            'Op zijn pet staat een slogan van een bekende voetbalploeg.': 'ɔp zɛin pɛt staːt ən ˈsloːɣɑn vɑn ən bəˈkɛndə ˈvutbɑlˌplux',
            'Optimisme is de sleutel voor een goed leven.': 'ɔptiˈmɪsmə ɪs də ˈsløːtəl voːr ən ɣut ˈleːvən',
            'Over een uurtje gaan we doorgaan.': 'ˈoːvər ən ˈyːrtjə ɣaːn ʋə ˈdoːrˌɣaːn',
            'Overblijven in de pauze wordt steeds normaler.': 'ˈoːvərˌblɛivən ɪn də ˈpɑuzə ʋɔrt steːts nɔrˈmaːlər',
            'Papa en mama hebben mijn broer geholpen bij zijn verhuis.': 'ˈpaːpa ɛn ˈmaːma ˈhɛbən mɛin brur ɣəˈhɔlpən bɛi zɛin vərˈhœys',
            'Papagaaien kunnen heel oud worden.': 'papaˈɣaːiən ˈkʏnən heːl ɑut ˈʋɔrdən',
            'Rijke kinderen beginnen toch met een voorsprong.': 'ˈrɛikə ˈkɪndərən bəˈɣɪnən tɔx mɛt ən ˈvoːrsprɔŋ',
            'Romeo sprak tegen Julia die op het balkon stond.': 'ˈroːmeˌoː sprɑk ˈteːɣən ˈjylia di ɔp hət bɑlˈkɔn stɔnt',
            'Rond Arnhem is veel gevochten in de Tweede Wereldoorlog.': 'rɔnt ˈɑrnhɛm ɪs veːl ɣəˈvɔxtən ɪn də ˈtʋeːdə ˈʋeːrəltˌoːrlɔx',
            'Rozijnen zijn echt heel lekker.': 'roˈzɛinən zɛin ɛxt heːl ˈlɛkər',
            'Samen zetten we zijn levenswerk voort.': 'ˈsaːmən ˈzɛtən ʋə zɛin ˈleːvənsˌʋɛrk voːrt',
            'Samen zongen ze het volkslied.': 'ˈsaːmən ˈzɔŋən zə hət ˈvɔlkslit',
            'Schrijf je dat woord met of zonder trema?': 'sxrɛif jə dɑt ʋoːrt mɛt ɔf ˈzɔndər ˈtreːma',
            'Servië en Kosovo komen niet al te best overeen.': 'ˈsɛrvijə ɛn ˈkɔsovoː ˈkoːmən nit ɑl tə bɛst ˈoːvəreːn',
            'Slaap je in een stapelbed?': 'slaːp jə ɪn ən ˈstaːpəlˌbɛt',
            'Slangen kunnen venijnig en giftig zijn.': 'ˈslɑŋən ˈkʏnən vəˈnɛinəx ɛn ˈɣɪftəx zɛin',
            'Sleep me nou niet mee in je persoonlijke problemen.': 'sleːp mə nɑu nit meː ɪn jə pɛrˈsoːnləkə proˈbleːmən',
            'Sommige bureaustoelen zijn ergonomisch niet verantwoord.': 'ˈsɔməɣə byˈroːˌstulən zɛin ɛrɣoˈnoːmis nit vərˈɑntʋoːrt',
            'Sommige zinnen worden apart gehouden om het model te testen.': 'ˈsɔməɣə ˈzɪnən ˈʋɔrdən aˈpɑrt ɣəˈhɑudən ɔm hət moˈdɛl tə ˈtɛstən',
            'Spaghetti en andere pasta komen volgens mij uit Italië.': 'spaˈɣɛti ɛn ˈɑndərə ˈpɑsta ˈkoːmən ˈvɔlɣəns mɛi œyt iˈtaːlijə',
            'Speelt Moeskroen nog in eerste klasse?': 'speːlt ˈmuskrun nɔx ɪn ˈeːrstə ˈklɑsə',
            'Spijtig genoeg zijn de treinen nog vaak duurder dan vliegen.': 'ˈspɛitəx ɣəˈnux zɛin də ˈtrɛinən nɔx vaːk ˈdyːrdər dɑn ˈvliɣən',
            'Stamppot is een gerecht voor liefhebbers.': 'ˈstɑmpɔt ɪs ən ɣəˈrɛxt voːr ˈlifˌhɛbərs',
            'Steenkool wordt gedolven uit de grond.': 'ˈsteːnkoːl ʋɔrt ɣəˈdɔlvən œyt də ɣrɔnt',
            'Stella Artois is een pils die wordt gebrouwen in Leuven.': 'ˈstɛla ɑrˈtʋa ɪs ən pɪls di ʋɔrt ɣəˈbrɑuʋən ɪn ˈløːvən',
            'tv': 'teːˈveː',
            'Te voet begaven ze zich van Amersfoort naar Utrecht.': 'tə vut bəˈɣaːvən zə zɪx vɑn ˈaːmərsˌfoːrt naːr ˈytrɛxt',
            'Tegenover het café zat een groot kantoorpand.': 'ˈteːɣənˌoːvər hət kaˈfeː zɑt ən ɣroːt kɑnˈtoːrˌpɑnt',
            'Telenet en Proximus hebben concurrerende producten in Vlaanderen.': 'ˈteːləˌnɛt ɛn ˈprɔksimʏs ˈhɛbən kɔŋkyˈreːrəndə proˈdʏktən ɪn ˈvlaːndərən',
            'Tennis speel je met een ronde bal.': 'ˈtɛnɪs speːl jə mɛt ən ˈrɔndə bɑl',
            'Thuis hebben we een anti-inbraakalarm.': 'tœys ˈhɛbən ʋə ən ˈɑntiˌɪnbraːkˌaˌlɑrm',
            'Tijdens Kerstmis zijn er gewoonlijk minder winkels open.': 'ˈtɛidəns ˈkɛrstmɪs zɛin ər ɣəˈʋoːnlək ˈmɪndər ˈʋɪŋkəls ˈoːpən',
            'Tijdens de crisis kon hij zijn flat amper verkopen.': 'ˈtɛidəns də ˈkrisɪs kɔn hɛi zɛin flɛt ˈɑmpər vərˈkoːpən',
            'Tijgers en leeuwen zijn eng.': 'ˈtɛiɣərs ɛn ˈleːuʋən zɛin ɛŋ',
            'Toen ze jong waren hadden ze samen een boomhut gebouwd.': 'tun zə jɔŋ ˈʋaːrən ˈhɑdən zə ˈsaːmən ən ˈboːmhʏt ɣəˈbɑut',
            'Transcripteren is het neerschrijven van gesproken tekst.': 'trɑnskripˈteːrən ɪs hət ˈneːrˌsxrɛivən vɑn ɣəˈsproːkən tɛkst',
            'Tussen twee woorden zet je een spatie.': 'ˈtʏsən tʋeː ˈʋoːrdən zɛt jə ən ˈspaːtsi',
            'Twee pintjes en een cola alsjeblieft.': 'tʋeː ˈpɪntjəs ɛn ən ˈkoːla ɑlsjəˈblift',
            'Utrecht is eigenlijk best een kleine provincie.': 'ˈytrɛxt ɪs ˈɛiɣənlək bɛst ən ˈklɛinə proˈvɪnsi',
            'Uv-straling wordt tegengehouden door ozon in de ozonlaag.': 'yˈveːˌstraːlɪŋ ʋɔrt ˈteːɣənɣəˌhɑudən doːr oˈzɔn ɪn də oˈzɔnlaːx',
            'Van Zaventem vlogen we naar Schiphol.': 'vɑn ˈzaːvəntɛm ˈvloːɣən ʋə naːr ˈsxɪpɔl',
            'Van appelsienen kan je appelsiensap maken.': 'vɑn ɑpəlˈsinən kɑn jə ɑpəlˈsinsɑp ˈmaːkən',
            'Vanaf toen ging alles bergaf.': 'vɑnˈɑf tun ɣɪŋ ˈɑləs ˈbɛrxˌɑf',
            'Vandalen hebben mijn auto beschadigd.': 'vɑnˈdaːlən ˈhɛbən mɛin ˈɑutoː bəˈsxaːdəxt',
            'Veel mensen vinden het leuk om te reageren op fora.': 'veːl ˈmɛnsən ˈvɪndən hət løːk ɔm tə reːaˈɣeːrən ɔp ˈfoːra',
            'Veel van de soldaten vormden samen een team.': 'veːl vɑn də sɔlˈdaːtən ˈvɔrmdən ˈsaːmən ən tim',
            'Veganistisch eten gaat verder dan vegetarisch.': 'veːɣaˈnɪstis ˈeːtən ɣaːt ˈvɛrdər dɑn veːɣəˈtaːris',
            'Venetië wordt overspoeld door toeristen.': 'vəˈneːtsijə ʋɔrt ˈoːvərˌspult doːr tuˈrɪstən',
            'Via Google kan je wel een afbeelding vinden.': 'ˈvia ˈɣuɣəl kɑn jə ʋɛl ən ˈɑfˌbeːldɪŋ ˈvɪndən',
            'Vier plus acht is twaalf.': 'viːr plʏs ɑxt ɪs tʋaːlf',
            'Vijf gedeeld door vier is één komma vijfentwintig.': 'vɛif ɣəˈdeːlt doːr viːr ɪs eːn ˈkɔma ˌvɛifənˈtʋɪntəx',
            'Vleermuizen kunnen vliegen zonder licht.': 'ˈvleːrmœyzən ˈkʏnən ˈvliɣən ˈzɔndər lɪxt',
            'Volg het jaagpad dat naast het kanaal ligt.': 'vɔlx hət ˈjaːxpɑt dɑt naːst hət kaˈnaːl lɪxt',
            'Volgens de boordcomputer is er iets mis met mijn bandenspanning.': 'ˈvɔlɣəns də ˈboːrtkɔmˌpjutər ɪs ər its mɪs mɛt mɛin ˈbɑndənˌspɑnɪŋ',
            'Volgens mij werkt Skype niet meer.': 'ˈvɔlɣəns mɛi ʋɛrkt skɑip nit meːr',
            'Voor de vrijgezellen gaan we een namiddag paintballen en karten.': 'voːr də ˈvrɛiɣəˌzɛlən ɣaːn ʋə ən ˈnaːˌmɪdɑx ˈpeːntbɔlən ɛn ˈkɑrtən',
            'Voor dringende medische problemen ga je best naar de spoeddienst.': 'voːr ˈdrɪŋəndə ˈmeːdisə proˈbleːmən ɣaː jə bɛst naːr də ˈsputdinst',
            'Voor een begrafenis dragen de meeste mensen zwarte kleren.': 'voːr ən bəˈɣraːfənɪs ˈdraːɣən də ˈmeːstə ˈmɛnsən ˈzʋɑrtə ˈkleːrən',
            'Voor het eten was je eerst je handen met zeep.': 'voːr hət ˈeːtən ʋɑs jə eːrst jə ˈhɑndən mɛt zeːp',
            'Vroeger had dat dorp een omwalling.': 'ˈvruɣər hɑt dɑt dɔrp ən ɔmˈʋɑlɪŋ',
            'Waar gaat al het afval naartoe?': 'ʋaːr ɣaːt ɑl hət ˈɑfvɑl ˈnaːrtu',
            'Waarom moet ik weer de boeman zijn?': 'ʋaːrˈɔm mut ɪk ʋeːr də ˈbumɑn zɛin',
            'Walvissen zijn zoogdieren die in het water leven.': 'ˈʋɑlvɪsən zɛin ˈzoːxˌdirən di ɪn hət ˈʋaːtər ˈleːvən',
            'Wat is de topsnelheid van die wagen?': 'ʋɑt ɪs də ˈtɔpsnɛlˌhɛit vɑn di ˈʋaːɣən',
            'Wat is de weersverwachting deze week?': 'ʋɑt ɪs də ˈʋeːrsvərˌʋɑxtɪŋ ˈdeːzə ʋeːk',
            'Wat is je gebruikersnaam en wachtwoord?': 'ʋɑt ɪs jə ɣəˈbrœykərsˌnaːm ɛn ˈʋɑxtˌʋoːrt',
            'Wat je niet verdient kan je ook niet uitgeven.': 'ʋɑt jə nit vərˈdint kɑn jə oːk nit ˈœytˌɣeːvən',
            'Wat voor een lens heb je gekocht?': 'ʋɑt voːr ən lɛns hɛp jə ɣəˈkɔxt',
            'We hebben moeten constateren dat er veel geld verspild wordt.': 'ʋə ˈhɛbən ˈmutən kɔnstaˈteːrən dɑt ər veːl ɣɛlt vərˈspɪlt ʋɔrt',
            'We kunnen echt geen cent meer uitgeven aan de tuin.': 'ʋə ˈkʏnən ɛxt ɣeːn sɛnt meːr ˈœytˌɣeːvən aːn də tœyn',
            'We kunnen ook voor Thais gaan.': 'ʋə ˈkʏnən oːk voːr taːis ɣaːn',
            'We lossen dat wel op in de montage.': 'ʋə ˈlɔsən dɑt ʋɛl ɔp ɪn də mɔnˈtaːʒə',
            'We vlogen in een Airbus.': 'ʋə ˈvloːɣən ɪn ən ˈɛːrbʏs',
            'We willen onze eigen nieuwbouw project realiseren.': 'ʋə ˈʋɪlən ˈɔnzə ˈɛiɣən ˈniubɑu proˈjɛkt reːaliˈzeːrən',
            'We wonen in een appartement op de derde verdieping.': 'ʋə ˈʋoːnən ɪn ən ɑpɑrtəˈmɛnt ɔp də ˈdɛrdə vərˈdipɪŋ',
            'We zijn gaan zeilen bij de Waddeneilanden.': 'ʋə zɛin ɣaːn ˈzɛilən bɛi də ˈʋɑdənˌɛilɑndən',
            'Welk beleg heb je op je boterhammen gedaan?': 'ʋɛlk bəˈlɛx hɛp jə ɔp jə ˈboːtərˌhɑmən ɣəˈdaːn',
            'Welke dag is het vandaag?': 'ˈʋɛlkə dɑx ɪs hət vɑnˈdaːx',
            'Welke drug is het sterkst?': 'ˈʋɛlkə drʏx ɪs hət stɛrkst',
            'Welke landen behoren tot de Balkan?': 'ˈʋɛlkə ˈlɑndən bəˈhoːrən tɔt də ˈbɑlkɑn',
            'Wenen is de hoofdstad van Oostenrijk.': 'ˈʋeːnən ɪs də ˈhoːftstɑt vɑn ˈoːstənrɛik',
            'Wie is de burgemeester van Kopenhagen?': 'ʋi ɪs də ˈbʏrɣəˌmeːstər vɑn koːpənˈhaːɣən',
            'Wie is verantwoordelijk voor deze misdaad?': 'ʋi ɪs vərˈɑntˌʋoːrdələk voːr ˈdeːzə mɪsˈdaːt',
            'Wij liepen door de wei.': 'ʋɛi ˈlipən doːr də ʋɛi',
            'Wil je feedback geven op het idee?': 'ʋɪl jə ˈfitbɛk ˈɣeːvən ɔp hət iˈdeː',
            'Winkeldiefstal kost winkeliers handenvol geld.': 'ˈʋɪŋkəlˌdifstɑl kɔst ˈʋɪŋkəlirs ˈhɑndənˌvɔl ɣɛlt',
            'Wist je dat ze elke januari dicht zijn?': 'ʋɪst jə dɑt zə ˈɛlkə jɑnyˈaːri dɪxt zɛin',
            'Zalig kerstfeest en een gelukkig Nieuwjaar.': 'ˈzaːləx ˈkɛrstˌfeːst ɛn ən ɣəˈlʏkəx ˈniuˌjaːr',
            'Zaventem en Schiphol zijn twee grote luchthavens.': 'ˈzaːvəntɛm ɛn ˈsxɪpɔl zɛin tʋeː ˈɣroːtə ˈlʏxtˌhaːvəns',
            'Ze bezitten een groot landgoed op het platteland.': 'zə bəˈzɪtən ən ɣroːt ˈlɑntˌɣut ɔp hət ˈplɑtəlɑnt',
            'Ze brak haar arm op meerdere plekken.': 'zə brɑk haːr ɑrm ɔp ˈmeːrdərə ˈplɛkən',
            'Ze draagt een petje om een zonnesteek te vermijden.': 'zə draːxt ən ˈpɛtjə ɔm ən ˈzɔnəˌsteːk tə vərˈmɛidən',
            'Ze dronken de melk en vielen in slaap.': 'zə ˈdrɔŋkən də mɛlk ɛn ˈvilən ɪn slaːp',
            'Ze ergert zich aan de trage computer.': 'zə ˈɛrɣərt zɪx aːn də ˈtraːɣə kɔmˈpjutər',
            'Ze feestten verder tot in de vroege uurtjes.': 'zə ˈfeːstən ˈvɛrdər tɔt ɪn də ˈvruɣə ˈyːrtjəs',
            'Ze gaat morgen voor de allereerste keer naar school.': 'zə ɣaːt ˈmɔrɣən voːr də ˈɑləreːrstə keːr naːr sxoːl',
            'Ze had al veel levenservaring voor haar leeftijd.': 'zə hɑt ɑl veːl ˈleːvənsɛrˌvaːrɪŋ voːr haːr ˈleːftɛit',
            'Ze had een krachtige handdruk.': 'zə hɑt ən ˈkrɑxtəɣə ˈhɑntdrʏk',
            'Ze hadden dat bouwvallig gebouw al jaren geleden moeten stutten.': 'zə ˈhɑdən dɑt ˈbɑuˌvɑləx ɣəˈbɑu ɑl ˈjaːrən ɣəˈleːdən ˈmutən ˈstʏtən',
            'Ze hebben hem de laan uitgestuurd.': 'zə ˈhɛbən hɛm də laːn ˈœytɣəˌstyːrt',
            'Ze hebben urenlang vastgezeten in de skilift.': 'zə ˈhɛbən ˈyːrənlɑŋ ˈvɑstɣəˌzeːtən ɪn də ˈskiˌlɪft',
            'Ze heeft dat geleerd op de tekenles.': 'zə heːft dɑt ɣəˈleːrt ɔp də ˈteːkənˌlɛs',
            'Ze hielden elkaars hand vast.': 'zə ˈhildən ɛlˈkaːrs hɑnt vɑst',
            'Ze is een bekend model.': 'zə ɪs ən bəˈkɛnt moˈdɛl',
            'Ze kent veel van de Indonesische cultuur.': 'zə kɛnt veːl vɑn də ɪndoˈneːzisə kʏlˈtyːr',
            'Ze moet lachen met mijn mopjes.': 'zə mut ˈlɑxən mɛt mɛin ˈmɔpjəs',
            'Ze overdrijven toch met hun heksenjacht.': 'zə ˈoːvərˌdrɛivən tɔx mɛt hʏn ˈhɛksənˌjɑxt',
            'Ze peddelden met de kajak naar het eiland.': 'zə ˈpɛdəldən mɛt də ˈkaːjɑk naːr hət ˈɛilɑnt',
            'Ze plofte neer in de comfortabele stoel.': 'zə ˈplɔftə neːr ɪn də kɔmfɔrˈtaːbələ stul',
            'Ze spelen buiten in de tuin.': 'zə ˈspeːlən ˈbœytən ɪn də tœyn',
            'Ze staarde uit het raam.': 'zə ˈstaːrdə œyt hət raːm',
            'Ze verleende voorrang aan de vrachtwagen.': 'zə vərˈleːndə ˈvoːrɑŋ aːn də ˈvrɑxtˌʋaːɣən',
            'Ze voelt zich op haar gemak bij hem.': 'zə vult zɪx ɔp haːr ɣəˈmɑk bɛi hɛm',
            'Ze waadde door het ondiepe water.': 'zə ˈʋaːdə doːr hət ˈɔndipə ˈʋaːtər',
            'Ze was geboren om grootse dingen te verwezenlijken.': 'zə ʋɑs ɣəˈboːrən ɔm ˈɣroːtsə ˈdɪŋən tə vərˈʋeːzənˌlɛikən',
            'Ze willen het aantal franchises dit jaar sterk uitbreiden.': 'zə ˈʋɪlən hət ˈaːntɑl frɛnˈʃɑizəs dɪt jaːr stɛrk ˈœytˌbrɛidən',
            'Ze wou weer de heldin uithangen.': 'zə ʋɑu ʋeːr də ˈhɛldɪn ˈœytˌhɑŋən',
            'Ze zijn geëmigreerd naar Indië in tweeduizend en vijf.': 'zə zɛin ɣəeːmiˈɣreːrt naːr ˈɪndijə ɪn ˌtʋeːˈdœyzənt ɛn vɛif',
            'Ze zijn in het park gaan picknicken.': 'zə zɛin ɪn hət pɑrk ɣaːn ˈpɪknɪkən',
            'Ze zijn momenteel bezig met de fundering van het gebouw.': 'zə zɛin moˈmɛnteːl ˈbeːzəx mɛt də fʏnˈdeːrɪŋ vɑn hət ɣəˈbɑu',
            'Ze zijn op zoek naar de schat van de piraten.': 'zə zɛin ɔp zuk naːr də sxɑt vɑn də piˈraːtən',
            'Ze zit op de achterbank.': 'zə zɪt ɔp də ˈɑxtərˌbɑŋk',
            'Ze zocht nog naar haar bh.': 'zə zɔxt nɔx naːr haːr beːˈhaː',
            'Zes keer twee is twaalf.': 'zɛs keːr tʋeː ɪs tʋaːlf',
            'Zetelverwarming is enorm comfortabel in de winter.': 'ˈzeːtəlvərˌʋɑrmɪŋ ɪs eˈnɔrm kɔmfɔrˈtaːbəl ɪn də ˈʋɪntər',
            'Zij beweert dat de verlichting en de renaissance hetzelfde zijn.': 'zɛi bəˈʋeːrt dɑt də vərˈlɪxtɪŋ ɛn də rəneˈsɑns hɛtˈzɛlfdə zɛin',
            'Zij doet kaartentrucs die niemand kan begrijpen.': 'zɛi dut ˈkaːrtənˌtrʏks di ˈnimɑnt kɑn bəˈɣrɛipən',
            'Zij droeg de balk op haar schouder.': 'zɛi drux də bɑlk ɔp haːr ˈsxɑudər',
            'Zij heeft een kat in een zak gekocht.': 'zɛi heːft ən kɑt ɪn ən zɑk ɣəˈkɔxt',
            'Zij heeft een publicatie in dat wetenschappelijk tijdschrift.': 'zɛi heːft ən pybliˈkaːtsi ɪn dɑt ˈʋeːtənsxɑpələk ˈtɛitsxrɪft',
            'Zij heeft een uitgebreide woordenschat.': 'zɛi heːft ən ˈœytɣəˌbrɛidə ˈʋoːrdənsxɑt',
            'Zij heeft het wereldkampioenschap snooker gewonnen.': 'zɛi heːft hət ˈʋeːrəltˌkɑmpiˌunsxɑp ˈsnukər ɣəˈʋɔnən',
            'Zij heeft nog nooit in het ziekenhuis gelegen.': 'zɛi heːft nɔx noːit ɪn hət ˈzikənˌhœys ɣəˈleːɣən',
            'Zij heeft nooit een tweede zit gehad.': 'zɛi heːft noːit ən ˈtʋeːdə zɪt ɣəˈhɑt',
            'Zij is verantwoordelijk voor het onderhoud van de vaten.': 'zɛi ɪs vərˈɑntˌʋoːrdələk voːr hət ˈɔndərˌhɑut vɑn də ˈvaːtən',
            'Zij is voorzitster van de raad van bestuur.': 'zɛi ɪs voːrˈzɪtstər vɑn də raːt vɑn bəˈstyːr',
            'Zij is wel een pientere dame.': 'zɛi ɪs ʋɛl ən piˈɛntərə ˈdaːmə',
            'Zij kan echt enorm goed gitaar spelen.': 'zɛi kɑn ɛxt eˈnɔrm ɣut ɣiˈtaːr ˈspeːlən',
            'Zij snoof cocaïne van een bankkaart.': 'zɛi snoːf kokaˈinə vɑn ən ˈbɑŋkˌkaːrt',
            'Zij verloor haar evenwicht en viel.': 'zɛi vərˈloːr haːr ˈeːvənˌʋɪxt ɛn vil',
            'Zij was nooit voorzichtig geweest.': 'zɛi ʋɑs noːit voːrˈzɪxtəx ɣəˈʋeːst',
            'Zij was volledig in shock na die traumatische ervaring.': 'zɛi ʋɑs vɔˈleːdəx ɪn ʃɔk naː di trɑuˈmaːtisə ɛrˈvaːrɪŋ',
            'Zij werkt als rechter voor het Europees Hof van Justitie.': 'zɛi ʋɛrkt ɑls ˈrɛxtər voːr hət øːroˈpeːs hɔf vɑn jʏsˈtitsi',
            'Zij wordt graag gemasseerd op een massagetafel.': 'zɛi ʋɔrt ɣraːx ɣəmɑˈseːrt ɔp ən mɑˈsaːʒəˌtaːfəl',
            'Zijn blik dwaalde af naar het afgebrande huis.': 'zɛin blɪk ˈdʋaːldə ɑf naːr hət ˈɑfɣəˌbrɑndə hœys',
            'Zijn blonde krullen zijn onweerstaanbaar.': 'zɛin ˈblɔndə ˈkrʏlən zɛin ˈɔnˌʋeːrˌstaːnbaːr',
            'Zijn er nog puntjes voor het varia onderdeel?': 'zɛin ər nɔx ˈpʏntjəs voːr hət ˈvaːria ˈɔndərˌdeːl',
            'Zijn vader had een mooie jacht.': 'zɛin ˈvaːdər hɑt ən ˈmoːiə jɑxt',
            'Zowel in het Duits als in Latijn gebruiken ze naamvallen.': 'zoˈʋɛl ɪn hət dœyts ɑls ɪn laˈtɛin ɣəˈbrœykən zə ˈnaːmˌvɑlən',
            'Zwitserland bleef neutraal in de Tweede Wereldoorlog.': 'ˈzʋɪtsərlɑnt bleːf nøˈtraːl ɪn də ˈtʋeːdə ˈʋeːrəltˌoːrlɔx',
            '\'s morgens': 'ˈsmɔrɣəns',
            'radio': 'ˈraːdioː',
            'deken': 'ˈdeːkən',
            'uitzuigen': 'ˈœytsœyɣən',
            'ballon': 'bɑˈlɔn',
            'warm': 'ʋɑrm',
            'longen': 'ˈlɔŋən',
            'jammer': 'ˈjɑmər',
            'goed': 'ɣut'
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
            # Diphthongs (two vowels pronounced as one)
            'ɛi', 'œy', 'ɑu',
            
            # Long vowels
            'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'ɑː', 'ɔː', 'ɛː', 'ɛ:', 'ɵ:',
            
            # True consonant clusters that function as single units
            'ŋk',  # As in "bank" - velar nasal + stop
            'sx',  # "sch" sound
            'tʃ',  # "ch" as in "church"
            'dʒ',  # "j" as in "judge"
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
            
        # Parse sentence-level transcriptions into individual words
        self.add_sentence_transcriptions()
    
        # Create reverse phoneme map
        self.reverse_phoneme_map = self._create_reverse_phoneme_map()
        
    def __getitem__(self, word: str) -> Optional[str]:
        """
        Get phonetic transcription for a word 

        """
        # Check custom entries first, then fall back to base dictionary
        return self.custom_entries.get(word, self.dictionary.get(word))
    
    def __contains__(self, word: str) -> bool:
        return word in self.custom_entries or word in self.dictionary
    
    def get_transcription(self, word: str) -> Optional[str]:
        return self[word]    
    
    def count_phonemes(self, word: str) -> int:
        
        chars_to_strip = string.punctuation + string.whitespace + '""„"\''
        word_cleaned = word.strip(chars_to_strip).lower()
        
        # Now check cleaned word
        if word_cleaned not in self:
            # Rough estimation
            return max(1, len(word_cleaned) - word_cleaned.count('oe') - 
                      word_cleaned.count('ie') - word_cleaned.count('ui') - 
                      word_cleaned.count('ij') - word_cleaned.count('eu'))
        
        # Get phonetic transcription
        transcription = self[word_cleaned]
        
        # Check if transcription is None
        if transcription is None:
            return max(1, len(word_cleaned) - word_cleaned.count('oe') - 
                      word_cleaned.count('ie') - word_cleaned.count('ui') - 
                      word_cleaned.count('ij') - word_cleaned.count('eu'))
        
        cleaned = self.clean_transcription(transcription)
        
        # Initialize phoneme count
        phoneme_count = len(cleaned)
        
        # Adjust for complex phonemes
        for cp in self.COMPLEX_PHONEMES:
            phoneme_count -= cleaned.count(cp)
        
        return max(1, phoneme_count)    
    
    def extract_phonemes(self, word: str) -> List[str]:
        """Extract individual phonemes from a word's transcription."""
        
        chars_to_strip = string.punctuation + string.whitespace + '""„"\''
        word_cleaned = word.strip(chars_to_strip).lower()
        
        # Check cleaned word
        if word_cleaned not in self:
            # Return characters as approximation (already cleaned)
            if hasattr(self, 'log'):
                self.log(f"Word not in dictionary: '{word_cleaned}' → using letter fallback")
            else:
                print(f"PhoneticDictionary: Word not in dictionary: '{word_cleaned}' → using letter fallback")

            # Track missing words
            if not hasattr(self, '_missing_words'):
                self._missing_words = set()
            self._missing_words.add(word_cleaned)
        
            return list(word_cleaned)
        
        # Get phonetic transcription using cleaned word
        transcription = self[word_cleaned]
        
         # Check if transcription is None
        if transcription is None:
            if hasattr(self, 'log'):
                self.log(f"Transcription is None for: '{word_cleaned}' → using letter fallback")
            else:
                print(f"PhoneticDictionary: Transcription is None for: '{word_cleaned}' → using letter fallback")
            
            if not hasattr(self, '_missing_words'):
                self._missing_words = set()
            self._missing_words.add(word_cleaned)
            
            return list(word_cleaned)
        
        # Clean transcription 
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
        markers_to_remove = ['ˈ', 'ˌ', '.', '|', '‖', '(', ')', "'", '?', ',', '"']
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
            
         #   '~u': ['u', 'uː', 'y', 'yː', 'ʏ', 'ɑu'],           
         #   'a/o long': ['oː', 'aː', 'ɔː'],
         #   'o back': ['ɔ', 'ə', 'o'],
         #   'i/e': ['i', 'ɪ', 'e'],
         #   'i/e long': ['eː', 'iː', 'ɛː', 'øː'],
         #   '~e': ['ɛi', 'ɛ'],
         #  '~a': ['œy', 'a', 'ɑ'],      
         # Vowels - short names
            'i-type': ['i', 'iː', 'ɪ', 'y', 'yː', 'ʏ'],     # high front
            'u-type': ['u', 'uː'],                           # high back
            'e-type': ['e', 'eː', 'ɛ', 'ɛː', 'øː'],         # mid front
            'o-type': ['o', 'oː', 'ɔ', 'ɔː'],               # mid back
            'a-type': ['a', 'aː', 'ɑ', 'ɑː'],               # low
            'schwa': ['ə'],                                  # reduced
            'diph': ['ɛi', 'ɑu', 'œy'],                     # diphthongs
            
            'l/n': ['l', 'n'],
            'k/g': ['k', 'g'],
            'p/b': ['p', 'b'],
            'sh/zh': ['ʃ', 'ʒ'], 
            '~x': ['x', 'h', 'ɦ', 'sx', 'ɣ', 'χ'],
            'm/n': ['m'],
            'f/v/w': ['f', 'v', 'w', 'ʋ'],
            'r': ['r'],
            't/d': ['t', 'd'],
            's/z': ['s', 'z'],
            'affricates': ['tʃ', 'dʒ', 'tɕ', 'dʑ', 'ts'],
            'palatal': ['j', 'ŋ', 'ŋk'],
            'ʋ': ['ʋ'],   # labiodental approximant - should be in 'f/v/w' or its own group
            'ʔ': ['ʔ'],   # glottal stop - mentioned in comments but not in active groups

        }
        
        # Create reverse mapping from phoneme to group
        self.phoneme_to_group = {}
        for group, phonemes in self.phoneme_groups.items():
            for phoneme in phonemes:
                self.phoneme_to_group[phoneme] = group
        
        self.phoneme_to_group['ˌ'] = 'marker'
        self.phoneme_to_group['?'] = 'unknown'
    
        return self.phoneme_groups
        
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
        
    def get_missing_words_summary(self):
        """Get summary of words that fell back to letter approximation."""
        if not hasattr(self, '_missing_words'):
            return "No missing words tracked"
        
        missing = self._missing_words
        return f"Missing words: {len(missing)}\nExamples: {list(missing)[:20]}"

    def reset_missing_words_tracker(self):
        """Reset the missing words tracker."""
        if hasattr(self, '_missing_words'):
            self._missing_words = set()
                        
    def add_sentence_transcriptions(self):
        """
        Parse sentence-level transcriptions into individual word entries.
        Handles contractions and merged words gracefully.
        """
        sentence_dict = {}
        
        # Separate sentence entries
        for key, transcription in self.dictionary.items():
            if ' ' in key:
                sentence_dict[key] = transcription
        
        self.log(f"Found {len(sentence_dict)} sentence-level entries")
        
        words_added = 0
        skipped_mismatches = 0
        
        for sentence, transcription in sentence_dict.items():
            # Clean sentence
            sentence_clean = sentence.strip('."""„"\'').lower()
            words = sentence_clean.split()
            
            # Clean transcription
            transcription_clean = transcription.replace('ˈ', '').replace('ˌ', '')
            phoneme_groups = transcription_clean.split()
            
            # Handle perfect matches
            if len(words) == len(phoneme_groups):
                for word, phonemes in zip(words, phoneme_groups):
                    word_clean = word.strip(string.punctuation).lower()
                    
                    if word_clean and word_clean not in self.dictionary:
                        self.dictionary[word_clean] = phonemes
                        words_added += 1
            
            # Handle contractions: merge orthographic words
            elif len(words) > len(phoneme_groups):
                # Try to align by merging contractions like "'s avonds" → "savonds"
                # Simple heuristic: merge words with apostrophes to next word
                merged_words = []
                i = 0
                while i < len(words):
                    if i + 1 < len(words) and words[i].startswith("'"):
                        # Merge with next word
                        merged = words[i] + words[i+1]
                        merged_words.append(merged.strip("'"))
                        i += 2
                    else:
                        merged_words.append(words[i])
                        i += 1
                
                # Try again with merged words
                if len(merged_words) == len(phoneme_groups):
                    for word, phonemes in zip(merged_words, phoneme_groups):
                        word_clean = word.strip(string.punctuation).lower()
                        if word_clean and word_clean not in self.dictionary:
                            self.dictionary[word_clean] = phonemes
                            words_added += 1
                else:
                    self.log(f"Mismatch (after merge): '{sentence}': {len(merged_words)} words vs {len(phoneme_groups)} phoneme groups")
                    skipped_mismatches += 1
            
            else:
                # More phoneme groups than words - less common
                self.log(f"Mismatch: '{sentence}': {len(words)} words vs {len(phoneme_groups)} phoneme groups")
                skipped_mismatches += 1
        
        self.log(f"Added {words_added} individual word entries")
        self.log(f"Skipped {skipped_mismatches} sentences with unresolvable mismatches")