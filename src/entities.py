"""
Entity extraction v2 — expanded patterns for relationships, ages, roles, dates, amounts.
Pure regex/pattern matching, no LLM needed.
"""
import re
from typing import List, Dict


class EntityExtractor:
    """Extract structured entities from text chunks."""
    
    def __init__(self):
        self.patterns = {
            'ip_address': re.compile(r'(?:IP|ip|address)[:\s]*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'),
            'ip_bare': re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'),
            'email': re.compile(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'),
            'phone': re.compile(r'\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
            'url': re.compile(r'(https?://[^\s\)\]]+)'),
            'password': re.compile(r'(?:password|passwd|pwd)[:\s]+[`"\']*([^\s`"\']+)', re.I),
            'token': re.compile(r'(?:token|key|secret|api[_\s]?key)[:\s]+[`"\']*([^\s`"\']{10,})', re.I),
            'ein': re.compile(r'(?:EIN)[:\s]*(\d{2}-\d{7})'),
            'ssn': re.compile(r'(?:SSN|social)[:\s]*(\d{3}-\d{2}-\d{4})', re.I),
            'address': re.compile(r'(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Trail|Street|St|Ave|Avenue|Road|Rd|Blvd|Boulevard|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir)\s*(?:North|South|East|West|N|S|E|W|NW|NE|SW|SE)?)(?:[,\s]|$)', re.I),
            'zip_code': re.compile(r'\b(\d{5}(?:-\d{4})?)\b'),
            'date': re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
            'money': re.compile(r'\$[\d,]+(?:\.\d{2})?(?:/(?:mo|month|year|yr|week|day))?', re.I),
            'percentage': re.compile(r'\b(\d+(?:\.\d+)?)\s*%'),
        }
        
        # Key-value patterns
        self.kv_pattern = re.compile(
            r'[-•]\s*\*{0,2}([^*:]+?)\*{0,2}\s*:\s*\*{0,2}(.+?)(?:\*{0,2})$',
            re.MULTILINE
        )
        
        # Relationship patterns (expanded)
        self.relationship_patterns = [
            # "X is [someone's] brother/sister/etc"
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was)\s+(?:(?:\w+)\'?s?\s+)?(?:brother|sister|daughter|son|wife|husband|father|mother|uncle|aunt|cousin|nephew|niece|grandpa|grandma|grandfather|grandmother|partner|girlfriend|boyfriend|fiancee?|ex)', re.I),
            # "brother/sister/etc: X" or "brother/sister/etc is X"
            re.compile(r'(?:brother|sister|daughter|son|wife|husband|father|mother|uncle|aunt|cousin|nephew|niece|grandpa|grandma|grandfather|grandmother|partner|girlfriend|boyfriend|fiancee?|ex)[:\s]+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
            # "### Daughter: [Name]" markdown heading style
            re.compile(r'#+\s*(?:Brother|Sister|Daughter|Son|Wife|Husband|Father|Mother)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
        ]
        
        # Age patterns
        self.age_patterns = [
            re.compile(r'(?:at|age|aged?)\s+(\d{1,3})\s+(?:years?\s+old|yo)', re.I),
            re.compile(r'(\d{1,3})\s+years?\s+old', re.I),
            re.compile(r'(?:At|age|aged?)\s+(\d{1,3})[,\s]', re.I),
        ]
        
        # Role/occupation patterns
        self.role_patterns = [
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|works as|serves as)\s+(?:a|an|the)\s+([a-z][a-z\s]+?)(?:\.|,|$)', re.I),
            re.compile(r'(?:serves?|work(?:s|ing)?|stationed)\s+(?:in|at|as)\s+(?:the\s+)?([A-Z][\w\s.]+?)(?:\.|,|and)', re.I),
        ]
        
        # State patterns (for insurance context)
        self.state_pattern = re.compile(r'\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b', re.I)
        
        # Product/service patterns
        self.product_patterns = [
            re.compile(r'(?:SR-?22|FR-?44|non-?owner|broad\s*form)\s+(?:insurance|policy|filing|coverage)', re.I),
        ]
        
        # v2.3: Person name patterns for conversation data
        self.person_name_patterns = [
            # "New hire announcement: First Last is joining..."
            re.compile(r'(?:New hire|new hire|Welcome|welcome)[^.]*?([A-Z][a-z]+\s+[A-Z][a-z]+)', re.I),
            # "Employee ID" / "NX-1234" near a name
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\(?(NX-\d+)\)?'),
            # "Team transfer: First Last is moving..."
            re.compile(r'(?:Team transfer|transfer|Transfer)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)', re.I),
            # "Directory entry: First Last"
            re.compile(r'(?:Directory entry|directory)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)', re.I),
            # "First Last — Role on Team"
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+[-—]\s+[A-Z][a-z]'),
            # "update from First Last:"
            re.compile(r'(?:from|by)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*:|,)', re.I),
            # "allergic to X" / "allergy" near a name
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is\s+)?(?:allergic|has\s+an?\s+allergy)', re.I),
            # "First Last's pet/hobby/birthday"
            re.compile(r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:'s?\s+)(?:pet|hobby|hobbies|birthday|allergy|allergies|diet|favorite|seat|desk)", re.I),
            # "First Last mentioned/said/brought/shared"
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:mentioned|said|brought|shared|announced|reported|confirmed|noted)', re.I),
        ]
        
        # v2.3: Employee ID pattern
        self.employee_id_pattern = re.compile(r'(NX-\d{3,5})')
        
        # v2.3b: Personal fact patterns (allergy, hobby, pet, seat, diet)
        self.personal_fact_patterns = [
            # "X is allergic to Y"
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is\s+allergic\s+to\s+(\w+(?:\s+\w+)?)', re.I), 'allergy'),
            # "X has a Y allergy"  
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+has\s+(?:a|an)\s+(\w+)\s+allergy', re.I), 'allergy'),
            # "X's pet/dog/cat is named Y" or "X's Y (breed)"
            (re.compile(r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:'s?\s+)(?:pet|dog|cat|bird|fish|rabbit|hamster)\s+(?:is\s+)?(?:named\s+)?(\w+)", re.I), 'pet'),
            # "X brought Y (their dog/cat)"
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+brought\s+(\w+)\s+\((?:their|his|her)\s+(?:dog|cat|pet)', re.I), 'pet'),
            # "X sits at Y" / "X's desk is Y"
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:sits?\s+(?:at|in)|desk\s+(?:is|at))\s+([A-Z0-9][\w-]+)', re.I), 'seat'),
            # "X enjoys/likes Y" / "X's hobby is Y"
            (re.compile(r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:'s?\s+hobb(?:y|ies)\s+(?:is|are|include)\s+)(\w+(?:\s+\w+)?)", re.I), 'hobby'),
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:enjoys?|likes?|into|loves?)\s+(\w+(?:\s+\w+)?)', re.I), 'hobby'),
            # "X joined on DATE" / "X is joining... starting DATE"
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is\s+)?joining\s+.*?(?:starting|on)\s+(\d{4}-\d{2}-\d{2})', re.I), 'join_date'),
            # "X from CITY" / "X based in CITY"
            (re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+).*?(?:from|based in|lives in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.I), 'hometown'),
        ]
        
        # Fiction character patterns (v2.2)
        self.fiction_patterns = [
            # "characters: X, Y" or "main characters: X and Y"
            re.compile(r'(?:character|protagonist|antagonist|villain|hero)s?\s*(?:include|are|:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
            # "X, a former police officer" / "X, a criminal"  
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:a\s+)?(?:former\s+)?(?:police|intelligence|criminal|operative|agent|assassin|hacker|officer)', re.I),
            # "known as X" / "figure such as X"
            re.compile(r'(?:known as|figures? such as|called)\s+([A-Z][a-z]+(?:(?:\s*,\s*|\s+and\s+)[A-Z][a-z]+)*)', re.I),
            # "organization known as X"
            re.compile(r'(?:organization|group|syndicate|gang|cartel)\s+(?:known as|called)\s+(?:the\s+)?([A-Z][A-Za-z\s]+?)(?:\.|,|$)', re.I),
        ]
    
    def extract(self, text: str, file_path: str = '') -> List[Dict]:
        """Extract entities from a text chunk."""
        entities = []
        seen = set()
        
        def add(etype, name, value):
            key = (etype, name.lower(), value.lower()[:50])
            if key not in seen:
                seen.add(key)
                entities.append({
                    'entity_type': etype,
                    'entity_name': name.strip(),
                    'entity_value': value.strip(),
                })
        
        # IP addresses
        for m in self.patterns['ip_address'].finditer(text):
            line = text[max(0, text.rfind('\n', 0, m.start())):text.find('\n', m.end())]
            label = line.split(':')[0].strip(' -•*') if ':' in line else 'ip'
            add('ip', label, m.group(1))
        
        for m in self.patterns['ip_bare'].finditer(text):
            ip = m.group(1)
            parts = ip.split('.')
            if all(0 <= int(p) <= 255 for p in parts) and not any(e['entity_value'] == ip for e in entities):
                add('ip', 'ip_address', ip)
        
        # Emails
        for m in self.patterns['email'].finditer(text):
            add('email', 'email', m.group(1))
        
        # Phone numbers
        for m in self.patterns['phone'].finditer(text):
            phone = m.group(1)
            # Filter out likely non-phone numbers (version numbers, etc)
            digits = re.sub(r'\D', '', phone)
            if len(digits) == 10:
                add('phone', 'phone', phone)
        
        # URLs
        for m in self.patterns['url'].finditer(text):
            url = m.group(1).rstrip('.,;)')
            if not any(skip in url for skip in ['github.com/openclaw', 'docs.openclaw']):
                add('url', 'url', url)
        
        # Passwords & tokens
        for m in self.patterns['password'].finditer(text):
            add('credential', 'password', m.group(1))
        for m in self.patterns['token'].finditer(text):
            add('credential', 'token', m.group(1))
        
        # EIN, SSN
        for m in self.patterns['ein'].finditer(text):
            add('business', 'EIN', m.group(1))
        for m in self.patterns['ssn'].finditer(text):
            add('pii', 'SSN', m.group(1))
        
        # Physical addresses
        for m in self.patterns['address'].finditer(text):
            add('location', 'address', m.group(1))
        
        # Dates
        for m in self.patterns['date'].finditer(text):
            add('date', 'date', m.group(1))
        
        # Money amounts
        for m in self.patterns['money'].finditer(text):
            # Get context
            start = max(0, m.start() - 40)
            context = text[start:m.start()].strip().split('\n')[-1]
            label = context[-30:] if context else 'amount'
            add('financial', label, m.group(0))
        
        # Key-value pairs
        for m in self.kv_pattern.finditer(text):
            key = m.group(1).strip()
            value = m.group(2).strip()
            if len(key) < 50 and len(value) < 200:
                key_lower = key.lower()
                if any(w in key_lower for w in ['password', 'token', 'key', 'secret', 'credential']):
                    etype = 'credential'
                elif any(w in key_lower for w in ['ip', 'port', 'host', 'url', 'ssh', 'vm', 'server']):
                    etype = 'config'
                elif any(w in key_lower for w in ['name', 'age', 'birthday', 'phone', 'email', 'address', 
                                                    'brother', 'sister', 'son', 'daughter', 'wife', 'husband',
                                                    'father', 'mother', 'family', 'spouse']):
                    etype = 'person'
                elif any(w in key_lower for w in ['ein', 'llc', 'business', 'company', 'product', 'revenue',
                                                    'sales', 'profit', 'carrier', 'agency']):
                    etype = 'business'
                elif any(w in key_lower for w in ['da', 'dr', 'seo', 'ranking', 'keyword', 'traffic',
                                                    'backlink', 'domain']):
                    etype = 'seo'
                elif any(w in key_lower for w in ['model', 'version', 'cron', 'script', 'command', 'path']):
                    etype = 'config'
                else:
                    etype = 'fact'
                add(etype, key, value)
        
        # Family relationships (expanded)
        for pattern in self.relationship_patterns:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                # Get the relationship type from the match context
                match_text = m.group(0).lower()
                for rel in ['brother', 'sister', 'daughter', 'son', 'wife', 'husband', 
                           'father', 'mother', 'uncle', 'aunt', 'cousin', 'partner',
                           'girlfriend', 'boyfriend', 'fiancee', 'fiance', 'ex',
                           'nephew', 'niece', 'grandpa', 'grandma', 'grandfather', 'grandmother']:
                    if rel in match_text:
                        add('person', rel, name)
                        break
                else:
                    add('person', 'family', name)
        
        # Ages
        for m in self.age_patterns:
            for match in m.finditer(text):
                age = match.group(1)
                # Try to find who the age belongs to
                start = max(0, match.start() - 60)
                context = text[start:match.start()]
                # Look for a name before the age
                name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', context)
                if name_match:
                    add('person', f'{name_match.group(1)}_age', age)
                else:
                    add('fact', 'age', age)
        
        # Roles/occupations
        for pattern in self.role_patterns:
            for m in pattern.finditer(text):
                groups = m.groups()
                if len(groups) >= 2:
                    person = groups[0].strip()
                    role = groups[1].strip()
                    if len(person) < 40 and len(role) < 60:
                        add('person', f'{person}_role', role)
        
        # States mentioned (insurance context)
        states_found = set()
        for m in self.state_pattern.finditer(text):
            state = m.group(1)
            if state not in states_found:
                states_found.add(state)
                add('location', 'state', state)
        
        # Insurance products
        for pattern in self.product_patterns:
            for m in pattern.finditer(text):
                add('business', 'insurance_product', m.group(0))
        
        # Fiction characters and organizations (v2.2)
        for pattern in self.fiction_patterns:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                # Split on commas/and for lists
                names = re.split(r'\s*,\s*|\s+and\s+', name)
                for n in names:
                    n = n.strip()
                    if len(n) > 1 and len(n) < 40:
                        add('fiction', 'character', n)
        
        # Note: For project-specific fiction character names, users can extend
        # this by adding known names to a config file or environment variable.
        
        # v2.3: Person name extraction for conversation data
        for pattern in self.person_name_patterns:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                if len(name) > 3 and len(name) < 50:
                    add('person', 'person_name', name)
                    # Check for employee ID in same match
                    if m.lastindex and m.lastindex >= 2:
                        try:
                            emp_id = m.group(2)
                            if emp_id:
                                add('person', f'{name}_employee_id', emp_id)
                        except IndexError:
                            pass
        
        # v2.3: Extract employee IDs
        for m in self.employee_id_pattern.finditer(text):
            emp_id = m.group(1)
            # Try to find who it belongs to by looking nearby
            start = max(0, m.start() - 80)
            context = text[start:m.start()]
            name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', context)
            if name_match:
                add('person', f'{name_match.group(1)}_employee_id', emp_id)
            else:
                add('person', 'employee_id', emp_id)
        
        # v2.3b: Extract personal facts (allergy, pet, seat, hobby, join date, hometown)
        for pattern, fact_type in self.personal_fact_patterns:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                value = m.group(2).strip()
                if len(name) > 3 and len(value) > 1:
                    add('person', f'{name}_{fact_type}', value)
        
        # v2.3: Extract full names from "First Last" patterns in structured data
        # Match "First Last" when followed by role/team indicators
        for m in re.finditer(r'([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,20})\s+(?:is|was|will|has|from|on the|at|—|-)', text):
            name = m.group(1).strip()
            # Filter out false positives (common phrases that look like names)
            false_positives = {'New Hire', 'Team Transfer', 'Directory Entry', 'Project Mercury',
                             'Account Review', 'Sprint Planning', 'Security Credential',
                             'Date Monday', 'Date Tuesday', 'Date Wednesday', 'Date Thursday',
                             'Date Friday', 'Date Saturday', 'Date Sunday'}
            if name not in false_positives and len(name) > 4:
                add('person', 'person_name', name)
        
        return entities
