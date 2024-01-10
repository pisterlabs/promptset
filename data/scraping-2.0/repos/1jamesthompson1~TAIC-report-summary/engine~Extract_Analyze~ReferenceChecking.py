from enum import Enum
import re, copy
from .ReportExtracting import ReportExtractor
from ..OpenAICaller import openAICaller

class ReferenceType(Enum):
    """
    Enum for the type of reference. These can either be a citation or a quote.
    """
    citation = "citation"
    quote = "quote"

class Reference():
    """
    Reference object used as a helper in the ReferenceValidator class.
    """
    def __init__(self, text: str, reference_str: str, type: ReferenceType):
        self.text = text # This is the text that is being referenced
        self.reference_str = reference_str
        self.reference = self._parse_reference(reference_str) # This is the reference pointing to a partiuclar section, paragraph, or subparagraph.
        self.type = type
        self.validated = False
        self.invalid = False
        self.updated = False
        self.old_reference = None
        self.unrepairable = False

    def set_reference(self, reference_str: str):
        """
        Sets the reference to the given reference string.
        """
        self.reference_str = reference_str
        self.reference = self._parse_reference(reference_str)
    
    def _parse_reference(self, reference_str: str):
        """
        Parses the given reference into a list sections

        The references may be in three forms
        5.3, 5.6, 5.7
        5.3-5.7
        5.3

        It will be parsed into a list of references like [5.3, 5.6, 5.7], for the ranges it will expanded.
        """
        reference = list(map(lambda str: str.strip(), reference_str.split(',')))
        reference_is_range = reference_str.find('-') != -1

        if reference_is_range:
            reference = list(map(lambda str: str.strip(), reference_str.split('-')))
            # Expand the range

            start_section, end_section = reference
            try:
                start_section = list(map(int, start_section.split('.')))
                end_section = list(map(int, end_section.split('.')))
            except:
                self.set_invalid()
                return None

            if len(start_section) == 1:
                start_section += [0, 0]
            elif len(start_section) == 2:
                start_section += [0]
            if len(end_section) == 1:
                end_section += [0, 0]
            elif len(end_section) == 2:
                end_section += [0]

            start_section, start_paragraph, start_subparagraph = start_section
            end_section, end_paragraph, end_subparagraph = end_section

            if start_section == end_section and start_paragraph == end_paragraph:
                reference = [f"{start_section}.{start_paragraph}.{start_subparagraph + i}" for i in range(end_subparagraph - start_subparagraph + 1)]
            elif start_section == end_section:
                reference = [f"{start_section}.{start_paragraph + i}" for i in range(end_paragraph - start_paragraph + 1)]

        if reference is None:
            self.set_invalid()
            return None
        return reference

    def set_validated(self):
        """
        Sets the validated value of the reference.
        """
        self.validated = True

    def set_repaired(self, new_reference):
        """
        Sets the repaired value of the reference.
        """
        self.old_reference = copy.deepcopy(self)
        self.updated = True

        self.set_reference(new_reference.reference_str)
        self.text = new_reference.text
        self.type = new_reference.type

        self.set_validated()

    def set_unrepairable(self):
        """
        Sets the unrepairable value of the reference.
        """
        self.unrepairable = True

    def set_invalid(self):
        """
        Sets the invalid value of the reference. This is to be used when it is ill formed.
        """
        self.invalid = True

    def to_string(self):
        """
        Returns a string representation of the reference.
        """
        if self.invalid:
            match self.type:
                case ReferenceType.quote:
                    return f'''"{self.text}" ({self.reference_str} (invalid formatted quote))'''
                case ReferenceType.citation:
                    return f"{self.text} ({self.reference_str} (invalid formatted citation)))"
        elif self.unrepairable:
            match self.type:
                case ReferenceType.quote:
                    return f'''"{self.text}" ({self.reference_str} (unvalidated and unrepairable quote))'''
                case ReferenceType.citation:
                    return f"{self.text} ({self.reference_str} (unvalidated and unrepairable citation)))"
        match self.type:
            case ReferenceType.quote:
                return f'''"{self.text}" ({self.reference_str})'''
            case ReferenceType.citation:
                return f"{self.text} ({self.reference_str})"
        

class ReferenceValidator():
    """
    Can be used to check if the references in a section of text are valid or not but comparing them to the original text.
    """
    def __init__(self, original_text: str, debug=False):
        self.original_text = original_text
        self.debug = debug
        self.quote_repairer = QuoteRepairer(self, self.debug)
        self.reference_regex = '''("([^"]+)" {0,2}\((\d+\.\d+(?:\.\d{1,2})?)\))|(([^."]+)\(((?:\d+\.\d+(?:\.\d{1,2})?)(?:, \d+\.\d+(?:\.\d{1,2})?)*(?: ?- ?\d+\.\d+(?:\.\d{1,2})?)?)\))'''

    def _print(self, message):
        """
        Prints the given message if debug is set to true.
        """
        if self.debug:
            print(message)

    def validate_references(self, text) -> [Reference]:
        """
        Checks if all of the references are valid or not. A single wrong references will return false for the whole text.

        Returns a tuple of the processed text and the number of references that were validated.  
        Returns None if there is any unrepairable references.
        """
        text = text.replace("\n", "")

        try: 
            references = self._extract_references(text)
        except Exception as e:
            return "Invalid format"
        
        if references is None:
            self._print(f"   No references found")
            return None
        updated_references_counter = 0
        for reference in references:
            if not reference.invalid:
                processed_reference = self._validate_reference(reference, True)
            else:
                processed_reference = reference
            quote_regex =  fr'''"{processed_reference.text}" {{0,2}}\({processed_reference.reference_str}\)'''

            citation_regex = fr'''{processed_reference} {{0,2}}\({processed_reference.reference_str}\)'''

            if processed_reference.unrepairable or processed_reference.invalid:
                print(f"   Invalid {reference.type}: {reference.reference_str} for text {reference.text}")
                if processed_reference.type == ReferenceType.citation:
                    text = re.sub(citation_regex, processed_reference.to_string(), text, flags=re.IGNORECASE)
                elif processed_reference.type == ReferenceType.quote:
                    text = re.sub(quote_regex, processed_reference.to_string(), text, flags=re.IGNORECASE)
                updated_references_counter += 1
            if processed_reference.updated and processed_reference.type == ReferenceType.quote:
                self._print(f"  Fixed reference: {processed_reference.old_reference.reference_str} to {processed_reference.reference_str} for text {processed_reference.text}")
                quote_regex =  fr'''"{processed_reference.text}" {{0,2}}\({processed_reference.old_reference.reference_str}\)'''
                text = re.sub(quote_regex, processed_reference.to_string(), text, flags=re.IGNORECASE)
                updated_references_counter += 1

        return text, len(references), updated_references_counter

    def _extract_references(self, text) -> [Reference]:
        """
        Extracts all the references from the given text and returns a list of them.
        """

        # Firstly make sure that it is using a valid convention for the references
        # It has been noted that it could repsond will all citation formatted like ("3.54").

        invalid_reference_regex = r'\("\d.\d{1,2}(.\d{1,2})?"\)'
        if re.search(invalid_reference_regex, text):
            self._print(f"""  Reference formatted with ("3.45") style which is not allowed.""")
            raise Exception("Invalid reference format")

        references = []
        for match in re.finditer(self.reference_regex, text.lower()):
            new_referenece = None
            if match.group(2) and match.group(3):
                quote = match.group(2).lower()
                new_referenece = Reference(quote, match.group(3), ReferenceType.quote)
            elif match.group(5) and match.group(6):
                new_referenece = Reference(match.group(5), match.group(6), ReferenceType.citation)
            else:
                raise Exception(f"Invalid reference format see {match.group(0)}")

            references.append(new_referenece) 

        if len(references) == 0:
            self._print(f"  Cant find any references in {text}")
            return None
        return references

    def _validate_reference(self, reference: Reference, attempt_repair: bool):
        """
        Checks if the given reference is valid or not.
        """
        reportExtractor = ReportExtractor(self.original_text, "Not known")
        if reference.invalid:
            print(f"   Invalid reference (note this wa a bug fix and the program shouldnt ever make it here.) {reference.to_string()}")
            return reference
        source_sections = list(map(lambda reference: reportExtractor.extract_section(reference), reference.reference))

        if all(v is None for v in source_sections):
            reference.set_invalid()
            if attempt_repair:
                return reference
            return False
        
        # remove all non source sections
        source_sections = list(filter(lambda section: section is not None, source_sections))
        
        source_sections = "\n".join(map(lambda str: str.replace("\n", "").lower(), source_sections)).lower()

        match reference.type:
            case ReferenceType.citation:
                return self._validate_citation(reference, source_sections, attempt_repair)
            case ReferenceType.quote:
                return self._validate_quote(reference, source_sections, attempt_repair)
            case _:
                raise Exception("Invalid reference type")

    def _validate_citation(self, citation: Reference, source_section: str, attempt_repair: bool):
        """
        Checks if the given citation is valid or not. Uses a llm to see if the quotation makes sense.
        """
        if attempt_repair:
            self._print(f"   Validating citation: {citation.reference_str} with reference {citation.text}")
        system_message = """
You are helping me check that references are correct.

Whitespace differences are too be ignored.

You will be given a indirect quote and the source text.
Return "yes" if you think that the indirect quote is supported by the source text.
Return "no" if you cant find any evidence for the indirect quote in the source text.

Note that the indirect quote might make claims that are treated as a given fact. The indirect quote is supported if atleast 50% of it can be proven with the source text.
"""

        user_message = f"""
Here is the reference:
{citation.text}

Here is the source text:
{source_section}
""" 
        valid = openAICaller.query(
            system_message,
            user_message,
            large_model=True,
            temp = 0
        )

        if valid.lower() == "yes":
            self._print(f"    Validated citation")
            citation.set_validated()
            return citation
        elif valid.lower() != "no":
            self._print(f"""  Invalid response from model: \n"\n{valid}\n""")
        if attempt_repair:
            self._print(f"   Invalid citation couldn't be justified to have come from\n   {source_section}")
        else:
            return False
        
        fixed_citation = self.quote_repairer._find_reference_location(citation)
        if fixed_citation:
            self._print(f"   Fixed quote to be {fixed_citation.to_string()}")
            citation.set_repaired(fixed_citation)
            return citation
        
        citation.set_unrepairable()
        return citation
            


    def _validate_quote(self, quote: Reference, source_section: str, attempt_repair: bool):
        """
        Checks if the given quote is valid or not. This is done by just using Regex. If the quote cant be found in the source section then it is invalid. There may be extra problems with additional spaces that can be added into the source section by the text extraction from a pdf.
        """
        format_punctuation = lambda text: re.sub(r'''(('')|['"])''', r'''(('')|['"])''' , text).replace(",", r",?")

        if attempt_repair:
            self._print(f"   Validating quote: {quote.text} with reference {quote.reference_str}")
        quote_regex = re.compile(quote.text, re.MULTILINE | re.IGNORECASE)
        if not re.search(quote_regex, source_section) is None:
            self._print(f"   Validated quote")
            quote.set_validated()
            return quote

        # Add a opional space between each character in the quote
        quote_regex = re.compile(format_punctuation(r"\s*".join(list(quote.text.strip()))), re.MULTILINE | re.IGNORECASE)
        if not re.search(quote_regex, source_section) is None:
            self._print(f"   Validated quote with extra spaces")
            quote.set_validated()
            return quote
        
        if attempt_repair:
            self._print(f"   Invalid quote {quote.to_string()} not found in\n{source_section}")
        else:
            return False

        # There can be a tendency to get the attributue section wrong. Therefore we will check if the quote is in one of the sections either just before or just after.


        fixed_quote = self.quote_repairer._find_reference_location(quote)

        if fixed_quote:
            self._print(f"   Fixed quote to be {fixed_quote.reference_str}")
            quote.set_repaired(fixed_quote)
            return quote
        
        quote.set_unrepairable()
        return quote
    

class QuoteRepairer():
    """
    Used to help fix a reference that is broken. Either by finding the correct location, or by finding the correct quote.
    """
    def __init__(self, ReferenceValidator: ReferenceValidator,  debug=False):
        self.reference_validator = ReferenceValidator
        self.debug = debug

    def _print(self, message):
        """
        Prints the given message if debug is set to true.
        """
        if self.debug:
            print(message)


    def _find_reference_location(self, reference: Reference):
        """
        Search neighbouring sections and find where the reference was from
        """
        potential_locations = self._get_potential_locations(reference)

        self._print(f"  Potential locations: {potential_locations}")

        for location in potential_locations:
            self._print(f"   Checking location {location}")
            temp_reference = Reference(reference.text, location, reference.type)
            if self.reference_validator._validate_reference(temp_reference,False):
                return temp_reference

        self._print(f"  Couldn't find {reference.type} location")

        return False
    
    def _get_potential_locations(self, reference: Reference):
        """
        Get the two neighbourough section to a reference.
        """
        possible_locations = reference.reference
        
        sections_to_search_around = [reference.reference[0], reference.reference[-1]]

        sections_to_search_around
        for location_str in sections_to_search_around:
            parts = location_str.split('.')
            if len(parts) == 1:
                # Handling sections like "5"
                section, paragraph, subparagraph = int(parts[0]), 0, 0
                preceding_sections = map(lambda args: self._parse_section(*args) ,[(section-i, paragraph, subparagraph) for i in range(1, 3)])
                succeeding_sections = map(lambda args: self._parse_section(*args),[(section + i, paragraph, subparagraph) for i in range(1, 3)])
                jump_to_next = (section + 1 ,0, 0)
            elif len(parts) == 2:
                # Handling sections like "5.2"
                section, paragraph, subparagraph = list(map(int, parts)) + [0]
                preceding_sections = map(lambda args: self._parse_section(*args) ,[(section, paragraph  - i, subparagraph) for i in range(1, 3)])

                succeeding_sections = map(lambda args: self._parse_section(*args),[(section, paragraph + i, subparagraph) for i in range(1, 3)])
                jump_to_next = (section + 1, 1, 0)
            elif len(parts) == 3:
                # Handling sections like "5.2.1"
                section, paragraph, subparagraph = list(map(int, parts))
                preceding_sections = map(lambda args: self._parse_section(*args) ,[(section, paragraph, subparagraph - i) for i in range(1, 3)])

                succeeding_sections = map(lambda args: self._parse_section(*args),[(section, paragraph, subparagraph + i) for i in range(1, 3)])
                jump_to_next = (section, paragraph+1, 1)
            else:
                # Invalid input
                return []
            
            possible_locations.append(self._parse_section(*jump_to_next))
            possible_locations.append(self._parse_section(section+1, paragraph, subparagraph))
            possible_locations.append(self._parse_section(section-1, paragraph, subparagraph))
            possible_locations.extend(list(preceding_sections))
            possible_locations.extend(list(succeeding_sections))

        return list(dict.fromkeys(possible_locations))

    def _parse_section(self, section, paragraph, subparagraph):
        if subparagraph < 0:
            subparagraph = 0
            paragraph -= 1
        if paragraph < 0:
            paragraph = 0
            section -= 1
        if section < 0:
            return None

        formatted_reference = "{}.{}.{}".format(section, paragraph, subparagraph)

        return re.sub(r'\.0(?!.)', "", formatted_reference)
